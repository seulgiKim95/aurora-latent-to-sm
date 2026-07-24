import numpy as np
import pandas as pd
import tqdm
import xarray as xr
import os
import sys
sys.path.append('../')
import joblib
import gc
import numba
numba.set_num_threads(64)
import warnings
warnings.filterwarnings('ignore')


rollout_int = 30
rollout = f'{rollout_int}D'
start_date = f'2015-04-{rollout_int:02d} 00:00:00'
lulc_list = range(1, 17)
nodThp = 2.5
min_nodTh = 100
max_nodTh = 100
#nodTh = 
corrTh = 0

path = '/data/python_modules/subin/project_TCA/Results_FM_SMAP_ASCAT/'
file1 = path + f'FM_SM_3d_2015_2025_{rollout}.npy'
file2 = path + 'SMAP_SM_3d_2015_2025.npy'
file3 = path + 'ASCAT_SM_3d_2015_2025.npy'


# load dataset
data1 = np.load(file1, mmap_mode='r')
data2 = np.load(file2, mmap_mode='r')
data3 = np.load(file3, mmap_mode='r')

st, en = 0, data1.shape[-1] #365*4
def load_f32(path):
    mm = np.load(path, mmap_mode='r')
    if st == 0 and en == data1.shape[-1]:
        return mm.astype(np.float32)
    else:
        return mm[:, :, st:en].astype(np.float32)

FM, SMAP, ASCAT = joblib.Parallel(n_jobs=3, prefer="threads")(
    joblib.delayed(load_f32)(f) for f in [file1, file2, file3]
)

# datetime
start = pd.to_datetime(start_date) + pd.Timedelta(hours=st*6)
end = pd.to_datetime(start_date) + pd.Timedelta(hours=en*6)
datetimes = pd.date_range(start=start, end=end, freq='6h')[:-1]


# LULC masking
file = '/data/personal_data/project_aurora/static/2025_static.nc'
import xarray as xr
import pandas as pd
import HydroAI.Plot as hPlot

with xr.open_dataset(file) as ds:
    lat_target = ds.latitude.values[:600]
    lon_target = ds.longitude.values[:1440]

lon_target, lat_target = np.meshgrid(lon_target, lat_target)

lat_weights_2d = np.cos(np.deg2rad(lat_target)).astype(np.float32)
np.clip(lat_weights_2d, 0.0, None, out=lat_weights_2d)
w_flat = lat_weights_2d.reshape(-1)


hemi_masks_2d = {
    'NH': lat_target >= 0.0,
    'SH': lat_target < 0.0
}
hemi_list = ['NH', 'SH']


# load lulc
file = './lulc.npy'
lulc = np.load(file)
file = './lulc_meta.npy'
lulc_meta = np.load(file, allow_pickle=True).item()
for k, v in lulc_meta.items():
    lulc_meta[k] = v.replace('/', '_')


# del variables and garbage collect
def gc_collect():
    var = ['FM_lulc', 'SMAP_lulc', 'ASCAT_lulc', 'FM_conti', 'SMAP_conti', 'ASCAT_conti', 'FM_clean', 'SMAP_clean', 'ASCAT_clean', 'FM_ano', 'SMAP_ano', 'ASCAT_ano', 'W_clean']
    for v in var:
        if v in globals():
            del globals()[v]
    gc.collect()
    return


# lulc_masking
@numba.njit(parallel=True)
def lulc_masking(A, B, C, condition):
    rows, cols, depth = A.shape
    A_out = A.copy()
    B_out = B.copy()
    C_out = C.copy()
    for i in numba.prange(rows):
        for j in range(cols):
            if not condition[i, j]:
                for k in range(depth):
                    A_out[i, j, k] = np.nan
                    B_out[i, j, k] = np.nan
                    C_out[i, j, k] = np.nan
    return A_out, B_out, C_out


# masking for collocation
def collocation_masking(FM_conti, SMAP_conti, ASCAT_conti, w_flat):
    def process_col(j, a_col, b_col, c_col):
        mask = ~np.isnan(a_col) & ~np.isnan(b_col) & ~np.isnan(c_col)
        return j, a_col[mask], b_col[mask], c_col[mask], w_flat[mask]
    n_cols = FM_conti.shape[1]

    results = joblib.Parallel(n_jobs=64, prefer="threads")(  # -1 = all cores
        joblib.delayed(process_col)(j, FM_conti[:, j], SMAP_conti[:, j], ASCAT_conti[:, j])
        for j in range(n_cols)
    )

    results.sort(key=lambda x: x[0])
    FM_clean = [r[1] for r in results]
    SMAP_clean = [r[2] for r in results]
    ASCAT_clean = [r[3] for r in results]
    W_clean = [r[4] for r in results]
    return FM_clean, SMAP_clean, ASCAT_clean, W_clean


# convert anomaly
def compute_anomaly_list(arr_list, w_list, n_jobs=os.cpu_count()):
    def _wstat_worker(cols, wcols):
        n = len(cols)
        means = np.full(n, np.nan, dtype=np.float64)
        stds  = np.full(n, np.nan, dtype=np.float64)
        for i, (x, w) in enumerate(zip(cols, wcols)):
            if x.size < 2:
                continue
            W  = w.sum()
            if W <= 0:
                continue
            W2  = (w * w).sum()
            eff = W - W2 / W                       # Bessel correction (reliability)
            if eff <= 0:
                continue
            m = (w * x).sum() / W
            v = (w * (x - m) ** 2).sum() / eff
            means[i] = m
            stds[i]  = np.sqrt(v) if v > 0 else np.nan
        return means, stds

    indices = np.array_split(np.arange(len(arr_list)), n_jobs)
    chunks  = [arr_list[idx[0]:idx[-1]+1] for idx in indices]
    wchunks = [w_list  [idx[0]:idx[-1]+1] for idx in indices]
    results = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
        joblib.delayed(_wstat_worker)(c, w) for c, w in zip(chunks, wchunks)
    )
    means = np.concatenate([r[0] for r in results])
    stds  = np.concatenate([r[1] for r in results])
    ano_list = [
        (x - m) / s if s > 0 else np.full_like(x, np.nan, dtype=np.float32)
        for x, m, s in zip(arr_list, means, stds)
    ]
    return ano_list, means, stds


# variance, covariance
def parallel_stats_list(arr_a, arr_b, arr_c, w_list, n_jobs=os.cpu_count()):
    def _wstats_worker(ca, cb, cc, cw):
        n = len(ca)
        var_a  = np.full(n, np.nan, dtype=np.float64)
        var_b  = np.full(n, np.nan, dtype=np.float64)
        var_c  = np.full(n, np.nan, dtype=np.float64)
        cov_ab = np.full(n, np.nan, dtype=np.float64)
        cov_ac = np.full(n, np.nan, dtype=np.float64)
        cov_bc = np.full(n, np.nan, dtype=np.float64)
        for i, (a, b, c, w) in enumerate(zip(ca, cb, cc, cw)):
            if a.size < 2:
                continue
            W  = w.sum()
            if W <= 0:
                continue
            W2  = (w * w).sum()
            eff = W - W2 / W
            if eff <= 0:
                continue
            ma = (w * a).sum() / W
            mb = (w * b).sum() / W
            mc = (w * c).sum() / W
            a_dm = a - ma; b_dm = b - mb; c_dm = c - mc
            wa_dm = w * a_dm                       # reuse to save computation
            var_a[i]  = (wa_dm * a_dm).sum() / eff
            var_b[i]  = (w * b_dm * b_dm).sum() / eff
            var_c[i]  = (w * c_dm * c_dm).sum() / eff
            cov_ab[i] = (wa_dm * b_dm).sum() / eff
            cov_ac[i] = (wa_dm * c_dm).sum() / eff
            cov_bc[i] = (w * b_dm * c_dm).sum() / eff
        return var_a, var_b, var_c, cov_ab, cov_ac, cov_bc

    indices  = np.array_split(np.arange(len(arr_a)), n_jobs)
    chunks_a = [arr_a [idx[0]:idx[-1]+1] for idx in indices]
    chunks_b = [arr_b [idx[0]:idx[-1]+1] for idx in indices]
    chunks_c = [arr_c [idx[0]:idx[-1]+1] for idx in indices]
    chunks_w = [w_list[idx[0]:idx[-1]+1] for idx in indices]
    results = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
        joblib.delayed(_wstats_worker)(ca, cb, cc, cw)
        for ca, cb, cc, cw in zip(chunks_a, chunks_b, chunks_c, chunks_w)
    )
    var_FM    = np.concatenate([r[0] for r in results])
    var_SMAP  = np.concatenate([r[1] for r in results])
    var_ASCAT = np.concatenate([r[2] for r in results])
    cov_FM_SMAP    = np.concatenate([r[3] for r in results])
    cov_FM_ASCAT   = np.concatenate([r[4] for r in results])
    cov_SMAP_ASCAT = np.concatenate([r[5] for r in results])
    return var_FM, var_SMAP, var_ASCAT, cov_FM_SMAP, cov_FM_ASCAT, cov_SMAP_ASCAT


# calculate SNR
def calc_SNR(var_data, var_random):
    var_signal = var_data - var_random
    return var_signal / var_random


# calculate fMSE
def calc_fMSE(SNR):
    return 1 / (1 + SNR)


# flag masking
def get_flag():
    flag = ((cov_FM_SMAP < corrTh) | (cov_FM_ASCAT < corrTh) | (cov_SMAP_ASCAT < corrTh))
    flag = flag | ~((0 <= fMSE_FM) & (fMSE_FM <= 1) & (0 <= fMSE_SMAP) & (fMSE_SMAP <= 1) & (0 <= fMSE_ASCAT) & (fMSE_ASCAT <= 1))
    flag = flag | ((sig2_FM < 0) | (sig2_SMAP < 0) | (sig2_ASCAT < 0))

    lengths = np.fromiter(map(len, FM_ano), dtype=np.int32, count=len(FM_ano))
    flag = flag | (lengths < nodTh)
    return flag


# save
def save_nc():
    time = datetimes
    ds = xr.Dataset(
        {
            # sigma
            "sigma_FM": ("time", sigma_FM),
            "sigma_SMAP": ("time", sigma_SMAP),
            "sigma_ASCAT": ("time", sigma_ASCAT),
            # SNR
            "SNR_FM":   ("time", SNR_FM),
            "SNR_SMAP":   ("time", SNR_SMAP),
            "SNR_ASCAT":   ("time", SNR_ASCAT),
            # fMSE
            "fMSE_FM":  ("time", fMSE_FM),
            "fMSE_SMAP":  ("time", fMSE_SMAP),
            "fMSE_ASCAT":  ("time", fMSE_ASCAT),
        },
        coords={"time": time},
    )

    # ── meta data ──
    ds.attrs["description"] = "Spatial Triple Collocation results"
    ds.attrs["datasets"] = "FM, SMAP, ASCAT"

    for var in ["sigma_FM", "sigma_SMAP", "sigma_ASCAT"]:
        ds[var].attrs["long_name"] = f"Error std ({var[-1]})"
        ds[var].attrs["units"] = "m3/m3"  # fix units

    for var in ["SNR_FM", "SNR_SMAP", "SNR_ASCAT"]:
        ds[var].attrs["long_name"] = f"Signal-to-Noise Ratio ({var[-1]})"
        ds[var].attrs["units"] = "-"

    for var in ["fMSE_FM", "fMSE_SMAP", "fMSE_ASCAT"]:
        ds[var].attrs["long_name"] = f"Fractional MSE ({var[-1]})"
        ds[var].attrs["units"] = "-"

    file = f'/data/personal_data/project_aurora/TCA/unet10_SM_STC/STC_{rollout}_{hemi}_{idx_lulc:02d}_nodTh{nodTh}_{lulc_meta[idx_lulc].replace(" ", "_")}.nc'
    # -- save --
    ds.to_netcdf(file, encoding={
        v: {"dtype": "float32", "zlib": True, "complevel": 4}
        for v in ds.data_vars
    })
    return


for hemi in hemi_list:
    hemi_cond_2d = hemi_masks_2d[hemi]
    print(f'----- Hemisphere: {hemi} -----')
    for idx_lulc in lulc_list:
        condition = (lulc.astype(int) == idx_lulc) & hemi_cond_2d
        nodTh = int(min(max_nodTh, max(min_nodTh, condition.sum() * (nodThp /100))))
        print(f'---')
        print(f'start idx_lulc: {idx_lulc}')
        print(f'...lulc: {lulc_meta[idx_lulc]}')
        print(f'...nodTh: {nodTh}')
#       if condition.sum() < nodTh:
#           print(f'... -> skipped (insufficient pixels)')
#           continue

        # lulc masking
        FM_lulc, SMAP_lulc, ASCAT_lulc = lulc_masking(FM, SMAP, ASCAT, condition)

        # convert contigeously
        FM_conti = FM_lulc.reshape(-1, FM.shape[-1])
        SMAP_conti = SMAP_lulc.reshape(-1, SMAP.shape[-1])
        ASCAT_conti = ASCAT_lulc.reshape(-1, ASCAT.shape[-1])

        # collocation masking
        FM_clean, SMAP_clean, ASCAT_clean, W_clean = collocation_masking(FM_conti, SMAP_conti, ASCAT_conti, w_flat)

        # convert anomaly
        FM_ano,   FM_m,   FM_s   = compute_anomaly_list(FM_clean, W_clean,    n_jobs=64)
        SMAP_ano, SMAP_m, SMAP_s = compute_anomaly_list(SMAP_clean, W_clean,  n_jobs=64)
        ASCAT_ano,ASCAT_m,ASCAT_s= compute_anomaly_list(ASCAT_clean, W_clean, n_jobs=64)

        # variance, covariance
        var_FM, var_SMAP, var_ASCAT, cov_FM_SMAP, cov_FM_ASCAT, cov_SMAP_ASCAT = parallel_stats_list(
            FM_ano, SMAP_ano, ASCAT_ano, W_clean, n_jobs=64
        )

        # calculate sig2, sig
        sig2_FM = var_FM - cov_FM_SMAP * cov_FM_ASCAT / cov_SMAP_ASCAT
        sig2_SMAP = var_SMAP - cov_FM_SMAP * cov_SMAP_ASCAT / cov_FM_ASCAT
        sig2_ASCAT = var_ASCAT - cov_FM_ASCAT * cov_SMAP_ASCAT / cov_FM_SMAP
        sigma_FM = np.sqrt(sig2_FM)
        sigma_SMAP = np.sqrt(sig2_SMAP)
        sigma_ASCAT = np.sqrt(sig2_ASCAT)

        # calculate SNR
        SNR_FM = calc_SNR(var_FM, sigma_FM**2)
        SNR_SMAP = calc_SNR(var_SMAP, sigma_SMAP**2)
        SNR_ASCAT = calc_SNR(var_ASCAT, sigma_ASCAT**2)

        # calculate fMSE
        fMSE_FM = calc_fMSE(SNR_FM)
        fMSE_SMAP = calc_fMSE(SNR_SMAP)
        fMSE_ASCAT = calc_fMSE(SNR_ASCAT)

        # flag masking
        flag = get_flag()
        print(f'...total data: {len(flag)}')
        print(f'...masking data: {len(flag) - flag.sum()}')
        sigma_FM = np.where(flag, np.nan, sigma_FM)
        sigma_SMAP = np.where(flag, np.nan, sigma_SMAP)
        sigma_ASCAT = np.where(flag, np.nan, sigma_ASCAT)
        SNR_FM = np.where(flag, np.nan, SNR_FM)
        SNR_SMAP = np.where(flag, np.nan, SNR_SMAP)
        SNR_ASCAT = np.where(flag, np.nan, SNR_ASCAT)
        fMSE_FM = np.where(flag, np.nan, fMSE_FM)
        fMSE_SMAP = np.where(flag, np.nan, fMSE_SMAP)
        fMSE_ASCAT = np.where(flag, np.nan, fMSE_ASCAT)

        # save
        save_nc()

        # gc
        gc_collect()
