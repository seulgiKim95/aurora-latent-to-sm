import pandas as pd
import xarray as xr
import numpy as np
import datetime
import tqdm
import gc
from sklearn.utils import resample

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

start_date, end_date = '2015-04-01', '2025-11-30'
#start_date, end_date = '2015-04-01', '2015-05-30'
total_rollout_step = 120
nbootstrap = 2000
block_len = 30      # block-bootstrap block length (days); use >= the rollout overlap (~30 days)
model_name = 'unet_oracle'
nworkers = 4

platform = 'cpu'
if platform == 'k8s':
    cpuserver_data = '/home/jovyan/cpuserver/'
    nas2 = '/home/jovyan/data2'
elif platform == 'cpu':
    cpuserver_data = '/data/'
    nas2 = '/home/data_2/'


if model_name == 'mlp':
    BASE_PATH = nas2 + f'/Foundation_Model/Aurora/mlp/rollout/rollout_t+1_06_epoch25/'
    suffix = '_00_R45_swvl1_precipitation'
elif model_name == 'unet':
    BASE_PATH = nas2 + f'/Foundation_Model/Aurora/unet/epoch10/'
    suffix = '_00_R30_swvl1'
elif model_name == 'mlp_enc':
    BASE_PATH = nas2 + f'/Foundation_Model/Aurora/mlp_enc/epoch25/'
    suffix = '_00_R30_swvl1'
elif model_name == 'unet_oracle':
    BASE_PATH = nas2 + f'/Foundation_Model/Aurora/unet/10years_epoch10_oracle/'
    suffix = '_00_R30_swvl1'


# ---------- load target --------------
print('load target data...')

def load_target_meta():
    BASE_PATH = cpuserver_data + '/personal_data/project_aurora/ERA5/'

    target_meta = pd.read_csv(BASE_PATH + '/ERA5_swvl1_meta.csv', parse_dates=['time'])
    target_meta = target_meta.set_index('time')
    return target_meta

def load_target_mmap(target_meta):
    BASE_PATH = cpuserver_data + '/personal_data/project_aurora/ERA5/'

    total_steps = len(target_meta)
    target_mmap = np.memmap(BASE_PATH + '/ERA5_swvl1.bin', dtype=np.float32, mode='r', shape=(total_steps, 720, 1440))
    return target_mmap


# --------------- load land sea mask --------------
print('load land_sea_mask...')

def load_lsm():
    file = cpuserver_data + 'personal_data/project_aurora/static/2025_static.nc'
    with xr.open_dataset(file) as stat_ds:
        lsm = stat_ds['lsm'].values.squeeze()[:720, :] >= 0.5
    return lsm

def load_lat_weight():
    """Compute the cos(latitude) area weights once (all files share the same grid).

    Crop the static file's latitude coordinate the same way (:600) and return (600,) weights.
    """
    file = cpuserver_data + 'personal_data/project_aurora/static/2025_static.nc'
    with xr.open_dataset(file) as stat_ds:
        lat_name = 'latitude' if 'latitude' in stat_ds.variables else 'lat'
        lat = stat_ds[lat_name].values[:600]
    return np.cos(np.deg2rad(lat)).astype(np.float32)   # (600,)


# --------------- load climatology --------------
print('load climatology...')

def load_climatology():
    def adjust_longitude(value, lon):
        adjust_lon = lon.flatten() < 0 # adjust longitude to 0-360
        if int(adjust_lon.sum()) > 0:
            adjust_idx = int(np.argwhere(adjust_lon)[-1].item() + 1)
            if len(value.shape) == 1:
                value = np.roll(value, shift=adjust_idx, axis=-1)
            else:
                value = np.roll(value, shift=adjust_idx, axis=2)
            # value = np.ascontiguousarray(value)
        return value

    climatology_file = cpuserver_data + '/personal_data/project_aurora/static/climatology_era5_swvl1.nc'
    with xr.open_dataset(climatology_file) as climate_ds:
        tmp = climate_ds['longitude'].values
        climatology = climate_ds['swvl1'].values[:, :720, :]
        climatology = adjust_longitude(climatology, tmp)
    climatology = np.repeat(climatology, 4, axis=0)
    lsm = load_lsm()
    climatology = np.where(lsm, climatology, np.nan)
    return climatology


# --------------- load prediction data ------------
print('load prediction data...')

def _wmean(x, w):
    """NaN-safe cos(latitude) area-weighted mean over axis=(1, 2).

    x: (T, lat, lon); w: (lat,) or (lat, 1) latitude weights. NaNs from the ocean
    mask are excluded from both numerator and denominator, so only land cells
    enter the area-weighted mean.

    Reduce along lon first, then apply the lat weights, avoiding float64 upcasting
    and large (T, lat, lon) temporaries (cost similar to np.nanmean).
    """
    wl = w.reshape(-1)                       # (lat,)
    valid = ~np.isnan(x)
    lon_sum = np.where(valid, x, np.float32(0)).sum(axis=2)   # (T, lat)
    lon_cnt = valid.sum(axis=2)                              # (T, lat)
    return (lon_sum @ wl) / (lon_cnt @ wl)


def compute_daily(args):
    # target_meta, climatology, and w are shared as globals (loaded in the parent,
    # inherited by forked workers) so they are not pickled per task.
    i, date, BASE_PATH, suffix, total_rollout_step = args

    # 1. set file paths and load data
    date_filekey = date.strftime('%Y%m%d')
    file = f"{BASE_PATH}/{date.year}/{date_filekey}{suffix}.nc"

    try:
        target_mmap = load_target_mmap(target_meta)

        with xr.open_dataset(file) as ds:
            time_var = 'valid_time' if 'valid_time' in ds.variables else 'time'
            p = ds['swvl1'].isel({time_var: slice(0, total_rollout_step)}).values
            time = ds[time_var].isel({time_var: 0}).values

            idx = target_meta.loc[time].item()
            o = target_mmap[idx:idx + total_rollout_step]

            doy = min(int(pd.to_datetime(time).strftime('%j')), 365) - 1
            # circular-index only the needed 120 steps instead of np.roll (full 6GB copy)
            start = doy * 4 + 1
            c = climatology[(start + np.arange(total_rollout_step)) % climatology.shape[0]]

        p = p[:, :600, :]
        o = o[:, :600, :]
        c = c[:, :600, :]

        # 2. compute results (return only the needed scalars to save memory)
        res = {
            'mse': _wmean((p - o)**2, w),
            'mae': _wmean(np.abs(p - o), w),
            'bias': _wmean(p - o, w),
            'cov': _wmean((p-c)*(o-c), w),
            'varp': _wmean((p-c)**2, w),
            'varo': _wmean((o-c)**2, w)
        }
        return i, res
    except Exception as e:
        print(f'{date.strftime("%Y-%m-%d")} fail')
        print(e)
        return i, False


def block_bootstrap_idx(n, block_len):
    """Circular block-bootstrap indices.

    Resamples contiguous blocks of block_len with wrap-around boundaries,
    preserving the temporal correlation structure. Unlike IID resampling
    (np.random.choice), adjacent samples stay tied within a block, so the
    confidence intervals are not underestimated. Sampling is with replacement,
    so a given day may be drawn several times or not at all.
    """
    if block_len <= 1:
        return np.random.choice(n, n, replace=True)
    n_blocks = int(np.ceil(n / block_len))
    starts = np.random.randint(0, n, size=n_blocks)          # resample block start points with replacement
    idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel() % n
    return idx[:n]                                            # trim to n samples


if __name__ == '__main__':
    # Load large read-only data once here; forked workers inherit (share) it.
    # Keeping it out of the task tuples avoids pickling it for every task.
    target_meta = load_target_meta()
    climatology = load_climatology()
    w = load_lat_weight()

    total_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    tasks = [
        (i, date, BASE_PATH, suffix, total_rollout_step)
        for i, date in enumerate(total_dates)
    ]

    mp_ctx = multiprocessing.get_context('fork')   # use fork explicitly so globals are inherited
    with ProcessPoolExecutor(max_workers=nworkers, mp_context=mp_ctx) as executor:
        results = list(tqdm.tqdm(executor.map(compute_daily, tasks), total=len(tasks)))
#   compute_daily(tasks[0])

    shape = (len(total_dates), total_rollout_step)
    daily_mse = np.zeros(shape)
    daily_mae = np.zeros(shape)
    daily_bias = np.zeros(shape)
    daily_cov = np.zeros(shape)
    daily_varp = np.zeros(shape)
    daily_varo = np.zeros(shape)

    valid = np.zeros(len(total_dates), dtype=bool)
    for i, res in results:
        if res:
            valid[i] = True
            daily_mse[i], daily_mae[i], daily_bias[i] = res['mse'], res['mae'], res['bias']
            daily_cov[i], daily_varp[i], daily_varo[i] = res['cov'], res['varp'], res['varo']

    # drop failed initialization dates (so all-zero rows don't contaminate the mean/bootstrap)
    valid_idx = np.where(valid)[0]
    n_failed = len(total_dates) - len(valid_idx)
    print(f'valid dates: {len(valid_idx)} / {len(total_dates)} (failed: {n_failed})')

    daily_mse = daily_mse[valid_idx]
    daily_mae = daily_mae[valid_idx]
    daily_bias = daily_bias[valid_idx]
    daily_cov = daily_cov[valid_idx]
    daily_varp = daily_varp[valid_idx]
    daily_varo = daily_varo[valid_idx]

    # ---- save pre-bootstrap daily metrics (for re-running the bootstrap without recomputation) ----
    # each array shape = (n_valid_dates, total_rollout_step); ACC can be rebuilt from cov/varp/varo.
    daily_file = f'./{model_name}_daily.npz'
    np.savez(daily_file,
             mse=daily_mse, mae=daily_mae, bias=daily_bias,
             cov=daily_cov, varp=daily_varp, varo=daily_varo,
             dates=np.array([d.strftime('%Y-%m-%d') for d in total_dates[valid_idx]]),
             rollout_steps=np.arange(1, total_rollout_step + 1))
    print(f'saved daily metrics -> {daily_file}  (shape {daily_mse.shape})')

    # ---------------------- bootstrap ---------
    print('start bootstraping...')

    rollout_steps = np.arange(1, total_rollout_step +1)
    nsample = len(valid_idx)   # number of valid (successful) dates

    save_file = f'./{model_name}_metrics.ascii'
    with open(save_file, 'w') as wf:
        wf.write('# ID\tMODEL\tMETRIC\t' + '\t\t'.join(rollout_steps.astype(str)) + '\n')
        wf.write('# ' + '-'*982 + '\n')
    
    for bootstrap in range(nbootstrap):
        idx = block_bootstrap_idx(nsample, block_len)
        mae = np.mean(daily_mae[idx], axis=0)
        rmse = np.sqrt(np.mean(daily_mse[idx], axis=0))
        bias = np.mean(daily_bias[idx], axis=0)
        acc = np.mean(daily_cov[idx], axis=0) / (np.sqrt(np.mean(daily_varp[idx], axis=0)) * np.sqrt(np.mean(daily_varo[idx], axis=0)))

        # ------------- save metrics -------------
        with open(save_file, 'a') as wf:
            wf.write(f'{bootstrap +1}\t{model_name.upper()}\tACC\t' + '\t'.join([f'{d:.8f}' for d in acc]) + '\n')
            wf.write(f'{bootstrap +1}\t{model_name.upper()}\tMAE\t' + '\t'.join([f'{d:.8f}' for d in mae]) + '\n')
            wf.write(f'{bootstrap +1}\t{model_name.upper()}\tRMSE\t' + '\t'.join([f'{d:.8f}' for d in rmse]) + '\n')
            wf.write(f'{bootstrap +1}\t{model_name.upper()}\tBIAS\t' + '\t'.join([f'{d:.8f}' for d in bias]) + '\n')

        gc.collect()
    
        if (bootstrap +1) % 100 == 0:
            print(f'bootstrap: {bootstrap +1} / {nbootstrap} done')
