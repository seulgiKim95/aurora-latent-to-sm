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

start_date, end_date = '2015-04-01', '2025-11-30'
rollout_steps = [20, 40, 80, 120]
nbootstrap = 2000
model_name = 'unet'
nworkers = 8

platform = 'cpu'
if platform == 'k8s':
    cpuserver_data = '/home/jovyan/cpuserver/'
    nas2 = '/home/jovyan/data2'
elif platform == 'cpu':
    cpuserver_data = '/data/'
    nas2 = '/home/seulgi/data2/'


if model_name == 'mlp':
    BASE_PATH = nas2 + f'/Foundation_Model/Aurora/mlp/rollout/rollout_t+1_06_epoch25/'
    suffix = '_00_R45_swvl1_precipitation'
elif model_name == 'unet':
    BASE_PATH = nas2 + f'/Foundation_Model/Aurora/unet/epoch10/'
    suffix = '_00_R30_swvl1'
elif model_name == 'mlp_enc':
    BASE_PATH = nas2 + f'/Foundation_Model/Aurora/mlp_enc/epoch25/'
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
    target_mmap = np.memmap(BASE_PATH + '/ERA5_swvl1.bin', dtype=np.float32, mode='r+', shape=(total_steps, 720, 1440))
    return target_mmap

print('load land_sea_mask...')

def load_lsm():
    file = cpuserver_data + 'personal_data/project_aurora/static/2025_static.nc'
    with xr.open_dataset(file) as stat_ds:
        lsm = stat_ds['lsm'].values.squeeze()[:720, :] >= 0.5
    return lsm

def load_coords():
    file = cpuserver_data + 'personal_data/project_aurora/static/2025_static.nc'
    with xr.open_dataset(file) as stat_ds:
        lat = stat_ds['latitude'][:720]
        lon = stat_ds['longitude']
    return lat, lon


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

def compute_daily_2d_mean(args):
    i, date, BASE_PATH, suffix, target_meta, lsm, rollout_steps = args
    
    date_filekey = date.strftime('%Y%m%d')
    file = f"{BASE_PATH}/{date.year}/{date_filekey}{suffix}.nc"

    try:
        target_mmap = load_target_mmap(target_meta)

        with xr.open_dataset(file) as ds:
            time_var = 'valid_time' if 'valid_time' in ds.variables else 'time'

            p = ds['swvl1'].isel({time_var: rollout_steps}).values
            times = ds[time_var].isel({time_var: rollout_steps}).values

            idx = target_meta.loc[times, 'mmap_idx'].values
            o = target_mmap[idx]
            o = np.where(lsm[None], o, np.nan)

            resid = (p - o)**2 # [len(rollout_steps), 720, 1440)
        return i, resid
    except Exception as e:
        print(f'{date.strftime("%Y-%m-%d")} fail: {e}')
        return i, None


if __name__ == '__main__':
    target_meta = load_target_meta()
    lsm = load_lsm()

    total_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    tasks = [(i, d, BASE_PATH, suffix, target_meta, lsm, np.array(rollout_steps) -1) for i, d in enumerate(total_dates)]

    # accumulator array for 2D maps over all dates (720, 1440)
    final_sum_2d = np.zeros((720, 1440), dtype=np.float32)
    valid_counts = np.zeros((720, 1440), dtype=np.int32)

    sum_resid = np.zeros((len(rollout_steps), 720, 1440), dtype=np.float64)
    count_resid = np.zeros((len(rollout_steps), 720, 1440), dtype=np.int32)

    print(f"Generating 2D Residual Map (Mean of {rollout_steps} steps)...")
    with ProcessPoolExecutor(
        max_workers=nworkers,
    ) as executor:
#       results = list(tqdm.tqdm(executor.map(compute_daily_2d_mean, tasks), total=len(tasks))) # (len(rollout_steps), 720, 1440) * len(total_dates)
        for i, res in tqdm.tqdm(executor.map(compute_daily_2d_mean, tasks), total=len(tasks)):
            if res is not None:
                mask = ~np.isnan(res)
                sum_resid[mask] += res[mask]
                count_resid[mask] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        resid = np.divide(sum_resid, count_resid).astype(np.float32)
        resid = np.sqrt(resid)

    print('saving...')

    lat, lon = load_coords()
    data_vars = {}
    encoding = {}
    for i, rollout_step in enumerate(rollout_steps):
        key = f'swvl1_{rollout_step}'
        data_vars[key] = (['lat', 'lon'], resid[i])
        encoding[key] = {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
    ds_out = xr.Dataset(
        data_vars = data_vars,
        coords = dict(lat=lat, lon=lon),
    )
    encoding['lat'] = {"zlib": True, "complevel": 4},
    encoding['lon'] = {"zlib": True, "complevel": 4},
    file_out = f'./{model_name}_rmse_map.nc'
    ds_out.to_netcdf(file_out, mode='w', encoding=encoding)
    ds_out.close()
    print(f"Success: {file_out} saved.")
