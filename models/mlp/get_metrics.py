user = 'jovyan'

import pandas as pd
import xarray as xr
import numpy as np
import datetime
import tqdm
from train_hydr import parse_datetime
import gc
from sklearn.utils import resample

from train_hydr import load_nc_list
from train_hydr import load_data
from torch.utils.data import DataLoader
from train_hydr import aurora_collate_fn
from train_hydr import AuroraDataset
from train_hydr import AuroraDatasetLiteDecoder
from torch.utils.data import Subset

import os
from omegaconf import OmegaConf
# All paths come from train_hydr/config.yaml (single source of truth).
_cfg = OmegaConf.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_hydr", "config.yaml"))

start_date, end_date = '2015-04-01', '2015-06-30'
total_rollout_step = 120
model_name = 'mlp'

# ---------- load target --------------
print('load target data...')

target_meta = pd.read_csv('swvl1_meta.csv', parse_dates=['time'])
target_meta = target_meta.set_index('time')

total_steps = len(target_meta)
target_mmap = np.memmap('swvl1_10years.bin', dtype=np.float32, mode='r+', shape=(total_steps, 720, 1440))

# --------------- load climatology --------------
print('load climatology...')

def adjust_longitude(value, lon):
    adjust_lon = lon.flatten() < 0 # adjust longitude to 0~360
    if int(adjust_lon.sum()) > 0:
        adjust_idx = int(np.argwhere(adjust_lon)[-1].item() + 1)
        if len(value.shape) == 1:
            value = np.roll(value, shift=adjust_idx, axis=-1)
        else:
            value = np.roll(value, shift=adjust_idx, axis=1)
        # value = np.ascontiguousarray(value)
    return value

climatology_file = _cfg.paths.climatology_file
with xr.open_dataset(climatology_file) as climate_ds:
    tmp = climate_ds['longitude'].values
    climatology = climate_ds['swvl1'].values[:, :720, :]
    climatology = adjust_longitude(climatology, tmp)


# --------------- load prediction data ------------
print('load prediction data...')

if model_name == 'mlp':
    BASE_PATH = f'{_cfg.paths.rollout_root}mlp/rollout/rollout_t+1_06_epoch25/'
    suffix = '_00_R45_swvl1_precipitation'
elif model_name == 'unet':
    BASE_PATH = f'{_cfg.paths.rollout_root}unet/rollout/rollout_t+1_06_epoch10/'
    suffix = '_00_R45_swvl1'

total_dates = pd.date_range(start=start_date, end=end_date, freq='D')

shape = (len(total_dates), total_rollout_step)
daily_mse = np.zeros(shape)
daily_mae = np.zeros(shape)
daily_bias = np.zeros(shape)

daily_cov = np.zeros(shape)
daily_varp = np.zeros(shape)
daily_varo = np.zeros(shape)
for i, date in enumerate(tqdm.tqdm(total_dates)):
    start_datetime = date.strftime('%Y-%m-%d %H:%M:%S')
    date_filekey = date.strftime('%Y%m%d')
    file = f"{BASE_PATH}/{date.year}/{date_filekey}{suffix}.nc"
    with xr.open_dataset(file) as ds:
        time_var = 'valid_time' if 'valid_time' in ds.variables else 'time'
        p = ds['swvl1'].isel({time_var: slice(0, total_rollout_step)}).values.copy()

        time = ds[time_var].isel({time_var: 0}).values.copy()
        idx = target_meta.loc[time].item()
        o = target_mmap[idx:idx + total_rollout_step]

        doy = min(int(pd.to_datetime(time).strftime('%j')), 365) -1
        c = np.roll(climatology, -doy, axis=0)[:total_rollout_step]

    daily_mse[i, :] = np.nanmean((p - o)**2, axis=(1, 2))
    daily_mae[i, :] = np.nanmean(np.abs(p - o), axis=(1, 2))
    daily_bias[i, :] = np.nanmean(p - o, axis=(1, 2))

    anom_p, anom_o = p - c, o - c
    daily_cov[i, :] = np.nanmean(anom_p * anom_o, axis=(1, 2))
    daily_varp[i, :] = np.nanmean(anom_p**2, axis=(1, 2))
    daily_varo[i, :] = np.nanmean(anom_o**2, axis=(1, 2))


# ---------------------- bootstrap ---------
print('start bootstraping...')

rollout_steps = np.arange(1, total_rollout_step +1)
nsample = len(total_dates) # 30
nbootstrap = 5

save_file = './metrcis.ascii'
with open(save_file, 'w') as wf:
    wf.write('# ID\tMODEL\tMETRIC\t' + '\t\t'.join(rollout_steps.astype(str)) + '\n')
    wf.write('# ' + '-'*982 + '\n')
    
for bootstrap in tqdm.tqdm(range(nbootstrap)):
    idx = np.random.choice(len(total_dates), len(total_dates), replace=True)
    mae = np.mean(daily_mae[idx], axis=0)
    rmse = np.sqrt(np.mean(daily_mse[idx], axis=0))
    bias = np.mean(daily_bias[idx], axis=0)
    acc = np.mean(daily_cov[idx], axis=0) / (np.sqrt(np.mean(daily_varp[idx], axis=0)) * np.sqrt(np.mean(daily_varo[idx], axis=0)))

    # ------------- save metrics -------------
    with open(save_file, 'a') as wf:
        wf.write(f'{bootstrap}\t{model_name.upper()}\tACC\t' + '\t'.join([f'{d:.8f}' for d in acc]) + '\n')
        wf.write(f'{bootstrap}\t{model_name.upper()}\tMAE\t' + '\t'.join([f'{d:.8f}' for d in mae]) + '\n')
        wf.write(f'{bootstrap}\t{model_name.upper()}\tRMSE\t' + '\t'.join([f'{d:.8f}' for d in rmse]) + '\n')
        wf.write(f'{bootstrap}\t{model_name.upper()}\tBIAS\t' + '\t'.join([f'{d:.8f}' for d in bias]) + '\n')

    gc.collect()
    
    print(f'bootstrap: {bootstrap +1} / {nbootstrap} done')
