"""Persistence-baseline metrics (ACC / MAE / RMSE / BIAS per lead time).

Two baselines are computed together.

1) raw persistence: p[t] = o[0]  (hold the initial raw observation)
     - forecast anomaly (p - c) = o[0] - c[t]
2) anomaly persistence: p[t] = c[t] + (o[0] - c[0])  (hold the initial anomaly)
     - c[0] = climatology at the initial time; a0 = o[0] - c[0] (constant across leads)
     - forecast anomaly (p - c) = a0

ACC = Σ(p-c)(o-c) / sqrt(Σ(p-c)² · Σ(o-c)²)  (per-date components averaged, then combined)
  - (o - c) = o[t] - c[t]   : observed anomaly  (shared by both baselines)
Alignment of observation o / climatology c and the cos(lat) weighted spatial mean match get_metrics_backup_perf.

Instead of the original np.roll (a 6 GB copy), the climatology alignment fancy-indexes
only the needed 120 steps, giving the same result at much lower cost:
  roll(clim, -(doy*4+1))[t]  ==  clim[(doy*4+1 + t) % N]
The climatology index of the initial time (IC, 00Z) is one step before target[0]=init+6h, i.e. doy*4.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
import tqdm

# --------------------- config ---------------------
start_date, end_date = '2015-04-01', '2025-11-30'   # evaluation initialization dates
total_rollout_step = 120
NWORKERS = min(16, max(1, (os.cpu_count() or 4) - 2))

platform = 'cpu'
if platform == 'k8s':
    cpuserver_data = '/home/jovyan/cpuserver/'
elif platform == 'cpu':
    cpuserver_data = '/data/'

ERA5_PATH = cpuserver_data + '/personal_data/project_aurora/ERA5/'
STATIC = cpuserver_data + 'personal_data/project_aurora/static/2025_static.nc'
CLIM_FILE = cpuserver_data + '/personal_data/project_aurora/static/climatology_era5_swvl1.nc'

NLAT, NLON, NCROP = 720, 1440, 600


# --------------------- loaders ---------------------
def load_lsm():
    with xr.open_dataset(STATIC) as ds:
        return ds['lsm'].values.squeeze()[:NLAT, :] >= 0.5


def load_lat_weight():
    with xr.open_dataset(STATIC) as ds:
        lat_name = 'latitude' if 'latitude' in ds.variables else 'lat'
        lat = ds[lat_name].values[:NCROP]
    return np.cos(np.deg2rad(lat)).astype(np.float32)       # (600,)


def load_climatology_600():
    """Return the 6-hourly climatology cropped to lat :600. (1460, 600, 1440), ocean=NaN."""
    def adjust_longitude(value, lon):
        adjust_lon = lon.flatten() < 0
        if int(adjust_lon.sum()) > 0:
            adjust_idx = int(np.argwhere(adjust_lon)[-1].item() + 1)
            value = np.roll(value, shift=adjust_idx,
                            axis=-1 if value.ndim == 1 else 2)
        return value

    with xr.open_dataset(CLIM_FILE) as ds:
        lon = ds['longitude'].values
        clim = ds['swvl1'].values[:, :NLAT, :]
        clim = adjust_longitude(clim, lon)
    clim = np.repeat(clim, 4, axis=0)                       # daily -> 6-hourly (1460,720,1440)
    lsm = load_lsm()
    clim = np.where(lsm, clim, np.nan)
    return np.ascontiguousarray(clim[:, :NCROP, :].astype(np.float32))


def _wmean(x, w):
    """NaN-safe cos(lat) weighted spatial mean (T, lat, lon) -> (T,). Same as eval."""
    wl = w.reshape(-1)
    valid = ~np.isnan(x)
    lon_sum = np.where(valid, x, np.float32(0)).sum(axis=2)
    lon_cnt = valid.sum(axis=2)
    return (lon_sum @ wl) / (lon_cnt @ wl)


# --------------------- worker ---------------------
_G = {}


def _init_worker():
    meta = pd.read_csv(ERA5_PATH + '/ERA5_swvl1_meta.csv', parse_dates=['time'])
    _G['meta'] = meta.set_index('time')
    _G['mm'] = np.memmap(ERA5_PATH + '/ERA5_swvl1.bin', dtype=np.float32,
                         mode='r', shape=(len(meta), NLAT, NLON))
    _G['clim'] = load_climatology_600()      # (1460, 600, 1440)
    _G['w'] = load_lat_weight()              # (600,)
    _G['nclim'] = _G['clim'].shape[0]


def _compute_daily(args):
    i, date = args
    try:
        time = pd.Timestamp(date)          # initialization (00Z) time
        idx = int(_G['meta'].loc[time].item())

        # persistence = hold the antecedent SM (=IC) of the initial time.
        # As in the model eval, the first target is init+6h (= target[idx+1]); no nowcast.
        ante = np.asarray(_G['mm'][idx, :NCROP, :], dtype=np.float32)[None]        # (1,600,1440)
        o = np.asarray(_G['mm'][idx + 1:idx + 1 + total_rollout_step, :NCROP, :],
                       dtype=np.float32)                                           # (120,600,1440)

        doy = min(int(time.strftime('%j')), 365) - 1
        cidx = (doy * 4 + 1 + np.arange(total_rollout_step)) % _G['nclim']
        c = _G['clim'][cidx]                                        # (120, 600, 1440), init+(t+1)*6h
        c0 = _G['clim'][(doy * 4) % _G['nclim']]                   # (600, 1440), climatology at IC (00Z)

        w = _G['w']
        ao = o - c                                                # observed anomaly (shared)

        # (1) raw persistence: forecast = o[0]
        ap = ante - c                                             # raw persistence anomaly = o[0]-c[t]
        # (2) anomaly persistence: forecast = c[t] + (o[0]-c[0])
        a0 = ante - c0                                            # initial anomaly (constant across leads)
        pa = c + a0                                               # anomaly persistence forecast field

        res = {
            'mse':  _wmean((ante - o) ** 2, w),
            'mae':  _wmean(np.abs(ante - o), w),
            'bias': _wmean(ante - o, w),
            'cov':  _wmean(ap * ao, w),
            'varp': _wmean(ap * ap, w),
            'varo': _wmean(ao * ao, w),
            'mse_a':  _wmean((pa - o) ** 2, w),
            'mae_a':  _wmean(np.abs(pa - o), w),
            'bias_a': _wmean(pa - o, w),
            'cov_a':  _wmean(a0 * ao, w),
            'varp_a': _wmean(a0 * a0, w),                         # varo is shared with raw
        }
        return i, res
    except Exception as e:
        print(f'{pd.Timestamp(date).strftime("%Y-%m-%d")} fail: {e}')
        return i, False


def main():
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    tasks = list(enumerate(dates))

    with ProcessPoolExecutor(max_workers=NWORKERS, initializer=_init_worker) as ex:
        results = list(tqdm.tqdm(ex.map(_compute_daily, tasks), total=len(tasks)))

    keys = ['mse', 'mae', 'bias', 'cov', 'varp', 'varo',
            'mse_a', 'mae_a', 'bias_a', 'cov_a', 'varp_a']
    buf = {k: [] for k in keys}
    for i, res in results:
        if res:
            for k in keys:
                buf[k].append(res[k])
    arr = {k: np.array(v) for k, v in buf.items()}          # (ndates, 120)
    n = len(arr['mse'])
    print(f'\npersistence: valid dates {n}/{len(dates)}')

    # raw persistence
    mae = arr['mae'].mean(0)
    rmse = np.sqrt(arr['mse'].mean(0))
    bias = arr['bias'].mean(0)
    acc = arr['cov'].mean(0) / (np.sqrt(arr['varp'].mean(0)) * np.sqrt(arr['varo'].mean(0)))

    # anomaly persistence (varo shared with raw)
    mae_a = arr['mae_a'].mean(0)
    rmse_a = np.sqrt(arr['mse_a'].mean(0))
    bias_a = arr['bias_a'].mean(0)
    acc_a = arr['cov_a'].mean(0) / (np.sqrt(arr['varp_a'].mean(0)) * np.sqrt(arr['varo'].mean(0)))

    lead = np.arange(1, total_rollout_step + 1)
    df = pd.DataFrame({'lead_step': lead,
                       'persistence_acc': acc,
                       'persistence_mae': mae,
                       'persistence_rmse': rmse,
                       'persistence_bias': bias,
                       'anom_persistence_acc': acc_a,
                       'anom_persistence_mae': mae_a,
                       'anom_persistence_rmse': rmse_a,
                       'anom_persistence_bias': bias_a})
    out = './persistence_metrics.csv'
    df.to_csv(out, index=False, float_format='%.8f')
    print(f'saved {out}\n')

    # summary: ACC at key leads (6h, 1d, 2d, 5d, 10d, 30d)
    # lead time of array index t = (t+1)*6h  (same as the model rollout, index 0 = 6h)
    picks = [0, 3, 7, 19, 39, 79, 119]
    labels = ['6h', '1d', '2d', '5d', '10d', '20d', '30d']
    print('[raw persistence]         [anomaly persistence]')
    print('lead    ACC      RMSE     MAE   |   ACC      RMSE     MAE')
    for p, lab in zip(picks, labels):
        print(f'{lab:>4}  {acc[p]:.4f}  {rmse[p]:.5f}  {mae[p]:.5f}  |  '
              f'{acc_a[p]:.4f}  {rmse_a[p]:.5f}  {mae_a[p]:.5f}')


if __name__ == '__main__':
    main()
