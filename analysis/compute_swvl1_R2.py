"""
Anomaly-field R2 of swvl1: mlp_linear predictions (NetCDF) vs ERA5 target (memmap).

- Anomaly: a = value - climatology (at that time's DOY).
- For each valid_time, compute spatial R2/ACC over land grid cells (time series).
- R2_anom = 1 - Σ w(a_p - a_o)^2 / Σ w(a_o - <a_o>_w)^2   (a_p-a_o = p-o)
- ACC = Σ w·a_p·a_o / √(Σ w·a_p^2 · Σ w·a_o^2)   (uncentered, same as get_metrics.py)

Conventions follow get_metrics.py:
  - latitude weighting : w = cos(lat) area weighting.
  - latitude crop      : LAT_CROP=600 (90° to -59.75°; excludes Antarctic below -60°).
  - land mask          : cells where p,o,c are all finite (exactly matches static lsm>=0.5).
"""
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

H, W = 720, 1440

CKPT = 25
PRED_DIRS = {
    '4ch': f'/home/data_2/Foundation_Model/Aurora/mlp_linear/4ch_epoch{CKPT}',
    '3ch': f'/home/data_2/Foundation_Model/Aurora/mlp_linear/3ch_epoch{CKPT}',
    '1ch': f'/home/data_2/Foundation_Model/Aurora/mlp_linear/1ch_epoch{CKPT}',
}
TARGET_BIN = '/data/personal_data/project_aurora/ERA5/ERA5_swvl1.bin'
TARGET_META = '/data/personal_data/project_aurora/ERA5/ERA5_swvl1_meta.csv'
CLIM_NC = '/data/personal_data/project_aurora/static/climatology_era5_swvl1.nc'  # daily (365,lat,lon)
SAVE_CSV = True

# latitude crop + cos(lat) area weighting (same as get_metrics.py)
LAT_CROP = 600                                              # 90° to -59.75° (excludes Antarctic)
LAT = 90.0 - 0.25 * np.arange(H)                            # 90 -> -89.75
WEIGHT = np.cos(np.deg2rad(LAT[:LAT_CROP]))[:, None]        # (600, 1)

# Longitude-alignment notes (from diagnostics):
#   - predictions (save_result): lon 0-360, already aligned with the memmap.
#   - target memmap       : saved as 0-360 by target_memmap.py -> no extra roll needed.
#   - climatology (.nc)   : stored as lon -180-180 -> only adjust_longitude to 0-360 using its own lons.
# (check: p-vs-o correlation is r=0.85 at roll 0 but 0.07 at roll 720, so rolling would break alignment)


def adjust_longitude(value, lon):
    """Same as common_utils._adjust_longitude. If any lon<0, roll to 0-360 (lon axis=-1)."""
    adjust_lon = lon.flatten() < 0
    if int(adjust_lon.sum()) > 0:
        adjust_idx = int(np.argwhere(adjust_lon)[-1].item() + 1)
        value = np.roll(value, shift=adjust_idx, axis=-1)
        value = np.ascontiguousarray(value)
    return value


def load_target():
    """Return memmap(T,720,1440) and {time: idx}."""
    T = os.path.getsize(TARGET_BIN) // (H * W * 4)
    mmap = np.memmap(TARGET_BIN, dtype=np.float32, mode='r', shape=(T, H, W))
    df = pd.read_csv(TARGET_META, parse_dates=['time'])
    idx_map = {pd.Timestamp(t).to_datetime64(): int(i)
               for t, i in zip(df['time'], df['mmap_idx'])}
    return mmap, idx_map


def load_climatology():
    """Align daily climatology (365,720,1440) to the prediction/target grid (0-360, lat 90 to -89.75).

    The climatology .nc is stored with lon -180-180 and lat 90 to -90 (721 rows), so:
      - adjust_longitude using its own lons -> 0-360 (same as predictions/memmap)
      - lat is north-first, so dropping the southernmost row via [:720] matches the target order.
    """
    with xr.open_dataset(CLIM_NC) as ds:
        clim = ds['swvl1'].values[:, :H, :].astype(np.float32)   # (365,720,1440), lat 90→-89.75
        clim = adjust_longitude(clim, ds['longitude'].values)     # align to 0-360 using its own lons
    return clim   # index: doy = min(int(%j),365)-1


def spatial_anomaly_metrics(p, o, c, w):
    """One time step of (LAT_CROP,W) prediction/observation/climatology -> cos(lat)-weighted anomaly R2/ACC etc.

    p,o,c,w are all cropped to [:LAT_CROP]. Only cells where p,o,c are all finite (=land) are used.
    a_p=p-c, a_o=o-c.  The numerator (sum w(p-o)^2) is shared by R2/R2_anom; only the denominators differ.
    R2      = 1 - Σ w(p-o)^2   / Σ w(o - <o>_w)^2      (raw values, includes the static climate pattern -> large positive)
    R2_anom = 1 - Σ w(a_p-a_o)^2 / Σ w(a_o-<a_o>_w)^2  (anomaly-based)
    ACC     = Σ w·a_p·a_o / √(Σ w·a_p^2 · Σ w·a_o^2)    (uncentered)
    """
    m = np.isfinite(p) & np.isfinite(o) & np.isfinite(c)
    if m.sum() < 2:
        return None
    W = np.broadcast_to(w, p.shape)[m]
    om = o[m]
    ap = p[m] - c[m]
    ao = om - c[m]
    e = ap - ao                       # = p - o
    sw = W.sum()

    cov = np.sum(W * ap * ao)
    vp = np.sum(W * ap * ap)          # uncentered (spatial mean not removed)
    vo = np.sum(W * ao * ao)
    acc = cov / np.sqrt(vp * vo) if vp > 0 and vo > 0 else np.nan

    ss_res = np.sum(W * e ** 2)       # sum w(p-o)^2, shared numerator of R2/R2_anom
    o_wmean = np.sum(W * om) / sw
    ss_tot_raw = np.sum(W * (om - o_wmean) ** 2)    # raw observed variance
    ao_wmean = np.sum(W * ao) / sw
    ss_tot_ano = np.sum(W * (ao - ao_wmean) ** 2)   # anomaly observed variance
    return {
        'N': int(m.sum()),
        'R2': 1 - ss_res / ss_tot_raw if ss_tot_raw > 0 else np.nan,
        'R2_anom': 1 - ss_res / ss_tot_ano if ss_tot_ano > 0 else np.nan,
        'ACC': acc,
        'RMSE': np.sqrt(ss_res / sw),
        'MAE': np.sum(W * np.abs(e)) / sw,
        'bias': np.sum(W * e) / sw,
    }


def per_time_metrics(pred_dir, mmap, idx_map, clim):
    """Return the spatial anomaly-R2 time series (DataFrame) over all prediction times in the folder."""
    rows = []
    n_miss = 0
    for f in sorted(glob.glob(os.path.join(pred_dir, '**', '*.nc'), recursive=True)):
        with xr.open_dataset(f) as ds:
            for ti, t in enumerate(ds['valid_time'].values):
                key = pd.Timestamp(t).to_datetime64()
                if key not in idx_map:
                    n_miss += 1
                    continue
                p = ds['swvl1'].isel(valid_time=ti).values.astype(np.float64)[:LAT_CROP]
                o = np.asarray(mmap[idx_map[key]], dtype=np.float64)[:LAT_CROP]  # memmap is 0-360, no roll needed

                doy = min(int(pd.Timestamp(t).strftime('%j')), 365) - 1
                c = clim[doy][:LAT_CROP]                          # already aligned during load

                met = spatial_anomaly_metrics(p, o, c, WEIGHT)
                if met is not None:
                    rows.append({'time': pd.Timestamp(t), **met})

    if not rows:
        return None, n_miss
    df = pd.DataFrame(rows).sort_values('time').reset_index(drop=True)
    return df, n_miss


def main():
    mmap, idx_map = load_target()
    clim = load_climatology()
    print(f"target T={mmap.shape[0]}, meta times={len(idx_map)}, clim days={clim.shape[0]}")

    summary = []
    for name, pred_dir in PRED_DIRS.items():
        print(f"\n=== {name} ===")
        df, n_miss = per_time_metrics(pred_dir, mmap, idx_map, clim)
        if df is None:
            print(f"  no valid comparison points (missing-in-meta={n_miss})")
            continue

        print(df[['time', 'N', 'R2', 'R2_anom', 'ACC', 'RMSE', 'MAE', 'bias']].to_string(index=False))
        if SAVE_CSV:
            out = f'./r2ano_{name}.csv'
            df.to_csv(out, index=False)
            print(f"  -> {out}")

        summary.append({'name': name, 'n_times': len(df),
                        'R2_mean': df['R2'].mean(),
                        'R2anom_mean': df['R2_anom'].mean(), 'R2anom_std': df['R2_anom'].std(),
                        'ACC_mean': df['ACC'].mean(), 'RMSE_mean': df['RMSE'].mean()})

    if summary:
        print("\n=== summary (time mean) ===")
        print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
