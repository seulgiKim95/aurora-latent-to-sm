"""SMAP / Aurora (FM) eSSMI generation + drought-mask IoU/Recall scoring pipeline.

Consolidates the eSSMI-generation code and the IoU-scoring code into one file
(plots excluded), self-contained in the same style as ESSMI_IoU_ECMWFS2S.py.

Pipeline (per region, in order):
    1. smap   : SMAP observations -> SMAP_eSSMI_{region}_... .nc
    2. aurora : Aurora predictions (leads 1-30) -> Aurora_eSSMI_{region}_... .nc
    3. iou    : IoU/Recall of the two drought masks (eSSMI <= -1) -> IoU_SMAP_FM_{region}_... .nc

Usage:
    python ESSMI_IoU_SMAP_FM.py                          # all regions, all steps
    python ESSMI_IoU_SMAP_FM.py Oklahoma --steps iou     # score only, from saved eSSMI files
    python ESSMI_IoU_SMAP_FM.py Argentina --steps smap,aurora
"""

import argparse
import os
import shutil
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.special import ndtr
from scipy.stats import norm

try:
    from tqdm.auto import tqdm
except ImportError:  # works without tqdm
    def tqdm(iterable, **kwargs):
        return iterable

# ---------------------------------------------------------------- configuration

SMAP_ROOT = Path("/home/jovyan/cpuserver/personal_data/project_aurora/SMAP/SPL3SMP_E.006_QC")
AURORA_ROOT = Path("/home/jovyan/data2/Foundation_Model/Aurora/unet/epoch10")
OUT_DIR = Path("./")

DATA_START = date(2015, 4, 1)
DATA_END = date(2025, 12, 31)
WINDOW_DAYS = 15
MIN_SAMPLES = 30
LOGIT_EPS = 1e-4

SMAP_VAR = "SMAP_SM_mean_utc"
AURORA_VAR = "swvl1"
ROLLOUT_DAYS_IN_FILENAME = 30
LEAD_DAYS_LIST = list(range(1, 31))
TARGET_HOURS = [0, 6, 12, 18]
ESSMI_TEMPORAL_MODE = "daily_mean"  # "daily_mean" or "utc_bin"
TARGET_RADIUS_DAYS = 0
SMAP_DISTRIBUTION_MODE = "target_window"  # "target_window" or "full_period"
SAVE_OUTPUTS = True
DROUGHT_THRESHOLD = -1.0
SMAP_DAY_CACHE_SIZE = 32

# Number of parallel workers for Aurora lead/time tasks. The job opens many netCDF/HDF5 files, so 2-4 is stable.
AURORA_MAX_WORKERS = min(4, os.cpu_count() or 1)
AURORA_PARALLEL_BACKEND = "loky"  # process-based; threading risks netCDF/HDF5 crashes.

REGIONS = {
    "Argentina": {
        "target_center": date(2022, 10, 15),
        "selection": "bbox",
        "label": "Argentina + Chile + Uruguay box",
        "lat_min": -56,
        "lat_max": -21,
        "lon_min": -76,
        "lon_max": -52,
    },
    "Oklahoma": {
        "target_center": date(2022, 3, 15),
        "selection": "bbox",
        "label": "Oklahoma surrounding box",
        "lat_min": 24,
        "lat_max": 38,
        "lon_min": -107,
        "lon_max": -92,
    },
    "Zambia": {
        "target_center": date(2024, 2, 15),
        "selection": "bbox",
        "label": "Zambia + Zimbabwe box",
        "lat_min": -23,
        "lat_max": -8,
        "lon_min": 13,
        "lon_max": 26,
    },
}

# ---------------------------------------------------------------- date/coordinate helpers


def date_range(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def target_date_range(center_date, radius_days):
    return [center_date + timedelta(days=i) for i in range(-radius_days, radius_days + 1)]


def make_target_times(center_date, radius_days=0, hours=None):
    hours = TARGET_HOURS if hours is None else hours
    return [
        datetime(d.year, d.month, d.day, h)
        for d in target_date_range(center_date, radius_days)
        for h in hours
    ]


def doy365(d):
    if d.month == 2 and d.day == 29:
        return None
    return date(2001, d.month, d.day).timetuple().tm_yday


def circular_doy_distance(a, b):
    diff = abs(a - b)
    return min(diff, 365 - diff)


def to_minus180_180(lon):
    return ((np.asarray(lon) + 180) % 360) - 180


def lon_bounds_for_data(da, lon_name, lon_min, lon_max):
    lon = da[lon_name].values
    data_min = float(np.nanmin(lon))
    data_max = float(np.nanmax(lon))

    if data_min >= 0 and lon_min < 0:
        lon_min = lon_min % 360
        lon_max = lon_max % 360
    elif data_max <= 180 and lon_min > 180:
        lon_min = float(to_minus180_180([lon_min])[0])
        lon_max = float(to_minus180_180([lon_max])[0])

    return lon_min, lon_max


def crop_region_bbox(da, region, lat_name, lon_name):
    lat = da[lat_name].values
    lon = da[lon_name].values
    lat_min = region["lat_min"]
    lat_max = region["lat_max"]
    lon_min, lon_max = lon_bounds_for_data(da, lon_name, region["lon_min"], region["lon_max"])

    lat_slice = slice(lat_max, lat_min) if lat[0] > lat[-1] else slice(lat_min, lat_max)
    lon_slice = slice(lon_min, lon_max) if lon[0] < lon[-1] else slice(lon_max, lon_min)
    return da.sel({lat_name: lat_slice, lon_name: lon_slice})


def regionmask_source(region):
    # regionmask downloads natural_earth data, so avoid loading it for bbox-only use.
    import regionmask

    mask_type = region.get("mask_type", "country")
    if mask_type == "country":
        return regionmask.defined_regions.natural_earth_v5_0_0.countries_110
    if mask_type == "us_state":
        return regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
    raise ValueError(f"unknown mask_type: {mask_type}")


def region_numbers(region):
    if region.get("selection", "bbox") == "bbox":
        return []
    source = regionmask_source(region)
    return [source.map_keys(name) for name in region["mask_names"]]


def regionmask_selected_mask(da, region, lat_name, lon_name):
    if region.get("selection", "bbox") == "bbox":
        return xr.DataArray(
            np.ones((da.sizes[lat_name], da.sizes[lon_name]), dtype=bool),
            coords={lat_name: da[lat_name], lon_name: da[lon_name]},
            dims=(lat_name, lon_name),
            name="bbox_selected_mask",
        )

    lat = da[lat_name].values
    lon = da[lon_name].values
    lon_for_mask = to_minus180_180(lon)
    source = regionmask_source(region)
    numbers = region_numbers(region)

    mask = source.mask(lon_for_mask, lat, wrap_lon=False)
    is_region = np.isin(mask.values, numbers)

    return xr.DataArray(
        is_region,
        coords={lat_name: da[lat_name], lon_name: da[lon_name]},
        dims=(lat_name, lon_name),
        name="regionmask_selected_mask",
    )


def sel_region(da, region, lat_name, lon_name):
    da = crop_region_bbox(da, region, lat_name, lon_name)
    if region.get("selection", "bbox") == "bbox":
        return da
    mask = regionmask_selected_mask(da, region, lat_name, lon_name)
    return da.where(mask)


def infer_lat_lon_names(da):
    lat_name = "latitude" if "latitude" in da.coords or "latitude" in da.dims else "lat"
    lon_name = "longitude" if "longitude" in da.coords or "longitude" in da.dims else "lon"
    return lat_name, lon_name


def normalize_plot_longitude(da, lon_name):
    lon = da[lon_name].values
    if np.nanmax(lon) > 180:
        da = da.assign_coords({lon_name: to_minus180_180(lon)}).sortby(lon_name)
    return da


def standardize_to_latlon(da):
    lat_name, lon_name = infer_lat_lon_names(da)
    da = normalize_plot_longitude(da, lon_name)
    rename = {}
    if lat_name != "latitude":
        rename[lat_name] = "latitude"
    if lon_name != "longitude":
        rename[lon_name] = "longitude"
    if rename:
        da = da.rename(rename)
    return da.sortby("latitude").sortby("longitude")


def daily_mean_for_target_date(da, target_date):
    day = da.sel(time=str(target_date))
    if day.sizes.get("time", 0) == 0:
        raise ValueError(f"no time samples for {target_date}")
    return day.mean("time", skipna=True)


# ---------------------------------------------------------------- eSSMI core (logit KDE CDF)


def logit_sm(x, eps=None):
    eps = LOGIT_EPS if eps is None else eps
    x = np.asarray(x, dtype="float32")
    x = np.where((x > 0) & (x < 1), x, np.nan)
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x)).astype("float32")


def essmi_from_samples(target_map, sample_stack, min_samples=None, logit_eps=None):
    min_samples = MIN_SAMPLES if min_samples is None else min_samples
    y = logit_sm(sample_stack, logit_eps)
    yt = logit_sm(target_map, logit_eps)

    valid = np.isfinite(y)
    n = valid.sum(axis=0).astype("float32")

    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nansum(y, axis=0) / np.where(n > 0, n, np.nan)
        var = np.nansum((y - mu[None, ...]) ** 2, axis=0) / np.where(n > 1, n - 1, np.nan)
        sigma = np.sqrt(var).astype("float32")
        bw = 1.06 * sigma * np.power(n, -1 / 5)
        bw = np.where((bw > 0) & np.isfinite(bw), bw, np.nan).astype("float32")
        z = (yt[None, ...] - y) / bw[None, ...]
        cdf = np.nanmean(ndtr(z), axis=0).astype("float32")

    cdf = np.where(n >= min_samples, cdf, np.nan)
    cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
    essmi = norm.ppf(cdf).astype("float32")

    return essmi, cdf.astype("float32"), n.astype("float32"), bw.astype("float32")


def save_dataset(ds, out_path):
    out_path = Path(out_path)
    tmp_path = out_path.parent / (out_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    ds_save = ds.load()
    ds_save.attrs = {k: str(v) for k, v in ds_save.attrs.items()}
    for v in ds_save.data_vars:
        ds_save[v].attrs = {k: str(val) for k, val in ds_save[v].attrs.items()}

    ds_save.to_netcdf(tmp_path, engine="netcdf4", format="NETCDF4")
    if out_path.exists():
        out_path.unlink()
    shutil.move(str(tmp_path), str(out_path))
    print("saved:", out_path)


# ---------------------------------------------------------------- 1) SMAP eSSMI generation


def smap_path_for_day(d):
    return Path(SMAP_ROOT) / f"{d.year}" / f"{d.year}.{d.month:02d}.{d.day:02d}" / f"{d:%Y-%m-%d}_smap_sm.nc"


def region_cache_key(region):
    return (
        region.get("selection", "bbox"),
        region.get("mask_type", "bbox"),
        tuple(region.get("mask_names", ())),
        float(region["lat_min"]),
        float(region["lat_max"]),
        float(region["lon_min"]),
        float(region["lon_max"]),
    )


def region_from_cache_key(key):
    selection, mask_type, mask_names, lat_min, lat_max, lon_min, lon_max = key
    return {
        "selection": selection,
        "mask_type": mask_type,
        "mask_names": list(mask_names),
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    }


@lru_cache(maxsize=SMAP_DAY_CACHE_SIZE)
def _open_smap_day_cached(day_ordinal, region_key):
    d = date.fromordinal(day_ordinal)
    fp = smap_path_for_day(d)
    if not fp.exists():
        return None

    region = region_from_cache_key(region_key)
    with xr.open_dataset(fp) as ds:
        da = ds[SMAP_VAR].astype("float32")
        da = sel_region(da, region, lat_name="latitude", lon_name="longitude")
        return da.load()


def open_smap_day(d, region):
    da = _open_smap_day_cached(d.toordinal(), region_cache_key(region))
    if da is None:
        return None
    return da.copy(deep=False)


def clear_smap_day_cache():
    _open_smap_day_cached.cache_clear()


def smap_temporal_map(da, temporal_mode=None, target_hour=None):
    temporal_mode = ESSMI_TEMPORAL_MODE if temporal_mode is None else temporal_mode
    if temporal_mode == "daily_mean":
        return da.sel(time=da.time.dt.hour.isin(TARGET_HOURS)).mean("time", skipna=True)
    if temporal_mode == "utc_bin":
        if target_hour is None:
            raise ValueError("target_hour is required when temporal_mode='utc_bin'")
        da_h = da.sel(time=da.time.dt.hour == target_hour)
        if da_h.sizes.get("time", 0) != 1:
            return None
        return da_h.isel(time=0)
    raise ValueError(f"unknown ESSMI_TEMPORAL_MODE: {temporal_mode}")


def target_output_time(td, temporal_mode=None, target_hour=None):
    temporal_mode = ESSMI_TEMPORAL_MODE if temporal_mode is None else temporal_mode
    if temporal_mode == "daily_mean":
        return np.datetime64(datetime(td.year, td.month, td.day, 12))
    if temporal_mode == "utc_bin":
        return np.datetime64(datetime(td.year, td.month, td.day, int(target_hour)))
    raise ValueError(f"unknown ESSMI_TEMPORAL_MODE: {temporal_mode}")


def collect_smap_climatology_samples(target_date, target_hour=None, region=None, temporal_mode=None):
    temporal_mode = ESSMI_TEMPORAL_MODE if temporal_mode is None else temporal_mode
    target_k = doy365(target_date)
    samples = []

    for d in date_range(DATA_START, DATA_END):
        k = doy365(d)
        if k is None or circular_doy_distance(k, target_k) > WINDOW_DAYS:
            continue

        da = open_smap_day(d, region)
        if da is None:
            continue

        sample_map = smap_temporal_map(da, temporal_mode=temporal_mode, target_hour=target_hour)
        if sample_map is not None:
            samples.append(sample_map.values.astype("float32"))

    if not samples:
        return None

    return np.stack(samples, axis=0)


def smap_essmi_output_path(region_name, target_center, radius_days=TARGET_RADIUS_DAYS, temporal_mode=None):
    temporal_mode = ESSMI_TEMPORAL_MODE if temporal_mode is None else temporal_mode
    return Path(OUT_DIR) / (
        f"SMAP_eSSMI_{region_name}_{target_center:%Y%m%d}_pm{radius_days}d_win{WINDOW_DAYS}_{temporal_mode}_{SMAP_VAR}.nc"
    )


def compute_smap_region_cdf(region_name, region=None, target_center=None, radius_days=TARGET_RADIUS_DAYS, temporal_mode=None):
    region = REGIONS[region_name] if region is None else region
    target_center = region["target_center"] if target_center is None else target_center
    temporal_mode = ESSMI_TEMPORAL_MODE if temporal_mode is None else temporal_mode
    target_dates = target_date_range(target_center, radius_days)

    template = None
    for d in target_dates:
        template = open_smap_day(d, region)
        if template is not None:
            break
    if template is None:
        raise FileNotFoundError("No readable SMAP UTC file in the target period.")

    lat = template["latitude"].values
    lon = template["longitude"].values

    essmi_list = []
    cdf_list = []
    n_list = []
    bw_list = []
    out_times = []

    for td in tqdm(target_dates, desc=f"SMAP {region_name} CDF ({temporal_mode})"):
        target_da = open_smap_day(td, region)
        if target_da is None:
            print("missing target:", td)
            continue

        if temporal_mode == "daily_mean":
            target_map = smap_temporal_map(target_da, temporal_mode=temporal_mode).values.astype("float32")
            sample_stack = collect_smap_climatology_samples(td, region=region, temporal_mode=temporal_mode)
            if sample_stack is None:
                print("no samples:", td, temporal_mode)
                continue

            essmi, cdf, n_valid, bw = essmi_from_samples(target_map, sample_stack)
            essmi_list.append(essmi)
            cdf_list.append(cdf)
            n_list.append(n_valid)
            bw_list.append(bw)
            out_times.append(target_output_time(td, temporal_mode=temporal_mode))

        elif temporal_mode == "utc_bin":
            for ti in range(target_da.sizes["time"]):
                target_time = target_da.time.values[ti]
                target_hour = int(target_da.time.dt.hour.values[ti])
                target_map = target_da.isel(time=ti).values.astype("float32")
                sample_stack = collect_smap_climatology_samples(
                    td,
                    target_hour=target_hour,
                    region=region,
                    temporal_mode=temporal_mode,
                )

                if sample_stack is None:
                    print("no samples:", td, target_hour)
                    continue

                essmi, cdf, n_valid, bw = essmi_from_samples(target_map, sample_stack)
                essmi_list.append(essmi)
                cdf_list.append(cdf)
                n_list.append(n_valid)
                bw_list.append(bw)
                out_times.append(target_time)
        else:
            raise ValueError(f"unknown ESSMI_TEMPORAL_MODE: {temporal_mode}")

    if not essmi_list:
        raise ValueError("SMAP CDF output is empty.")

    ds = xr.Dataset(
        data_vars={
            "SMAP_eSSMI": (
                ("time", "latitude", "longitude"),
                np.stack(essmi_list, axis=0),
                {
                    "long_name": "SMAP eSSMI from logit KDE CDF",
                    "description": f"eSSMI = norm.ppf(KDE percentile) using {temporal_mode} bbox-selected SMAP soil moisture",
                },
            ),
            "SMAP_KDE_percentile": (
                ("time", "latitude", "longitude"),
                np.stack(cdf_list, axis=0),
                {"long_name": "KDE CDF percentile"},
            ),
            "n_valid_samples": (
                ("time", "latitude", "longitude"),
                np.stack(n_list, axis=0),
                {"long_name": "number of valid climatology samples"},
            ),
            "kde_bandwidth_logit": (
                ("time", "latitude", "longitude"),
                np.stack(bw_list, axis=0),
                {"long_name": "Silverman bandwidth in logit space"},
            ),
        },
        coords={
            "time": np.array(out_times, dtype="datetime64[ns]"),
            "latitude": lat,
            "longitude": lon,
        },
        attrs={
            "source": str(SMAP_ROOT),
            "region": region_name,
            "selection": region.get("selection", "bbox"),
            "region_label": region.get("label", ""),
            "bbox_lat_min": str(region["lat_min"]),
            "bbox_lat_max": str(region["lat_max"]),
            "bbox_lon_min": str(region["lon_min"]),
            "bbox_lon_max": str(region["lon_max"]),
            "target_center": str(target_center),
            "target_range": f"{target_dates[0]} to {target_dates[-1]}",
            "temporal_window_days": str(WINDOW_DAYS),
            "temporal_mode": temporal_mode,
            "target_hours_for_daily_mean": str(TARGET_HOURS),
            "variable": SMAP_VAR,
            "method": f"bbox crop -> SMAP -> {temporal_mode} aggregation -> logit transform -> Gaussian KDE CDF -> norm.ppf",
        },
    )

    if SAVE_OUTPUTS:
        save_dataset(ds, smap_essmi_output_path(region_name, target_center, radius_days, temporal_mode))

    return ds


# ---------------------------------------------------------------- 2) Aurora eSSMI generation


def aurora_path_for_init(init_time, aurora_root=None, rollout_days=None, aurora_var=None):
    aurora_root = AURORA_ROOT if aurora_root is None else aurora_root
    rollout_days = ROLLOUT_DAYS_IN_FILENAME if rollout_days is None else rollout_days
    aurora_var = AURORA_VAR if aurora_var is None else aurora_var
    year_dir = Path(aurora_root) / f"{init_time.year}"
    return year_dir / f"{init_time:%Y%m%d}_{init_time.hour:02d}_R{rollout_days}_{aurora_var}.nc"


def read_aurora_lead(valid_time, lead_days, cfg, region, verbose=False, region_mask=None):
    init_time = valid_time - timedelta(days=lead_days)
    fp = aurora_path_for_init(
        init_time,
        cfg["aurora_root"],
        cfg["rollout_days_in_filename"],
        cfg["aurora_var"],
    )

    if not fp.exists():
        if verbose:
            print("missing file:", fp)
        return None

    with xr.open_dataset(fp) as ds:
        valid_time64 = np.datetime64(valid_time)
        if valid_time64 not in ds["valid_time"].values:
            if verbose:
                print("valid_time not in file:", valid_time, fp)
            return None

        da = ds[cfg["aurora_var"]].sel(valid_time=valid_time64).astype("float32")
        da = crop_region_bbox(da, region, lat_name="lat", lon_name="lon")
        if region.get("selection", "bbox") == "bbox":
            return da.load()
        mask = region_mask
        if mask is None:
            mask = regionmask_selected_mask(da, region, lat_name="lat", lon_name="lon")
        return da.where(mask).load()


def read_aurora_daily_mean(valid_date, lead_days, cfg, region, verbose=False, region_mask=None):
    maps = []
    for hour in cfg.get("target_hours", [0, 6, 12, 18]):
        valid_time = datetime(valid_date.year, valid_date.month, valid_date.day, int(hour))
        da = read_aurora_lead(valid_time, lead_days, cfg, region, verbose=verbose, region_mask=region_mask)
        if da is not None:
            maps.append(da)

    if not maps:
        return None

    return xr.concat(maps, dim="time").mean("time", skipna=True).astype("float32").load()


def collect_aurora_climatology_samples(target_item, lead_days, cfg, region, region_mask=None):
    temporal_mode = cfg.get("essmi_temporal_mode", "utc_bin")
    target_date = target_item if temporal_mode == "daily_mean" else target_item.date()
    target_k = doy365(target_date)
    samples = []

    for d in date_range(cfg["data_start"], cfg["data_end"]):
        k = doy365(d)
        if k is None or circular_doy_distance(k, target_k) > cfg["window_days"]:
            continue

        if temporal_mode == "daily_mean":
            da = read_aurora_daily_mean(d, lead_days, cfg, region, region_mask=region_mask)
        elif temporal_mode == "utc_bin":
            sample_valid_time = datetime(d.year, d.month, d.day, target_item.hour)
            da = read_aurora_lead(sample_valid_time, lead_days, cfg, region, region_mask=region_mask)
        else:
            raise ValueError(f"unknown essmi_temporal_mode: {temporal_mode}")

        if da is not None:
            samples.append(da.values.astype("float32"))

    if not samples:
        return None

    return np.stack(samples, axis=0)


def compute_aurora_lead_time_task(task):
    li, ti, lead_days, vt, region_name, region, cfg = task
    temporal_mode = cfg.get("essmi_temporal_mode", "utc_bin")

    if temporal_mode == "daily_mean":
        target_da = read_aurora_daily_mean(vt, lead_days, cfg, region)
        init_label = f"daily valid date {vt}, lead {lead_days}d"
    elif temporal_mode == "utc_bin":
        target_da = read_aurora_lead(vt, lead_days, cfg, region)
        init_label = f"valid {vt}, init {vt - timedelta(days=lead_days)}"
    else:
        raise ValueError(f"unknown essmi_temporal_mode: {temporal_mode}")

    if target_da is None:
        msg = f"missing target: {region_name} R {lead_days} {init_label}"
        return li, ti, None, msg

    region_mask = None
    if region.get("selection", "bbox") != "bbox":
        region_mask = regionmask_selected_mask(target_da, region, lat_name="lat", lon_name="lon")
    sample_stack = collect_aurora_climatology_samples(vt, lead_days, cfg, region, region_mask=region_mask)
    if sample_stack is None:
        msg = f"no climatology samples: {region_name} R {lead_days} {vt} ({temporal_mode})"
        return li, ti, None, msg

    essmi, cdf, n_valid, bw = essmi_from_samples(
        target_da.values.astype("float32"),
        sample_stack,
        min_samples=cfg["min_samples"],
        logit_eps=cfg["logit_eps"],
    )
    return li, ti, (essmi, cdf, n_valid, bw), None


def make_aurora_worker_config():
    return {
        "aurora_root": str(AURORA_ROOT),
        "aurora_var": AURORA_VAR,
        "rollout_days_in_filename": ROLLOUT_DAYS_IN_FILENAME,
        "data_start": DATA_START,
        "data_end": DATA_END,
        "window_days": WINDOW_DAYS,
        "min_samples": MIN_SAMPLES,
        "logit_eps": LOGIT_EPS,
        "target_hours": TARGET_HOURS,
        "essmi_temporal_mode": ESSMI_TEMPORAL_MODE,
    }


def find_aurora_template(target_dates, lead_days_list, region, cfg=None):
    cfg = make_aurora_worker_config() if cfg is None else cfg
    for lead_days in lead_days_list:
        for td in target_dates:
            if cfg.get("essmi_temporal_mode") == "daily_mean":
                da = read_aurora_daily_mean(td, lead_days, cfg, region)
                if da is not None:
                    return da
            else:
                for vt in make_target_times(td, 0):
                    da = read_aurora_lead(vt, lead_days, cfg, region)
                    if da is not None:
                        return da
    return None


def store_aurora_task_result(result, essmi_arr, cdf_arr, n_arr, bw_arr):
    li, ti, payload, message = result
    if message is not None:
        print(message)
        return

    essmi, cdf, n_valid, bw = payload
    essmi_arr[li, ti] = essmi
    cdf_arr[li, ti] = cdf
    n_arr[li, ti] = n_valid
    bw_arr[li, ti] = bw


def aurora_essmi_output_path(region_name, target_center, radius_days=TARGET_RADIUS_DAYS, lead_days_list=None):
    lead_days_list = LEAD_DAYS_LIST if lead_days_list is None else lead_days_list
    return Path(OUT_DIR) / (
        f"Aurora_eSSMI_{region_name}_{target_center:%Y%m%d}_pm{radius_days}d"
        f"_R{min(lead_days_list)}-{max(lead_days_list)}_win{WINDOW_DAYS}_{ESSMI_TEMPORAL_MODE}_{AURORA_VAR}.nc"
    )


def compute_aurora_region_leads(
    region_name,
    region=None,
    target_center=None,
    radius_days=TARGET_RADIUS_DAYS,
    lead_days_list=None,
    max_workers=None,
):
    region = REGIONS[region_name] if region is None else region
    target_center = region["target_center"] if target_center is None else target_center
    lead_days_list = LEAD_DAYS_LIST if lead_days_list is None else lead_days_list
    max_workers = AURORA_MAX_WORKERS if max_workers is None else max_workers

    cfg = make_aurora_worker_config()
    target_dates = target_date_range(target_center, radius_days)
    target_items = target_dates if ESSMI_TEMPORAL_MODE == "daily_mean" else make_target_times(target_center, radius_days)
    template = find_aurora_template(target_dates, lead_days_list, region, cfg=cfg)

    if template is None:
        raise FileNotFoundError("No readable Aurora R30 file in the target period.")

    lat = template["lat"].values
    lon = template["lon"].values
    shape = (len(lead_days_list), len(target_items), len(lat), len(lon))

    essmi_arr = np.full(shape, np.nan, dtype="float32")
    cdf_arr = np.full(shape, np.nan, dtype="float32")
    n_arr = np.full(shape, np.nan, dtype="float32")
    bw_arr = np.full(shape, np.nan, dtype="float32")

    tasks = [
        (li, ti, lead_days, vt, region_name, region, cfg)
        for li, lead_days in enumerate(lead_days_list)
        for ti, vt in enumerate(target_items)
    ]

    n_jobs = max(1, min(int(max_workers), len(tasks))) if max_workers is not None else 1
    desc = f"Aurora {region_name} lead-time tasks"
    used_backend = "serial" if n_jobs == 1 else AURORA_PARALLEL_BACKEND

    if n_jobs == 1:
        for task in tqdm(tasks, desc=desc):
            store_aurora_task_result(compute_aurora_lead_time_task(task), essmi_arr, cdf_arr, n_arr, bw_arr)
    else:
        from joblib import Parallel, delayed
        try:
            from joblib.externals.loky.process_executor import BrokenProcessPool
        except Exception:
            BrokenProcessPool = RuntimeError

        try:
            results = Parallel(
                n_jobs=n_jobs,
                backend=used_backend,
                return_as="generator_unordered",
                pre_dispatch=n_jobs,
            )(delayed(compute_aurora_lead_time_task)(task) for task in tasks)

            for result in tqdm(results, total=len(tasks), desc=f"{desc} ({n_jobs} workers, {used_backend})"):
                store_aurora_task_result(result, essmi_arr, cdf_arr, n_arr, bw_arr)
        except BrokenProcessPool as exc:
            print(f"{used_backend} backend failed ({exc}); retrying serially")
            used_backend = "serial_fallback"
            for task in tqdm(tasks, desc=f"{desc} (serial fallback)"):
                store_aurora_task_result(compute_aurora_lead_time_task(task), essmi_arr, cdf_arr, n_arr, bw_arr)

    ds = xr.Dataset(
        data_vars={
            "Aurora_eSSMI": (
                ("lead", "time", "lat", "lon"),
                essmi_arr,
                {
                    "long_name": "Aurora eSSMI from logit KDE CDF",
                    "description": "eSSMI = norm.ppf(KDE percentile) using bbox-selected Aurora swvl1 prediction",
                },
            ),
            "Aurora_KDE_percentile": (
                ("lead", "time", "lat", "lon"),
                cdf_arr,
                {"long_name": "KDE CDF percentile"},
            ),
            "n_valid_samples": (
                ("lead", "time", "lat", "lon"),
                n_arr,
                {"long_name": "number of valid climatology samples"},
            ),
            "kde_bandwidth_logit": (
                ("lead", "time", "lat", "lon"),
                bw_arr,
                {"long_name": "Silverman bandwidth in logit space"},
            ),
        },
        coords={
            "lead": np.array(lead_days_list, dtype="int16"),
            "time": (
                np.array([target_output_time(td) for td in target_items], dtype="datetime64[ns]")
                if ESSMI_TEMPORAL_MODE == "daily_mean"
                else np.array(target_items, dtype="datetime64[ns]")
            ),
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "source": str(AURORA_ROOT),
            "region": region_name,
            "selection": region.get("selection", "bbox"),
            "region_label": region.get("label", ""),
            "bbox_lat_min": str(region["lat_min"]),
            "bbox_lat_max": str(region["lat_max"]),
            "bbox_lon_min": str(region["lon_min"]),
            "bbox_lon_max": str(region["lon_max"]),
            "target_center": str(target_center),
            "target_radius_days": str(radius_days),
            "lead_days": f"{min(lead_days_list)}-{max(lead_days_list)}",
            "rollout_days_in_filename": str(ROLLOUT_DAYS_IN_FILENAME),
            "temporal_window_days": str(WINDOW_DAYS),
            "temporal_mode": ESSMI_TEMPORAL_MODE,
            "target_hours_for_daily_mean": str(TARGET_HOURS),
            "variable": AURORA_VAR,
            "method": f"bbox crop -> Aurora R30 forecast files -> selected lead -> {ESSMI_TEMPORAL_MODE} aggregation -> logit transform -> Gaussian KDE CDF -> norm.ppf",
            "parallel_workers": str(n_jobs),
            "parallel_backend": used_backend,
        },
    )

    if SAVE_OUTPUTS:
        save_dataset(ds, aurora_essmi_output_path(region_name, target_center, radius_days, lead_days_list))

    return ds


# ---------------------------------------------------------------- 3) IoU/Recall scoring


def iou_from_essmi(ref_da, pred_da, threshold=DROUGHT_THRESHOLD):
    ref = ref_da.values
    pred = pred_da.values
    valid = np.isfinite(ref) & np.isfinite(pred)
    ref_drought = (ref <= threshold) & valid
    pred_drought = (pred <= threshold) & valid

    intersection = int(np.logical_and(ref_drought, pred_drought).sum())
    union = int(np.logical_or(ref_drought, pred_drought).sum())
    ref_drought_count = int(ref_drought.sum())
    pred_drought_count = int(pred_drought.sum())
    iou = np.nan if union == 0 else intersection / union
    recall = np.nan if ref_drought_count == 0 else intersection / ref_drought_count

    return iou, recall, intersection, union, int(valid.sum()), ref_drought_count, pred_drought_count


def compute_lead_iou(smap_ds, aurora_ds, target_date, threshold=DROUGHT_THRESHOLD, region_name=None):
    smap_daily = standardize_to_latlon(daily_mean_for_target_date(smap_ds["SMAP_eSSMI"], target_date))

    lead_values = []
    iou_values = []
    recall_values = []
    intersection_values = []
    union_values = []
    valid_values = []
    smap_drought_values = []
    aurora_drought_values = []

    for lead in aurora_ds["lead"].values:
        aurora_daily = daily_mean_for_target_date(
            aurora_ds["Aurora_eSSMI"].sel(lead=lead),
            target_date,
        )
        aurora_daily = standardize_to_latlon(aurora_daily)
        aurora_on_smap = aurora_daily.interp(
            latitude=smap_daily["latitude"],
            longitude=smap_daily["longitude"],
            method="nearest",
        )

        iou, recall, intersection, union, valid, smap_drought, aurora_drought = iou_from_essmi(
            smap_daily,
            aurora_on_smap,
            threshold=threshold,
        )

        lead_values.append(int(lead))
        iou_values.append(iou)
        recall_values.append(recall)
        intersection_values.append(intersection)
        union_values.append(union)
        valid_values.append(valid)
        smap_drought_values.append(smap_drought)
        aurora_drought_values.append(aurora_drought)

    return xr.Dataset(
        data_vars={
            "IoU": ("lead", np.array(iou_values, dtype="float32")),
            "Recall": ("lead", np.array(recall_values, dtype="float32")),
            "intersection_pixels": ("lead", np.array(intersection_values, dtype="int32")),
            "union_pixels": ("lead", np.array(union_values, dtype="int32")),
            "valid_pixels": ("lead", np.array(valid_values, dtype="int32")),
            "SMAP_drought_pixels": ("lead", np.array(smap_drought_values, dtype="int32")),
            "Aurora_drought_pixels": ("lead", np.array(aurora_drought_values, dtype="int32")),
        },
        coords={"lead": np.array(lead_values, dtype="int16")},
        attrs={
            "region": region_name or "",
            "target_date": str(target_date),
            "drought_threshold": str(threshold),
            "reference": "SMAP_eSSMI drought mask, daily averaged before scoring",
            "prediction": "Aurora_eSSMI drought mask interpolated to SMAP grid, daily averaged before scoring",
            "recall_definition": "intersection_pixels / SMAP_drought_pixels",
        },
    )


def iou_output_path(region_name, target_center):
    return Path(OUT_DIR) / f"IoU_SMAP_FM_{region_name}_{target_center:%Y%m%d}.nc"


def score_iou(region_name, smap_ds=None, aurora_ds=None, threshold=DROUGHT_THRESHOLD, save=True):
    """Compute per-lead IoU/Recall from the eSSMI datasets (or the saved .nc files if not given)."""
    region = REGIONS[region_name]
    target_center = region["target_center"]

    if smap_ds is None:
        smap_ds = xr.open_dataset(smap_essmi_output_path(region_name, target_center))
    if aurora_ds is None:
        aurora_ds = xr.open_dataset(aurora_essmi_output_path(region_name, target_center))

    iou_ds = compute_lead_iou(smap_ds, aurora_ds, target_center, threshold=threshold, region_name=region_name)

    if save:
        save_dataset(iou_ds, iou_output_path(region_name, target_center))
    return iou_ds


# ---------------------------------------------------------------- pipeline / CLI


def run_pipeline(region_name, steps=("smap", "aurora", "iou"), threshold=DROUGHT_THRESHOLD, max_workers=None):
    """Run smap -> aurora -> iou in order for one region."""
    smap_ds = None
    aurora_ds = None

    if "smap" in steps:
        smap_ds = compute_smap_region_cdf(region_name)
    if "aurora" in steps:
        aurora_ds = compute_aurora_region_leads(region_name, max_workers=max_workers)
    if "iou" in steps:
        iou_ds = score_iou(region_name, smap_ds=smap_ds, aurora_ds=aurora_ds, threshold=threshold)
        best = int(np.nanargmax(iou_ds["IoU"].values))
        print(f"\n[{region_name}] target={REGIONS[region_name]['target_center']}  "
              f"best IoU=R{int(iou_ds['lead'][best])}: {float(iou_ds['IoU'][best]):.3f}")
        print(iou_ds[["IoU", "Recall"]].to_dataframe().to_string())
        return iou_ds
    return aurora_ds if aurora_ds is not None else smap_ds


def main():
    global SMAP_ROOT, AURORA_ROOT, OUT_DIR

    parser = argparse.ArgumentParser(description="SMAP/Aurora eSSMI generation + drought IoU/Recall scoring")
    parser.add_argument('regions', nargs='*', default=[],
                        help=f"target regions {list(REGIONS)} (default: all)")
    parser.add_argument('--steps', default='smap,aurora,iou',
                        help="steps to run, comma-separated (default: smap,aurora,iou)")
    parser.add_argument('--smap-root', type=Path, default=None, help="root of the daily SMAP files")
    parser.add_argument('--aurora-root', type=Path, default=None, help="root of the Aurora forecast files")
    parser.add_argument('--out', type=Path, default=None, help="output directory (default: current directory)")
    parser.add_argument('--workers', type=int, default=None, help="number of parallel Aurora workers (default: min(4, cores))")
    parser.add_argument('--threshold', type=float, default=DROUGHT_THRESHOLD, help="drought threshold (default -1.0)")
    args = parser.parse_args()

    if args.smap_root is not None:
        SMAP_ROOT = args.smap_root
    if args.aurora_root is not None:
        AURORA_ROOT = args.aurora_root
    if args.out is not None:
        OUT_DIR = args.out
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    steps = tuple(s.strip() for s in args.steps.split(',') if s.strip())
    unknown_steps = [s for s in steps if s not in ("smap", "aurora", "iou")]
    if unknown_steps:
        parser.error(f"unknown steps {unknown_steps}; choose from smap, aurora, iou")

    regions = args.regions or list(REGIONS)
    unknown = [r for r in regions if r not in REGIONS]
    if unknown:
        parser.error(f"unknown region {unknown}; choose from {list(REGIONS)}")

    for region_name in regions:
        run_pipeline(region_name, steps=steps, threshold=args.threshold, max_workers=args.workers)


if __name__ == '__main__':
    main()
