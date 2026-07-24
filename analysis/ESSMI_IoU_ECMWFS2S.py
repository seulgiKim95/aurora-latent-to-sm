from datetime import date, timedelta
from pathlib import Path
import shutil
import sys

import numpy as np
import pygrib
import xarray as xr
from scipy.special import ndtr
from scipy.stats import norm
from tqdm.auto import tqdm

HYDROAI_ROOT = Path('/home/jovyan/foundation_model')
if str(HYDROAI_ROOT) not in sys.path:
    sys.path.insert(0, str(HYDROAI_ROOT))
from HydroAI import Data as hData

ECMWF_ROOT = Path('/home/jovyan/cpuserver/personal_data/project_aurora/ECMWF/s2s_soil_moisture_top_20cm/target_day_15')
OUT_DIR = Path('/home/jovyan/foundation_model/data_analysis/kde_drought_outputs')
ECMWF_NC_DIR = OUT_DIR / 'ecmwf_s2s_native_grid_clean'
ECMWF_NC_DIR.mkdir(parents=True, exist_ok=True)

ECMWF_GRIB_VAR = 'sm20'
ECMWF_NC_VAR = 'ECMWF_sm20_volumetric'
ECMWF_ESSMI_VAR = 'ECMWF_eSSMI'
ECMWF_KDE_VAR = 'ECMWF_KDE_percentile'
ECMWF_LAYER_DEPTH_M = 0.20
ECMWF_WATER_DENSITY_KG_M3 = 1000.0
IOU_RESAMPLING_MAG_FACTOR = 10.0
DATA_START = date(2015, 4, 1)
DATA_END = date(2025, 12, 31)
WINDOW_DAYS = 15
MIN_SAMPLES = 30
LOGIT_EPS = 1e-4
DROUGHT_THRESHOLD = -1.0
LEAD_DAYS_LIST = list(range(1, 31))


def date_range(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def doy365(d):
    if d.month == 2 and d.day == 29:
        return None
    return date(2001, d.month, d.day).timetuple().tm_yday


def circular_doy_distance(a, b):
    diff = abs(a - b)
    return min(diff, 365 - diff)


def to_minus180_180(lon):
    return ((np.asarray(lon) + 180) % 360) - 180



def ecmwf_grib_path(init_date):
    return ECMWF_ROOT / f'{init_date.year}' / f'{init_date:%Y.%m.%d}' / f'{init_date:%Y-%m-%d}_s2s.grib'


def ecmwf_nc_path(init_date):
    return ECMWF_NC_DIR / f'{init_date:%Y-%m-%d}_s2s_native_{ECMWF_GRIB_VAR}.nc'


def save_dataset_atomic(ds, out_path):
    out_path = Path(out_path)
    tmp_path = Path('/tmp') / out_path.name
    if tmp_path.exists():
        tmp_path.unlink()
    ds_save = ds.load()
    ds_save.to_netcdf(tmp_path, engine='netcdf4', format='NETCDF4')
    if out_path.exists():
        out_path.unlink()
    shutil.move(str(tmp_path), str(out_path))
    return out_path


def ecmwf_sm20_to_m3m3(values, units):
    values = np.ma.filled(values, np.nan).astype('float32')
    values = np.where(np.isfinite(values) & (np.abs(values) >= 9000), np.nan, values)
    unit_text = str(units).replace(' ', '').lower()
    if 'kgm**-3' in unit_text or 'kgm-3' in unit_text:
        factor = 1.0 / ECMWF_WATER_DENSITY_KG_M3
        conversion = 'source kg m-3 divided by water density 1000 kg m-3'
    elif 'kgm**-2' in unit_text or 'kgm-2' in unit_text:
        factor = 1.0 / (ECMWF_WATER_DENSITY_KG_M3 * ECMWF_LAYER_DEPTH_M)
        conversion = 'source kg m-2 divided by water density 1000 kg m-3 and layer depth 0.20 m'
    else:
        factor = 1.0 / ECMWF_WATER_DENSITY_KG_M3
        conversion = f'assumed kg m-3; source units were {units!r}; divided by water density 1000 kg m-3'
    return (values * factor).astype('float32'), conversion


def _read_ecmwf_grib_messages(grib_path, max_lead=30):
    records = []
    grbs = pygrib.open(str(grib_path))
    try:
        for g in grbs:
            short_name = getattr(g, 'shortName', '')
            if short_name != ECMWF_GRIB_VAR:
                continue
            lead_days = int(getattr(g, 'forecastTime')) // 24
            if lead_days < 0 or lead_days > max_lead:
                continue
            values, lat2d, lon2d = g.data()
            source_units = getattr(g, 'units', '')
            values, conversion = ecmwf_sm20_to_m3m3(values, source_units)
            records.append({
                'lead': lead_days,
                'valid_time': np.datetime64(getattr(g, 'validDate')),
                'values': values,
                'lat2d': lat2d.astype('float32'),
                'lon2d': lon2d.astype('float32'),
                'units': 'm3 m-3',
                'source_units': source_units,
                'conversion': conversion,
            })
    finally:
        grbs.close()
    return records


def grib_to_native_nc(init_date, force=False, max_lead=30):
    out_path = ecmwf_nc_path(init_date)
    if out_path.exists() and not force:
        return out_path

    grib_path = ecmwf_grib_path(init_date)
    if not grib_path.exists():
        return None

    records = _read_ecmwf_grib_messages(grib_path, max_lead=max_lead)
    if not records:
        return None

    lat2d = records[0]['lat2d']
    lon2d = records[0]['lon2d']
    lat = lat2d[:, 0].astype('float32')
    lon = to_minus180_180(lon2d[0, :]).astype('float32')
    lon_order = np.argsort(lon)
    lon = lon[lon_order]

    leads = []
    valid_times = []
    frames = []

    for rec in tqdm(records, desc=f'cache ECMWF native {init_date:%Y-%m-%d}', leave=False):
        leads.append(rec['lead'])
        valid_times.append(rec['valid_time'])
        frames.append(rec['values'][:, lon_order].astype('float32'))

    order = np.argsort(leads)
    leads = np.asarray(leads, dtype='int16')[order]
    valid_times = np.asarray(valid_times, dtype='datetime64[ns]')[order]
    frames = np.stack(frames, axis=0)[order]

    ds = xr.Dataset(
        data_vars={
            ECMWF_NC_VAR: (
                ('lead', 'latitude', 'longitude'),
                frames,
                {
                    'long_name': 'ECMWF S2S top 20 cm soil moisture on native ECMWF grid',
                    'units': 'm3 m-3',
                    'source_units': records[0].get('source_units', ''),
                    'conversion': records[0].get('conversion', ''),
                    'missing_value_handling': 'masked/missing values and absolute values >= 9000 set to NaN before unit conversion',
                },
            ),
            'valid_time': ('lead', valid_times),
        },
        coords={
            'lead': leads,
            'latitude': lat,
            'longitude': lon,
        },
        attrs={
            'source_grib': str(grib_path),
            'init_date': str(init_date),
            'grid': 'ECMWF native grid; longitude converted to -180..180 and sorted',
            'note': 'ECMWF S2S is available only for initialized forecast dates present under ECMWF_ROOT, typically Monday/Thursday.',
        },
    )
    return save_dataset_atomic(ds, out_path)


def ensure_ecmwf_nc(init_date, force=False):
    return grib_to_native_nc(init_date, force=force, max_lead=max(LEAD_DAYS_LIST))


def lon_bounds_for_data(da, lon_name, lon_min, lon_max):
    lon = da[lon_name].values
    if float(np.nanmin(lon)) >= 0 and lon_min < 0:
        return lon_min % 360, lon_max % 360
    return lon_min, lon_max


def crop_region_bbox(da, region, lat_name='latitude', lon_name='longitude'):
    lat = da[lat_name].values
    lon = da[lon_name].values
    lat_min = region['lat_min']
    lat_max = region['lat_max']
    lon_min, lon_max = lon_bounds_for_data(da, lon_name, region['lon_min'], region['lon_max'])
    lat_slice = slice(lat_max, lat_min) if lat[0] > lat[-1] else slice(lat_min, lat_max)
    lon_slice = slice(lon_min, lon_max) if lon[0] < lon[-1] else slice(lon_max, lon_min)
    return da.sel({lat_name: lat_slice, lon_name: lon_slice})


def read_ecmwf_lead(valid_date, lead_days, region=None, create_nc=True):
    init_date = valid_date - timedelta(days=int(lead_days))
    nc_path = ensure_ecmwf_nc(init_date) if create_nc else ecmwf_nc_path(init_date)
    if nc_path is None or not Path(nc_path).exists():
        return None

    with xr.open_dataset(nc_path) as ds:
        if int(lead_days) not in ds['lead'].values:
            return None
        da = ds[ECMWF_NC_VAR].sel(lead=int(lead_days)).astype('float32')
        if region is not None:
            da = crop_region_bbox(da, region)
        da = da.load()
        da.attrs['init_date'] = str(init_date)
        da.attrs['valid_date'] = str(valid_date)
        da.attrs['lead_days'] = str(lead_days)
        return da


def logit_sm(x, eps=LOGIT_EPS):
    x = np.asarray(x, dtype='float32')
    x = np.where((x > 0) & (x < 1), x, np.nan)
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x)).astype('float32')


def essmi_from_samples(target_map, sample_stack, min_samples=MIN_SAMPLES, logit_eps=LOGIT_EPS):
    y = logit_sm(sample_stack, logit_eps)
    yt = logit_sm(target_map, logit_eps)
    valid = np.isfinite(y)
    n = valid.sum(axis=0).astype('float32')
    with np.errstate(invalid='ignore', divide='ignore'):
        mu = np.nansum(y, axis=0) / np.where(n > 0, n, np.nan)
        var = np.nansum((y - mu[None, ...]) ** 2, axis=0) / np.where(n > 1, n - 1, np.nan)
        sigma = np.sqrt(var).astype('float32')
        bw = 1.06 * sigma * np.power(n, -1 / 5)
        bw = np.where((bw > 0) & np.isfinite(bw), bw, np.nan).astype('float32')
        z = (yt[None, ...] - y) / bw[None, ...]
        cdf = np.nanmean(ndtr(z), axis=0).astype('float32')
    cdf = np.where(n >= min_samples, cdf, np.nan)
    cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
    essmi = norm.ppf(cdf).astype('float32')
    return essmi, cdf.astype('float32'), n.astype('float32'), bw.astype('float32')


def collect_ecmwf_climatology_samples(target_date, lead_days, region):
    target_k = doy365(target_date)
    samples = []
    sample_dates = []

    for d in date_range(DATA_START, DATA_END):
        k = doy365(d)
        if k is None or circular_doy_distance(k, target_k) > WINDOW_DAYS:
            continue
        da = read_ecmwf_lead(d, lead_days, region=region)
        if da is not None:
            samples.append(da.values.astype('float32'))
            sample_dates.append(d)

    if not samples:
        return None, sample_dates
    return np.stack(samples, axis=0), sample_dates


def find_ecmwf_template(target_date, lead_days_list, region):
    for lead_days in lead_days_list:
        da = read_ecmwf_lead(target_date, lead_days, region=region)
        if da is not None:
            return da
    return None


def compute_ecmwf_region_leads(region_name, region, target_date, lead_days_list=LEAD_DAYS_LIST, save=True):
    template = find_ecmwf_template(target_date, lead_days_list, region)
    if template is None:
        raise FileNotFoundError(f'No ECMWF forecast file found for {region_name} target {target_date} at requested leads.')

    lat = template['latitude'].values
    lon = template['longitude'].values
    shape = (len(lead_days_list), len(lat), len(lon))
    essmi_arr = np.full(shape, np.nan, dtype='float32')
    cdf_arr = np.full(shape, np.nan, dtype='float32')
    n_arr = np.full(shape, np.nan, dtype='float32')
    bw_arr = np.full(shape, np.nan, dtype='float32')
    available = np.zeros(len(lead_days_list), dtype=bool)
    n_dates = np.zeros(len(lead_days_list), dtype='int16')
    init_dates = []

    for li, lead_days in enumerate(tqdm(lead_days_list, desc=f'ECMWF {region_name} eSSMI')):
        target_da = read_ecmwf_lead(target_date, lead_days, region=region)
        init_date = target_date - timedelta(days=int(lead_days))
        init_dates.append(np.datetime64(init_date))
        if target_da is None:
            continue
        sample_stack, sample_dates = collect_ecmwf_climatology_samples(target_date, lead_days, region)
        if sample_stack is None:
            continue
        essmi, cdf, n_valid, bw = essmi_from_samples(target_da.values.astype('float32'), sample_stack)
        essmi_arr[li] = essmi
        cdf_arr[li] = cdf
        n_arr[li] = n_valid
        bw_arr[li] = bw
        available[li] = True
        n_dates[li] = len(sample_dates)

    ds = xr.Dataset(
        data_vars={
            ECMWF_ESSMI_VAR: (('lead', 'latitude', 'longitude'), essmi_arr),
            ECMWF_KDE_VAR: (('lead', 'latitude', 'longitude'), cdf_arr),
            'n_valid_samples': (('lead', 'latitude', 'longitude'), n_arr),
            'kde_bandwidth_logit': (('lead', 'latitude', 'longitude'), bw_arr),
            'target_available': ('lead', available),
            'n_climatology_dates': ('lead', n_dates),
            'init_date': ('lead', np.asarray(init_dates, dtype='datetime64[ns]')),
        },
        coords={
            'lead': np.asarray(lead_days_list, dtype='int16'),
            'latitude': lat,
            'longitude': lon,
        },
        attrs={
            'source': str(ECMWF_ROOT),
            'region': region_name,
            'target_date': str(target_date),
            'temporal_window_days': str(WINDOW_DAYS),
            'min_samples': str(MIN_SAMPLES),
            'method': 'ECMWF native grid -> logit KDE CDF -> norm.ppf; resampling to SMAP grid is applied only before IoU/Recall',
            'lead_note': 'ECMWF S2S init dates are sparse; unavailable target_date - lead combinations are retained as NaN.',
        },
    )

    if save:
        out_path = OUT_DIR / f'ECMWF_eSSMI_{region_name}_{target_date:%Y%m%d}_R{min(lead_days_list)}-{max(lead_days_list)}_win{WINDOW_DAYS}_{ECMWF_GRIB_VAR}.nc'
        save_dataset_atomic(ds, out_path)
    return ds


def daily_smap_essmi_path(region_name, target_date, radius_days=0, smap_var='SMAP_SM_mean_utc'):
    return OUT_DIR / f'SMAP_eSSMI_{region_name}_{target_date:%Y%m%d}_pm{radius_days}d_win{WINDOW_DAYS}_daily_mean_{smap_var}.nc'


def ecmwf_essmi_on_smap_path(region_name, target_date, lead_days_list=LEAD_DAYS_LIST):
    return OUT_DIR / f'ECMWF_eSSMI_on_SMAP_{region_name}_{target_date:%Y%m%d}_R{min(lead_days_list)}-{max(lead_days_list)}_win{WINDOW_DAYS}_{ECMWF_GRIB_VAR}.nc'


def standardize_to_latlon(da):
    rename = {}
    if 'lat' in da.dims:
        rename['lat'] = 'latitude'
    if 'lon' in da.dims:
        rename['lon'] = 'longitude'
    if rename:
        da = da.rename(rename)
    if 'longitude' in da.coords:
        lon = da['longitude'].values
        if np.nanmax(lon) > 180 and np.nanmin(lon) >= 0:
            da = da.assign_coords(longitude=to_minus180_180(lon).astype('float32'))
    return da.sortby('latitude').sortby('longitude')


def resample_ecmwf_essmi_to_smap(pred_da, smap_da, mag_factor=IOU_RESAMPLING_MAG_FACTOR):
    target_lon2d, target_lat2d = np.meshgrid(
        smap_da['longitude'].values.astype('float32'),
        smap_da['latitude'].values.astype('float32'),
    )
    input_lon2d, input_lat2d = np.meshgrid(
        pred_da['longitude'].values.astype('float32'),
        pred_da['latitude'].values.astype('float32'),
    )
    pred_on_smap = hData.Resampling(
        target_lon2d.astype('float32'),
        target_lat2d.astype('float32'),
        input_lon2d.astype('float32'),
        input_lat2d.astype('float32'),
        pred_da.values.astype('float32'),
        sampling_method='nearest',
        agg_method='mean',
        mag_factor=mag_factor,
    ).astype('float32')
    return xr.DataArray(
        pred_on_smap,
        coords={'latitude': smap_da['latitude'], 'longitude': smap_da['longitude']},
        dims=('latitude', 'longitude'),
        name=ECMWF_ESSMI_VAR,
        attrs={
            'resampling_function': 'HydroAI.Data.Resampling',
            'resampling_mag_factor': str(mag_factor),
            'sampling_method': 'nearest',
            'agg_method': 'mean',
            'target_grid': 'SMAP_eSSMI latitude/longitude',
        },
    )


def resample_ecmwf_ds_to_smap(smap_ds, ecmwf_ds, mag_factor=IOU_RESAMPLING_MAG_FACTOR):
    smap = standardize_to_latlon(smap_ds['SMAP_eSSMI'].mean('time', skipna=True))
    leads = ecmwf_ds['lead'].values
    out = np.full((len(leads), smap.sizes['latitude'], smap.sizes['longitude']), np.nan, dtype='float32')

    for li, lead in enumerate(tqdm(leads, desc='resample ECMWF eSSMI to SMAP')):
        pred = standardize_to_latlon(ecmwf_ds[ECMWF_ESSMI_VAR].sel(lead=lead))
        out[li] = resample_ecmwf_essmi_to_smap(pred, smap, mag_factor=mag_factor).values.astype('float32')

    data_vars = {
        ECMWF_ESSMI_VAR: (('lead', 'latitude', 'longitude'), out),
    }
    for name in ['target_available', 'n_climatology_dates', 'init_date']:
        if name in ecmwf_ds:
            data_vars[name] = ('lead', ecmwf_ds[name].values)

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            'lead': leads.astype('int16'),
            'latitude': smap['latitude'].values.astype('float32'),
            'longitude': smap['longitude'].values.astype('float32'),
        },
        attrs={
            'source': ecmwf_ds.attrs.get('source', str(ECMWF_ROOT)),
            'region': ecmwf_ds.attrs.get('region', ''),
            'target_date': ecmwf_ds.attrs.get('target_date', ''),
            'source_grid': 'ECMWF native eSSMI grid',
            'target_grid': 'SMAP daily_mean eSSMI grid',
            'resampling_function': 'HydroAI.Data.Resampling',
            'resampling_mag_factor': str(mag_factor),
            'sampling_method': 'nearest',
            'agg_method': 'mean',
            'note': 'ECMWF eSSMI is computed on the native ECMWF grid, then resampled to the SMAP grid for IoU/Recall.',
        },
    )


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


def compute_ecmwf_iou_against_smap(smap_ds, ecmwf_ds, threshold=DROUGHT_THRESHOLD, ecmwf_on_smap_ds=None):
    smap = standardize_to_latlon(smap_ds['SMAP_eSSMI'].mean('time', skipna=True))
    if ecmwf_on_smap_ds is None:
        ecmwf_on_smap_ds = resample_ecmwf_ds_to_smap(smap_ds, ecmwf_ds)

    leads = []
    rows = []
    for lead in ecmwf_on_smap_ds['lead'].values:
        pred_on_smap = standardize_to_latlon(ecmwf_on_smap_ds[ECMWF_ESSMI_VAR].sel(lead=lead))
        rows.append(iou_from_essmi(smap, pred_on_smap, threshold=threshold))
        leads.append(int(lead))
    arr = np.asarray(rows, dtype='float64')
    return xr.Dataset(
        data_vars={
            'IoU': ('lead', arr[:, 0].astype('float32')),
            'Recall': ('lead', arr[:, 1].astype('float32')),
            'intersection_pixels': ('lead', arr[:, 2].astype('int32')),
            'union_pixels': ('lead', arr[:, 3].astype('int32')),
            'valid_pixels': ('lead', arr[:, 4].astype('int32')),
            'SMAP_drought_pixels': ('lead', arr[:, 5].astype('int32')),
            'ECMWF_drought_pixels': ('lead', arr[:, 6].astype('int32')),
            'target_available': ('lead', ecmwf_ds['target_available'].values.astype(bool)),
            'n_climatology_dates': ('lead', ecmwf_ds['n_climatology_dates'].values.astype('int16')),
        },
        coords={'lead': np.asarray(leads, dtype='int16')},
        attrs={
            'drought_threshold': str(threshold),
            'reference': 'SMAP daily_mean eSSMI drought mask',
            'prediction': 'ECMWF S2S eSSMI drought mask resampled to SMAP grid with HydroAI.Data.Resampling',
            'resampling_mag_factor': str(IOU_RESAMPLING_MAG_FACTOR),
            'lead_note': 'Only Monday/Thursday ECMWF initialization dates exist; unavailable leads are NaN.',
        },
    )
