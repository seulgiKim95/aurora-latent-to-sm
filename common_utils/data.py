import xarray as xr
import pandas as pd
import numpy as np
import bisect
import torch

from aurora import Batch
from aurora import Metadata
from .utils import parse_datetime


def load_nc_list(BASE_PATH, static_path=None, start_date='2025-11-01', end_date='2025-11-30', output_list=('surf', 'atmo', 'stat')):
    """get netCDF4 data list

    Args:
        BASE_PATH (`str`): base path string
        static_path (`str`, optional): static file path (BASE_PATH + filename)
        start_date (`str`, optional): '0000-00-00'
        end_date (`str`, optional): '0000-00-00'
        output_list (`tuple` of `str`, optional): output filelist set

    Returns:
        Tuple(
            `list` of `str`: surface variable nc file list
            `list` of `str`: atmospheric variable nc file list
            `list` of `str`: hydrological variable nc file list
            `list` of `str`: evapotranspiration variable nc file list
            `list` of `str`: precipitation variable nc file list
            :obj:`xarray.Dataset`
        )
    """
    allowed_keys = {'surf', 'atmo', 'stat', 'hydr', 'et', 'p'}
    input_keys = set(output_list)
    invalid_keys = input_keys - allowed_keys
    if invalid_keys:
        raise ValueError('load_nc_list: Invalid Argument')

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    output = ()
    for key in output_list:
        if key == 'stat':
            files = xr.open_dataset(static_path, engine='netcdf4')
        else:
            files = []
            for date in date_range:
                FILE_PATH = f'{BASE_PATH}/{date.year}/{date.strftime("%Y.%m.%d")}'
                file = FILE_PATH + f'/{date.strftime("%Y-%m-%d")}_{key}.nc'
                files.append(file)
        output = output + (files,)
    return output


class AuroraDataset(torch.utils.data.Dataset):
    def __init__(self, surf_paths, stat_ds, atmo_paths, seq_len: int = 2, max_open_files: int = 10):
        """Make batch to input Aurora model
        Args:
            surf_paths (`list` of `str`): surface variables
            atmo_paths (`list` of `str`): atmospheric variables
            stat_ds (:obj:`xarray.Dataset`): static variables
            seq_len (`int`, optional): number of time steps each training sample contains (defaults 2)
        """
        # key mapping
        self.surf_vars = self.surf_vars.copy() # shallow copy
        self.atmo_vars = self.atmo_vars.copy()
        self.stat_vars = self.stat_vars.copy()
        self.seq_len = seq_len

        self.surf_paths = surf_paths
        self.atmo_paths = atmo_paths
        self.stat_ds = stat_ds

        self.dataset_lengths = [] # length of each dataset
        self.dataset_start_idx = [0] # start index of each dataset
        self._get_dataset_info()

        self.max_open_files = max_open_files
        self.file_handles = {} # {'path': ds, ...}
        self.access_history = [] # ['path', ...]

        # compute the total number of valid start indices for building samples
        # e.g., with 10 timesteps, seq_len=2, and a single target, indices 0~8 are valid (8 samples)
        self.num_samples = self.dataset_start_idx[-1] - self.seq_len
        return

    surf_vars = {'2t':'t2m', '10u':'u10', '10v':'v10', 'msl':'msl'} # {aurora key : ERA5 key}
    atmo_vars = {'t':'t', 'u':'u', 'v':'v', 'q':'q', 'z':'z'}
    stat_vars = {'z':'z', 'slt':'slt', 'lsm':'lsm'}

    def __len__(self) -> int:
        return self.num_samples

    def _get_dataset_info(self):
        """define attributes :obj:`self.dataset_length`, :obj:`self.dataset_start_idx`"""
        for path in self.surf_paths:
            with xr.open_dataset(path, engine='netcdf4') as ds:
                # fall back to time if valid_time is absent
                length = ds.variables['valid_time'].shape[0] if 'valid_time' in ds.variables else ds.variables['time'].shape[0]
            self.dataset_lengths.append(length)
            self.dataset_start_idx.append(self.dataset_start_idx[-1] + length)
        return None

    def _get_cached_dataset(self, path):
        if path in self.file_handles: # already opened file
            if path in self.access_history: # move to the end of the list when the nc file is used
                self.access_history.remove(path)
            self.access_history.append(path)
            return self.file_handles[path]

        # if the file is not already open
        if len(self.file_handles) >= self.max_open_files: # when the max_open_files limit is reached
            oldest_path = self.access_history.pop(0) # close the oldest unused file
            if oldest_path in self.file_handles:
                self.file_handles[oldest_path].close()
                del self.file_handles[oldest_path]

        ds = xr.open_dataset(path, engine='netcdf4', cache=False)
        self.file_handles[path] = ds
        self.access_history.append(path)
        return ds

    def _get_continuous_values(self, file_paths, var_name, global_start, length):
        """deal with multi-dataset data"""
        parts = []
        rem_len = length
        curr_idx = global_start

        while rem_len > 0:
            file_idx = bisect.bisect_right(self.dataset_start_idx, curr_idx) - 1
            local_start = curr_idx - self.dataset_start_idx[file_idx]
            file_len = self.dataset_lengths[file_idx]
            can_read = min(rem_len, file_len - local_start)

            current_path = file_paths[file_idx]

            ds = self._get_cached_dataset(current_path)
            time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
            val = ds[var_name].isel({time_dim: slice(local_start, local_start + can_read)}).values
            val = self._adjust_longitude(val, ds['longitude'].values)
            val = self._adjust_latitude(val, ds['latitude'].values)

            parts.append(val)
            rem_len -= can_read
            curr_idx += can_read

        if len(parts) == 1:
            return parts[0]
        else:
            return np.concatenate(parts, axis=0)

    def _get_surf_data(self, idx, end_idx):
        """get surface variables"""
        surf_data = {}
        length = end_idx - idx
        for k, v in self.surf_vars.items():
            data = self._get_continuous_values(self.surf_paths, v, idx, length)
            surf_data[k] = data[None]
        return surf_data

    def _get_atmo_data(self, idx, end_idx):
        """get atmosphere variables"""
        atmo_data = {}
        length = end_idx - idx
        for k, v in self.atmo_vars.items():
            data = self._get_continuous_values(self.atmo_paths, v, idx, length)
            atmo_data[k] = data[None]
        return atmo_data

    def _get_stat_data(self):
        """get static variables"""
        static_data = {}
        for k, v in self.stat_vars.items():
            val = self.stat_ds[v].values[0]
            val = self._adjust_longitude(val, self.stat_ds['longitude'].values)
            val = self._adjust_latitude(val, self.stat_ds['latitude'].values)
            static_data[k] = val
        return static_data

    def _get_meta_data(self, idx):
        """get meta data"""
        file_idx = bisect.bisect_right(self.dataset_start_idx, idx) - 1
        local_idx = idx - self.dataset_start_idx[file_idx]

        # with xr.open_dataset(self.surf_paths[file_idx], engine='netcdf4') as current_ds:
        #     time_var = 'valid_time' if 'valid_time' in current_ds else 'time'
        #     cur_time = current_ds[time_var].values.astype("datetime64[s]").tolist()[local_idx]
        current_ds = self._get_cached_dataset(self.surf_paths[file_idx])
        time_var = 'valid_time' if 'valid_time' in current_ds else 'time'
        cur_time = current_ds[time_var].values.astype("datetime64[s]").tolist()[local_idx]

        if not hasattr(self, 'atmos_levels'):
            with xr.open_dataset(self.atmo_paths[0], engine='netcdf4') as ds:
                self.atmos_levels = tuple(int(level) for level in ds.pressure_level.values)

        lon = self.stat_ds.longitude.values
        lon = self._adjust_longitude(lon, lon)
        con = lon < 0
        lon[con] = lon[con] + 360.
        lat = self.stat_ds.latitude.values
        lat = self._adjust_latitude(lat, lat)
        metadata = Metadata(
            lat=torch.Tensor(lat),
            lon=torch.Tensor(lon),
            time=(cur_time,),
            atmos_levels=self.atmos_levels
        )
        return metadata

    def _adjust_latitude(self, value, lat=None):
        if lat is not None and lat[0] < lat[-1]:
            value = np.ascontiguousarray(value[..., ::-1, :]) if value.ndim >= 2 else value[::-1]
        if len(value.shape) >= 2:
            value = value[..., :720, :]
        else:
            value = value[:720]
        return value

    def _adjust_longitude(self, value, lon):
        adjust_lon = lon.flatten() < 0 # adjust longitude to 0~360
        if int(adjust_lon.sum()) > 0:
            adjust_idx = int(np.argwhere(adjust_lon)[-1].item() + 1)
            value = np.roll(value, shift=adjust_idx, axis=-1)
            value = np.ascontiguousarray(value)
        return value

    def get_datetime_batch(self, dt):
        if isinstance(dt, str):
            dt = parse_datetime(dt)
        for idx in range(0, self.num_samples):
            end_idx = idx + self.seq_len
            metadata = self._get_meta_data(end_idx -1)
            time = metadata.time[0]
            if time == dt:
                return self.__getitem__(idx)
        raise ValueError(f"Invalid time: '{dt}'")

    def __getitem__(self, idx: int) -> Batch:
        # idx: start index, end_idx: end index (exclusive)
        end_idx = idx + self.seq_len # used as [idx:end_idx], end_idx not included
        target_idx = end_idx # used as [target_idx]

        # 1. Input data
        surf_data = self._get_surf_data(idx, end_idx)
        atmo_data = self._get_atmo_data(idx, end_idx)
        static_data = self._get_stat_data() # use the static data of the first timestep, same as the inference batch (shared)
        metadata = self._get_meta_data(end_idx -1)

        # 2. Target Data
        target_surf = self._get_surf_data(target_idx, target_idx +1)
        target_atmo = self._get_atmo_data(target_idx, target_idx +1)
        target_meta = self._get_meta_data(target_idx)

        # 3. return
        input_batch = Batch(
            surf_vars={k: torch.from_numpy(v) for k, v in surf_data.items()},
            static_vars={k: torch.from_numpy(v) for k, v in static_data.items()},
            atmos_vars={k: torch.from_numpy(v) for k, v in atmo_data.items()},
            metadata=metadata
        )
        target_batch = Batch(
            surf_vars={k: torch.from_numpy(v) for k, v in target_surf.items()},
            static_vars={k: torch.from_numpy(v) for k, v in static_data.items()},
            atmos_vars={k: torch.from_numpy(v) for k, v in target_atmo.items()},
            metadata=target_meta
        )

        return input_batch, target_batch
