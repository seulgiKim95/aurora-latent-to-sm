import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_workspace_dir = os.path.dirname(_current_dir)
if _workspace_dir not in sys.path:
    sys.path.append(_workspace_dir)

from train_hydr import rollout
from train_hydr import dict_to_device
from train_hydr import load_model
from train_hydr import load_nc_list
from train_hydr import load_data
from train_hydr import AuroraDatasetLiteDecoder
from train_hydr import locations, scales
import tqdm
import gc
import datetime
import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader
from netCDF4 import Dataset
from aurora.batch import _np
from omegaconf import OmegaConf

exp_name = 'mlp_enc'
suffix = '0227_1331_10years'
model_key = 'load_aurora_lite_model'
checkpoint_num = 25
start_idx = 1 # 3

start_date = '2015-03-31'
end_date = '2017-12-31'

device = 'cuda:0'

# ------ setting -------- #
# Paths come from config.yaml (single source of truth).
_cfg = OmegaConf.load(os.path.join(_current_dir, "config.yaml"))
BASE_PATH_era5 = _cfg.paths.base_era5
BASE_PATH_mswep = _cfg.paths.base_mswep
static_file_path = _cfg.paths.static_file
SAVE_PATH = f'{_cfg.paths.rollout_root}{exp_name}/rollout_t+1_06_epoch{checkpoint_num}/'

rollout_step = 120
output_keys = ['swvl1']

file_suffix = f'R{rollout_step//4}_swvl1'
name = 'Seulgi Kim'
today = datetime.date.today().strftime('%Y-%m-%d')
history = f'{name}; ({today}) Create rollout result of Aurora mlp_encoder head'

MODEL_PATH = f"{_cfg.paths.experiment_root}{exp_name}/{suffix}"
checkpoint = torch.load(MODEL_PATH + f"/checkpoint_epoch_{checkpoint_num:03d}.pth", map_location='cpu')
train_test_ratio = 0

def load_checkpoint(model_key, checkpoint):
    epoch = checkpoint['epoch']
    modelAurora, modelDecoder = load_model(model_key)
    modelDecoder.load_state_dict(checkpoint['model_state_dict'])
    modelAurora = modelAurora.to(device)
    modelDecoder = modelDecoder.to(device)
    modelAurora.eval()
    modelDecoder.eval()
    print('load checkpoint success')
    return modelAurora, modelDecoder

def set_normalize():
    new_vars = AuroraDatasetLiteDecoder.new_vars
    locations.update(dict.fromkeys(new_vars, 0.)) # mu
    scales.update(dict.fromkeys(new_vars, 1.)) # std
    return

def process(pred_dict, lsm_mask=None):
    result = {}
    for var in pred_dict.keys():
        data = _np(pred_dict[var][0]).copy() # torch.Tensor(B, T, H, W) -> np.ndarray(T, H, W)
        data = np.where(data >= 0, data, 0) # clamp negatives to 0
        if lsm_mask is not None: # masking sea
            data = np.where(lsm_mask, data, np.nan)
        if var in AuroraDatasetLiteDecoder.transformer: # unnormalize
            data = AuroraDatasetLiteDecoder.transformer[var].inverse_transform(data)
        result[var] = data
    return result

def create_ds(keys, data, time, lat, lon, history=None):
    coords = dict(valid_time=time, lon=lon, lat=lat)
    var_attr = {
        'swvl1': {'units': 'm3 m-3'},
        'precipitation': {'units': 'mm'}
    }
    attrs = {'history': history}

    data_vars = {}
    for k in keys:
        data_vars[k] = (['valid_time', 'lat', 'lon'], data[k], var_attr[k])

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs,
    )

    encoding = {
        "lat": {"zlib": True, "complevel": 4},
        "lon": {"zlib": True, "complevel": 4},
        "swvl1": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
        "valid_time": {"units": "seconds since 1970-01-01", "calendar": "standard"}
    }
    return ds_out, encoding



def main():
    print('device:', device)
    print('start_date:', start_date)
    print('end_date:', end_date)
    # ------ Load Data ------ #
    surf_path_list, atmo_path_list, hydr_path_list, et_path_list, stat_ds = load_nc_list(
        BASE_PATH_era5,
        static_path=static_file_path,
        start_date=start_date,
        end_date=end_date,
        output_list=('surf', 'atmo', 'hydr', 'et', 'stat'),
    )
    p_path_list,  = load_nc_list(
        BASE_PATH_mswep,
        start_date=start_date,
        end_date=end_date,
        output_list=('p',),
    )
    train_dataset, test_dataset = load_data(surf_path_list, atmo_path_list, hydr_path_list, et_path_list, p_path_list, stat_ds, train_test_ratio=train_test_ratio)

    # set_normalize()
    lsm_mask = stat_ds['lsm'].values.squeeze()[:720, :1440] >= 0.5
    lat = stat_ds['latitude'].values[:720]
    lon = stat_ds['longitude'].values[:1440]

    # ----- Load Model ----- #
    modelAurora, modelDecoder = load_checkpoint(model_key, checkpoint)

    # ----- Roll-Out ----- #
    print('rollout start')
    for idx in range(start_idx, len(test_dataset), 4):
        date = datetime.datetime.strptime(start_date, '%Y-%m-%d') + datetime.timedelta(days=(idx//4 + 1))
        date = date.strftime('%Y-%m-%d')
        try:
            # ----- Input Data ----- #
            (batch_obj, input_dict), target_obj = test_dataset.__getitem__(idx)
            # ----- Roll-Out ----- #
            with torch.inference_mode():
                preds = dict.fromkeys(output_keys, [])
                preds['time'] = []
                for pred in tqdm.tqdm(rollout(modelAurora, modelDecoder, batch_obj, input_dict, steps=rollout_step)):
                    pred_Aurora, pred_Decoder = pred
                    pred_Decoder = process(pred_Decoder, lsm_mask=lsm_mask)
                    time = pred_Aurora.metadata.time[0]

                    preds['time'].append(time)
                    for var in output_keys:
                        if var in preds.keys():
                            preds[var].append(pred_Decoder[var])
                        else:
                            pass ##### add pred_Aurora later if needed

            for var in output_keys:
                preds[var] = np.concatenate(preds[var], axis=0)
            # ----- Save ----- #
            time_r1 = batch_obj.metadata.time[0]
            time_yr = time_r1.year
            os.makedirs(SAVE_PATH + f'/{time_yr}', exist_ok=True)
            time_str = time_r1.strftime('%Y%m%d_%H')
            file = f'{time_str}_{file_suffix}.nc'
            ds_out, encoding = create_ds(output_keys, preds, preds['time'], lat, lon, history=history)
            ds_out.to_netcdf(SAVE_PATH + f'/{time_yr}/' + file, mode='w', encoding=encoding)
            ds_out.close()

            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'({now}) {date} done')
        except Exception as e:
            with open('../errors.txt', 'a') as wf:
                wf.write(f'{date}\n')
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'({now}) {date} fail')
            print(e)
        finally:
            if 'preds' in locals(): del preds
            if 'ds_out' in locals(): del ds_out
            if 'pred_Aurora' in locals(): del pred_Aurora
            if 'pred_Decoder' in locals(): del pred_Decoder

            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
