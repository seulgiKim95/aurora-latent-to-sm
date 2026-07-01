import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_workspace_dir = os.path.dirname(_current_dir)
if _workspace_dir not in sys.path:
    sys.path.append(_workspace_dir)

from train_hydr import dict_to_device
from train_hydr import _prepare_batch
from train_hydr import load_model
from train_hydr import load_nc_list
from train_hydr import load_data
from train_hydr import AuroraDatasetLiteDecoder
from train_hydr import locations, scales
import tqdm
import gc
import datetime
import numpy as np
import torch
from netCDF4 import Dataset
from aurora.batch import _np
from omegaconf import OmegaConf

exp_name = 'unet'
suffix = '0107_0803_10years'
model_key = 'load_aurora_lite_model'
checkpoint_num = 10
start_idx = 1 # 3

start_date = '2025-10-19'
end_date = '2025-12-31'

device = 'cuda:0'

# ------ setting -------- #
# Paths come from config.yaml (single source of truth).
_cfg = OmegaConf.load(os.path.join(_current_dir, "config.yaml"))
BASE_PATH_era5 = _cfg.paths.base_era5
BASE_PATH_mswep = _cfg.paths.base_mswep
static_file_path = _cfg.paths.static_file
SAVE_PATH = f'{_cfg.paths.rollout_root}{exp_name}/epoch{checkpoint_num}/'

rollout_step = 120
output_keys = ['swvl1']

file_suffix = f'R{rollout_step//4}_swvl1'
name = 'Seulgi Kim'
today = datetime.date.today().strftime('%Y-%m-%d')
history = f'{name}; ({today}) Create rollout result of Aurora unet head'

VAR_ATTR = {
    'swvl1': {'units': 'm3 m-3'},
    'precipitation': {'units': 'mm'},
}
EPOCH = datetime.datetime(1970, 1, 1)

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


def open_chain_file(start_time, lat, lon):
    """Open a per-chain NetCDF file with an unlimited time axis for incremental writes."""
    time_yr = start_time.year
    os.makedirs(SAVE_PATH + f'/{time_yr}', exist_ok=True)
    time_str = start_time.strftime('%Y%m%d_%H')
    path = SAVE_PATH + f'/{time_yr}/{time_str}_{file_suffix}.nc'

    ncf = Dataset(path, 'w', format='NETCDF4')
    ncf.history = history
    ncf.createDimension('valid_time', None)  # unlimited -> append per step
    ncf.createDimension('lat', len(lat))
    ncf.createDimension('lon', len(lon))

    v_lat = ncf.createVariable('lat', 'f4', ('lat',), zlib=True, complevel=4)
    v_lat[:] = lat
    v_lon = ncf.createVariable('lon', 'f4', ('lon',), zlib=True, complevel=4)
    v_lon[:] = lon
    v_time = ncf.createVariable('valid_time', 'f8', ('valid_time',))
    v_time.units = 'seconds since 1970-01-01'
    v_time.calendar = 'standard'
    for k in output_keys:
        var = ncf.createVariable(
            k, 'f4', ('valid_time', 'lat', 'lon'),
            zlib=True, complevel=4, fill_value=np.float32(np.nan),
        )
        var.units = VAR_ATTR[k]['units']
    return ncf, path


def write_chain_step(chain, time_t, pred_proc):
    """Append one roll-out step (one time slice) to the chain's NetCDF file."""
    ncf = chain['ncfile']
    wi = chain['write_idx']
    ncf.variables['valid_time'][wi] = (time_t - EPOCH).total_seconds()
    for k in output_keys:
        ncf.variables[k][wi, :, :] = pred_proc[k][0]  # (T=1, H, W) -> (H, W)
    chain['write_idx'] = wi + 1


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
    p = next(modelAurora.parameters())

    # ----- Roll-Out (streaming) ----- #
    # ERA5-forced atmosphere + autoregressive soil moisture.
    #
    # We sweep over time ONCE. At each timestep `t` the Aurora latent is computed a
    # single time and shared by every active roll-out chain (a chain = one 30-day
    # forecast started at a given day). Each chain at its own step consumes the same
    # latent[t] but keeps its own soil-moisture state, which is fed back
    # autoregressively. Compared with a naive per-roll-out loop this avoids reading the
    # same ERA5 timestep and recomputing the same Aurora forward ~rollout_step/4 times,
    # and its memory does not grow with the roll-out horizon (results are streamed to
    # NetCDF instead of accumulated in RAM).
    starts = set(range(start_idx, len(test_dataset) - rollout_step, 4))
    if not starts:
        print('no valid roll-out start indices for the given period / rollout_step')
        return
    sweep_end = max(starts) + rollout_step  # exclusive; last latent any chain needs

    decoder_lsm_mask = None
    active_chains = []

    print('rollout start (streaming)')
    with torch.inference_mode():
        for t in tqdm.tqdm(range(start_idx, sweep_end)):
            try:
                # ----- Aurora latent for this timestep (computed once) ----- #
                (batch, input_dict_t), _ = test_dataset[t]
                batch = _prepare_batch(modelAurora, batch)
                if decoder_lsm_mask is None:
                    decoder_lsm_mask = batch.static_vars['lsm'] >= 0.5

                pred_Aurora, lat_dec = modelAurora.forward(batch)
                latent = lat_dec.detach().clone()
                time_t = pred_Aurora.metadata.time[0]

                # ----- Start a new chain whose step 0 uses this latent ----- #
                if t in starts:
                    ncf, path = open_chain_file(batch.metadata.time[0], lat, lon)
                    active_chains.append({
                        'input_dict': dict_to_device(input_dict_t, p.device),
                        'ncfile': ncf,
                        'path': path,
                        'write_idx': 0,
                        'step': 0,
                    })

                # ----- Advance every active chain by one decoder step ----- #
                finished = []
                for chain in active_chains:
                    pred_Decoder = modelDecoder.forward(chain['input_dict'], latent, decoder_lsm_mask)
                    pred_proc = process(pred_Decoder, lsm_mask=lsm_mask)
                    write_chain_step(chain, time_t, pred_proc)
                    chain['input_dict'] = pred_Decoder  # autoregressive soil moisture
                    chain['step'] += 1
                    if chain['step'] >= rollout_step:
                        finished.append(chain)

                for chain in finished:
                    chain['ncfile'].close()
                    active_chains.remove(chain)

                del latent, pred_Aurora, batch
                if finished:
                    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f'({now}) {len(finished)} chain(s) done at t={t} (valid_time={time_t})')
                    gc.collect()
                    torch.cuda.empty_cache()

            except Exception as e:
                # Chains need consecutive latents, so a failed timestep invalidates every
                # currently-active chain. Drop and remove their (partial) files, then log.
                now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f'({now}) t={t} fail')
                print(e)
                with open('../errors.txt', 'a') as wf:
                    wf.write(f't={t}: {e}\n')
                for chain in active_chains:
                    try:
                        chain['ncfile'].close()
                        os.remove(chain['path'])
                    except Exception:
                        pass
                active_chains = []
                gc.collect()
                torch.cuda.empty_cache()

    # Close anything still open (defensive; should be empty if sweep_end is correct).
    for chain in active_chains:
        chain['ncfile'].close()


if __name__ == "__main__":
    main()
