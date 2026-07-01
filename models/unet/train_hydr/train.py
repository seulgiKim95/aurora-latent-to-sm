import os
import numpy as np
import torch
from torch.amp import autocast
from torch.amp import GradScaler
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    ConstantLR,
    SequentialLR,
)

import torch.nn.functional as F

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.distributed.checkpoint.state_dict import StateDictOptions
from torch.nn.attention import sdpa_kernel, SDPBackend

from aurora import Batch
from aurora.normalisation import normalise_surf_var
import train_hydr
from train_hydr.model import get_new_variable_parameters
from train_hydr import AuroraDataset
from train_hydr.data import AuroraDatasetLiteDecoder

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # for debugging
# torch.autograd.set_detect_anomaly(True)

torch.backends.cuda.matmul.allow_tf32 = True # lower compute precision to increase speed
torch.backends.cudnn.allow_tf32 = True
backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH] # do not use flash-attn; use only math and efficient_attention


class LossCriterion():
    def __init__(self, land_sea_mask):
        self.land_sea_mask = land_sea_mask
        self.epsilon = 1.e-6
        self.land_criterion = 0.5 # land_sea_mask value must be >= 0.5 to be considered land

        # variable weights
        self.w = self.set_weights(default=1.)
        for k in self.w.keys():
            if hasattr(train_hydr.config.loss.var_weights, k):
                self.w[k] = getattr(train_hydr.config.loss.var_weights, k)

        # loss weights (alpha)
        self.alpha = self.set_weights(default={'L1':1.})
        for k1 in self.alpha.keys():
            if hasattr(train_hydr.config.loss.alpha, k1):
                self.alpha[k1]["L1"] = getattr(train_hydr.config.loss.alpha, k1).L1

        # land sea weights (alpha_land_sea)
        self.alpha_land_sea = self.set_weights(default=0.5) # no weights for land & sea
        for k in self.alpha_land_sea.keys():
            if hasattr(train_hydr.config.loss.alpha_land_sea, k):
                self.alpha_land_sea[k] = getattr(train_hydr.config.loss.alpha_land_sea, k)
        self.land_weights = {}

        # latitude weights
        self.latitude_weights = None
        pass

    @staticmethod
    def set_weights(default=0., value=None):
        if value is None:
            value = default
        new_vars = AuroraDatasetLiteDecoder.new_vars

        w = {}
        w.update(dict.fromkeys(new_vars, value))
        return w

    def set_land_weights(self, device):
        for k in self.alpha_land_sea.keys():
            self.land_weights[k] = torch.Tensor(
                (self.land_sea_mask >= self.land_criterion) * self.alpha_land_sea[k]
                + (self.land_sea_mask < self.land_criterion) * (1. - self.alpha_land_sea[k])
            ).to(device)
        return

    def set_latitude_weights(self, metadata, device):
        lat_size = len(metadata.lat)
        lon_size = len(metadata.lon)
        level_size = len(metadata.atmos_levels)
        lat = metadata.lat.numpy()
        w_lat = torch.from_numpy(np.cos(np.deg2rad(lat)))
        self.latitude_weights = w_lat.view((1, 1, lat_size, 1)).expand(1, 1, -1, lon_size).to(device)
        return

    def loss_l1(self, pred, target, key):
        loss_sum = F.l1_loss(
            self.land_weights[key] * pred,
            self.land_weights[key] * target,
            reduction='sum',
            weight=self.latitude_weights,
        )
        loss = loss_sum / (self.land_weights[key].sum() + self.epsilon)
        return loss

    def calc_var_loss(self, pred: dict, target: dict, key) -> torch.Tensor:
        p = normalise_surf_var(pred[key], key)
        t = normalise_surf_var(target[key], key)
        l1_loss = self.loss_l1(p, t, key)
        var_loss = l1_loss
        return var_loss

    def loss(self, pred: dict, target: dict) -> torch.Tensor:
        loss = 0.
        for k in pred.keys():
            var_loss = self.calc_var_loss(pred, target, k)
            loss = loss + self.w[k] * var_loss
        return loss


def train_model(modelAurora, modelDecoder, dataloader, optimizer, criterion, device, epoch, scaler=None, scheduler=None, verbose=True):
    modelAurora.eval() # evaluation mode
    modelDecoder.train() # set the model to training mode
    total_loss = 0

    if verbose:
        train_hydr.logger.debug(f"--- Epoch {epoch} Start ---")

    lsm_mask = torch.Tensor(criterion.land_sea_mask) >= criterion.land_criterion

    for i, ((batch_obj, input_dict), target) in enumerate(dataloader):
        optimizer.zero_grad() # reset previous gradients

        batch_obj = batch_obj.to(device)
        input_dict = {k: v.to(device) for k, v in input_dict.items()}
        target = {k: v.to(device) for k, v in target.items()}

        with (
            sdpa_kernel(backends),
#           torch.autograd.detect_anomaly() # for debugging
        ):
            if scaler is not None: # float16, scaler
                with autocast(device_type='cuda', dtype=torch.float16):
                    with torch.no_grad():
                        pred_Aurora, lat_dec = modelAurora(batch_obj)
                        latent_decoder = lat_dec.detach().clone()
                    pred_Decoder = modelDecoder(input_dict, latent_decoder, lsm_mask)
                    loss_value = criterion.loss(pred_Decoder, target) # compute loss
                scaler.scale(loss_value).backward()
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                if scale_after >= scale_before:
                    scheduler_update = True
                else:
                    scheduler_update = False
            else: # bfloat16: same exponent as float32
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    with torch.no_grad():
                        pred_Aurora, lat_dec = modelAurora(batch_obj)
                        latent_decoder = lat_dec.detach().clone()
                    input_dict_bf16 = {
                        k: v.to(torch.bfloat16) if isinstance(v, torch.Tensor) else v
                        for k, v in input_dict.items()
                    }
                    pred_Decoder = modelDecoder(input_dict_bf16, latent_decoder, lsm_mask)
                    loss_value = criterion.loss(pred_Decoder, target) # compute loss
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(modelDecoder.parameters(), max_norm=1.0) # gradient clipping
                optimizer.step()
                scheduler_update = True

        if scheduler is not None and scheduler_update:
            scheduler.step()
        # print_gpu_usage()

        current_loss = loss_value.detach().cpu().item()
        total_loss = total_loss + current_loss

        if ((i + 1) % 5 == 0) and verbose:
            train_hydr.logger.debug(f'[Epoch {epoch}] Batch [{i+1}/{len(dataloader)}] - Loss: {total_loss / (i + 1):.8f}')

#       del loss_value, pred_Aurora, pred_Decoder, lat_dec, latent_decoder, current_loss
#       torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# Takes a list of Batch objects and merges them into a single Batch object (with batched tensors inside)
def aurora_collate_fn(batch_list):
    # batch_list = [(input_batch1, target_batch1), (input_batch2, target_batch2), ...]

    # 1. Separate inputs and targets
    inputs = [item[0] for item in batch_list]
    targets = [item[1] for item in batch_list]

    # 2. Helper function to merge the inner tensors
    def stack_batch_objs(obj_list):
        # Use the first object as the reference for the structure
        first_obj = obj_list[0]

        # Merge the tensors in the dict with torch.cat (they already have a [None] dimension, so use cat)
        # If there were no [None], use torch.stack
        surf = {k: torch.cat([b.surf_vars[k] for b in obj_list], dim=0) for k in first_obj.surf_vars}
        atmo = {k: torch.cat([b.atmos_vars[k] for b in obj_list], dim=0) for k in first_obj.atmos_vars}
        static = {k: torch.cat([b.static_vars[k] for b in obj_list], dim=0) for k in first_obj.static_vars}

        # Metadata cannot be merged, so keep it as a list or use only the first one
        # Here we use the first batch's metadata as the representative (modify if needed)
        metadata = first_obj.metadata

        return Batch(surf_vars=surf, atmos_vars=atmo, static_vars=static, metadata=metadata)

    def stack_dict_objs(obj_list):
        first_obj = obj_list[0]
        return {k: torch.cat([d[k] for d in obj_list], dim=0) for k in first_obj.keys()}

    # 3. Merge inputs and targets separately
    batched_input = stack_batch_objs(inputs)
    dict_target = stack_dict_objs(targets)

    return batched_input, dict_target


def initialize_weights(model, new_variables=[], automatic=False):
    if automatic:
        surf_vars = tuple(AuroraDataset.surf_vars.keys())
        atmo_vars = tuple(AuroraDataset.atmo_vars.keys())
        new_vars = model.surf_vars + model.atmos_vars
        new_vars = list(set(new_vars) - set(surf_vars) - set(atmo_vars))
        new_variables.extend([f'.{var}' for var in new_vars])
    with torch.no_grad():
        for name, param in model.named_parameters():
            if any([var in name for var in new_vars]) and param.requires_grad:
                if 'bias' in name:
                    param.zero_()
                else:
                    param.normal_(mean=0, std=0.001)
    return


def load_lr_scheduler(optimizer, warmup_steps, total_steps, lr_decay_value=5e-6):
    """load learning rate scheduler for warm-up and decay

    Args:
        optimizer: torch optimizer
        warmup_steps (`int`): steps for warm-up
        total_steps (`int`): total steps

    Returns:
        :obj:`torch.optim.lr_scheduler.SequentialLR`
    """
    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)

    main_steps = total_steps - warmup_steps
    scheduler_main = CosineAnnealingLR(optimizer, T_max=main_steps, eta_min=lr_decay_value) # cosine decay
    # scheduler_main = ConstantLR(optimizer) # constant

    scheduler = SequentialLR(
        optimizer, 
        schedulers=[scheduler_warmup, scheduler_main], 
        milestones=[warmup_steps]
    )
    return scheduler


def load_optimizer(model, new_variables=None, automatic=False):
    """load optimizer

    Args:
        model: Aurora model
        new_variables (`list` of `str`, optional): list of new variables
            (e.g., ['.lora'])
        automatic (`str`, optional): automatically finding new variables

    Returns:
        :obj:`torch.optim.adamw.AdamW`
    """
    if new_variables is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_hydr.config.train.lr_new_vars)
    else:
        if automatic:
            surf_vars = tuple(AuroraDataset.surf_vars.keys())
            atmo_vars = tuple(AuroraDataset.atmo_vars.keys())
            new_vars = model.surf_vars + model.atmos_vars
            new_vars = list(set(new_vars) - set(surf_vars) - set(atmo_vars))
            new_variables.extend([f'.{var}' for var in new_vars])
        new_params, exist_params = get_new_variable_parameters(model, new_variables)
        optimizer = torch.optim.AdamW([ # decoupled weight decay is standard for transformers
            {'params': exist_params, 'lr': train_hydr.config.train.lr_backbone},
            {'params': new_params, 'lr': train_hydr.config.train.lr_new_vars}, # differential learning rate
        ])
    return optimizer


def load_scaler(id_scaler):
    """load scaler

    Args:
        id_scaler (`str`): id of scaler
            'none': None
            'grad': GradScaler

    Returns:
        :obj:`torch.amp.GradScaler`
    """
    if id_scaler == 'none':
        scaler = None
    elif id_scaler == 'grad':
        scaler = GradScaler()
    else:
        scaler = None
    return scaler


def load_checkpoint(device, pth, model, optimizer=None, scheduler=None, scaler=None):
    checkpoint = torch.load(pth, map_location='cpu')
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    if optimizer is not None:
        if train_hydr.config.train.use_fsdp:
            optim_state_dict = FSDP.optim_state_dict_to_load(model, optimizer, checkpoint['optimizer_state_dict'])
        else:
            optim_state_dict = checkpoint['optimizer_state_dict']
        optimizer.load_state_dict(optim_state_dict)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return epoch, model, optimizer, scheduler, scaler


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, rank=0, folder_path="./checkpoints/", filename='.last_checkpoint.pth'):
    if rank == 0:
        os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, filename)

    # For FSDP, the model and optimizer sharded across GPUs must be gathered into one before saving.
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    # full_state_dict=True: full model mode
    # offload_to_cpu=True: gather parameters into CPU RAM
    model_state, optimizer_state = get_state_dict(model, optimizer, options=options)

    if rank == 0: # save on rank 0
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        }
        torch.save(checkpoint, filepath)
        if filename == '.last_checkpoint.pth':
            filepath_cp = os.path.join(folder_path, filename.removeprefix('.'))
            os.replace(filepath, filepath_cp)
        train_hydr.logger.debug(f"[Checkpoints] Epoch {epoch} | Checkpoint saved: {filepath}")

    if train_hydr.config.train.use_dist:
        dist.barrier(device_ids=[torch.cuda.current_device()])
    return


class EarlyStopping:
    def __init__(self, patience=3, delta=0.0, mode='min', rank=0, save_path='./checkpoints/', verbose=True):
        """
        Args:
            patience (`int`, optional): the epoch waiting for loss to improve
            delta (`float`, optional): loss fluctuation range. The loss change greater than this value is considered an improvement.
            mode (`str`, optional): minimize loss or maximize score ('min' or 'max')
            save_path (`str`, optional): save directory
            verbose (`bool`, optional): print message
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.save_path = save_path
        self.local_rank = rank
        self.verbose = verbose

        self.early_stop = False
        self.counter = 0
        self.best_score = -np.inf
        pass

    def __call__(self, model, optimizer, epoch, loss):
        if self.mode == 'min':
            score = -1 * loss # transfer loss to score (minimize -> maximize)
        else:
            score = loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, epoch, loss)
            self.counter = 0
            return

        if score > (self.best_score + self.delta): # improvement
            self.best_score = score
            self.save_checkpoint(model, optimizer, epoch, loss)
            self.counter = 0
        else: # patience
            self.counter = self.counter + 1
            if self.verbose:
                train_hydr.logger.debug(f'[EarlyStopping] Epoch {epoch} | Early Stopping Counter: {self.counter} out of {self.patience}, Best: {self.best_score:.8f}')

            if self.counter >= self.patience: # early stopping
                self.early_stop = True

    def save_checkpoint(self, model, optimizer, epoch, loss):
        filename = f'checkpoint_epoch_{epoch:03d}.pth'
        save_checkpoint(model, optimizer, None, None, epoch, rank=self.local_rank, folder_path=self.save_path, filename=filename)
        if self.verbose:
            train_hydr.logger.info(f'[EarlyStopping] Epoch {epoch} | Best Score: {self.best_score:.8f}')
