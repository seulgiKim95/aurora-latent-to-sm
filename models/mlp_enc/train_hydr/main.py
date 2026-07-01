import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_workspace_dir = os.path.dirname(_current_dir)
if _workspace_dir not in sys.path:
    sys.path.append(_workspace_dir)

import hydra
from omegaconf import OmegaConf
from omegaconf import open_dict
import train_hydr

import functools
import datetime

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from aurora.model.swin3d import Swin3DTransformerBlock

from train_hydr import (
    load_nc_list,
    get_now,
)
from train_hydr.data import load_data
from train_hydr.model import (
    load_aurora_lite_model,
    load_aurora_lite_small_model,
    load_decoder
)
from train_hydr.train import (
    initialize_weights,
    aurora_collate_fn,
    load_lr_scheduler,
    load_optimizer,
    load_scaler,
    load_checkpoint,
    EarlyStopping,
    LossCriterion,
    train_model,
    save_checkpoint,
)


def load_model(key):
    loaders = {
        'load_aurora_lite_model': load_aurora_lite_model,
        'load_aurora_lite_small_model': load_aurora_lite_small_model,
    }
    modelAurora = loaders[key]()
    modelDecoder = load_decoder(modelAurora)
    return modelAurora, modelDecoder


def setup_config(cfg):
    with open_dict(cfg):
        if "LOCAL_RANK" in os.environ:
            cfg.train.local_rank = int(os.environ["LOCAL_RANK"]) # index of the current process among GPU processes (0, 1, 2, 3)
        if "WORLD_SIZE" not in os.environ:
            cfg.train.use_ddp = False
            cfg.train.use_fsdp = False
        cfg.train.use_dist = True if cfg.train.use_ddp or cfg.train.use_fsdp else False
        cfg.train.verbose = True if (cfg.train.use_dist and cfg.train.local_rank == 0) or (not cfg.train.use_dist) else False

    if cfg.train.local_rank == 0:
        OmegaConf.save(config=cfg, f="experiment_config.yaml")
    return cfg


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    train_hydr.config = setup_config(cfg)
    if train_hydr.config.train.verbose:
        now = train_hydr.get_now()
        train_hydr.logger.info(f'===== {now} START =====')
        train_hydr.logger.info(f"config set up")

    try:
        # --- DDP initialization ---
        if train_hydr.config.train.use_dist:
            torch.cuda.set_device(train_hydr.config.train.local_rank) # bind the physical GPU to actually use
            dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60)) # create the process group that lets GPUs exchange data. NCCL is a library optimized for inter-GPU communication on NVIDIA GPUs
            world_size = int(os.environ["WORLD_SIZE"]) # total number of GPUs
        device = torch.device(f'cuda:{train_hydr.config.train.local_rank}')

        torch.manual_seed(train_hydr.config.train.seed)
        torch.cuda.manual_seed(train_hydr.config.train.seed)
        torch.cuda.manual_seed_all(train_hydr.config.train.seed)

        # --- dataset & sampler setup ---
        surf_path_list, atmo_path_list, hydr_path_list, et_path_list, stat_ds = load_nc_list(
            train_hydr.config.paths.base_era5,
            static_path=train_hydr.config.paths.static_file,
            start_date=train_hydr.config.data.start_date,
            end_date=train_hydr.config.data.end_date,
            output_list=('surf', 'atmo', 'hydr', 'et', 'stat'),
        )
        p_path_list, = load_nc_list(
            train_hydr.config.paths.base_mswep,
            start_date=train_hydr.config.data.start_date,
            end_date=train_hydr.config.data.end_date,
            output_list=('p',),
        )
        train_dataset, test_dataset = load_data(surf_path_list, atmo_path_list, hydr_path_list, et_path_list, p_path_list, stat_ds, train_test_ratio=train_hydr.config.data.train_test_ratio)
        if train_hydr.config.train.verbose:
            train_hydr.logger.info(f'train dataset: {len(train_dataset.indices)}')
            train_hydr.logger.info(f'test dataset: {len(test_dataset.indices)}')
 
        if train_hydr.config.train.use_dist:
            sampler = DistributedSampler(train_dataset, shuffle=True)
        else:
            sampler = None
 
        dataloader = DataLoader(
            train_dataset, 
            batch_size=1,  # batch size per GPU
            shuffle=False, # must be False since a sampler is used
            sampler=sampler,
            collate_fn=lambda x: x[0],
#           collate_fn=aurora_collate_fn,
            pin_memory=True, # speeds up CPU-to-GPU transfers (pins the memory)
            drop_last=True,
            num_workers=train_hydr.config.train.num_workers,
            prefetch_factor=train_hydr.config.train.prefetch_factor, # number of batches prefetched per worker
            persistent_workers=train_hydr.config.train.persistent_workers, # keep workers alive between epochs instead of shutting them down
        )
 
        # --- model setup ---
        modelAurora, modelDecoder = load_model(train_hydr.config.model.loader_name)
#       modelDecoder.configure_activation_checkpointing() # activation checkpointing is unnecessary for the MLP architecture
        modelAurora = modelAurora.to(device)
        modelDecoder = modelDecoder.to(device)
        if train_hydr.config.train.use_ddp:
            modelDecoder = DDP(
                modelDecoder,
                device_ids=[train_hydr.config.train.local_rank],
                find_unused_parameters=False
            ) # DDP wrapping
        elif train_hydr.config.train.use_fsdp:
#           auto_wrap_policy = functools.partial( # FSDP, block-based
#               transformer_auto_wrap_policy,
#               transformer_layer_cls={Swin3DTransformerBlock},
#           )
            auto_wrap_policy = functools.partial( # FSDP, size-based
                size_based_auto_wrap_policy,
                min_num_params=train_hydr.config.train.min_num_params # shard parameters with 10M or more
            )
            sharding_strategy = ShardingStrategy.FULL_SHARD # shard both gradients and weights
            modelAurora = FSDP(
                modelAurora,
                auto_wrap_policy=auto_wrap_policy,
                sharding_strategy=sharding_strategy,
                device_id=torch.cuda.current_device(),
                use_orig_params=True # required when using LoRA
            ) # used when the Aurora model is too large to fit on a single GPU at inference
            modelDecoder = FSDP(
                modelDecoder,
                auto_wrap_policy=auto_wrap_policy,
                sharding_strategy=sharding_strategy,
                device_id=torch.cuda.current_device(),
                use_orig_params=True # required when using LoRA
            ) # the Lite Decoder is small so DDP might be better, but FSDP is used for consistency

        # --- loss setup ---
        criterion = LossCriterion(
            land_sea_mask=train_dataset.dataset.get_static_var_data('lsm'),
#           porosity=train_dataset.dataset.get_static_var_data('porosity')
        )
        criterion.set_land_weights(device)
#       criterion.set_static_device(device)
        criterion.set_latitude_weights(train_dataset.dataset._get_meta_data(0), device)

        # optimizer
        optimizer = load_optimizer(modelDecoder)

        # scaler
        scaler = load_scaler('grad') if train_hydr.config.train.use_scaler else None

        # learning rate, scheduler
        total_steps = len(dataloader) * train_hydr.config.train.epochs
        scheduler = load_lr_scheduler(
            optimizer=optimizer,
            warmup_steps=train_hydr.config.train.warmup_steps,
            total_steps=total_steps,
            lr_decay_value=train_hydr.config.train.lr_decay,
        )
 
        # early stopping
        early_stopping = EarlyStopping(
            patience=train_hydr.config.train.early_stopping_patience,
            save_path=train_hydr.config.paths.save_dir,
            rank=train_hydr.config.train.local_rank,
            verbose=train_hydr.config.train.verbose,
        )

        # load checkpoint
        if train_hydr.config.model.checkpoint is not None:
            epoch_init, modelDecoder, optimizer, _, _ = load_checkpoint(
                device,
                train_hydr.config.model.checkpoint,
                modelDecoder,
                optimizer=optimizer,
                scheduler=None,
                scaler=None
            )
        else:
            epoch_init = 0

        # --- training loop ---
        # train
        for epoch in range(epoch_init +1, train_hydr.config.train.epochs +1):
            if train_hydr.config.train.use_dist:
                sampler.set_epoch(epoch) # set so that shuffling differs each epoch

            avg_loss = train_model( # train
                modelAurora=modelAurora,
                modelDecoder=modelDecoder,
                dataloader=dataloader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                scaler=scaler,
                scheduler=scheduler,
                verbose=train_hydr.config.train.verbose,
            )

            if train_hydr.config.train.use_dist: # synchronize loss over GPUs
                avg_loss_tensor = torch.tensor(avg_loss).cuda()
                dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
                avg_loss = avg_loss_tensor.item()

            if train_hydr.config.train.verbose: # loss record
                train_hydr.logger.info(f"[Loss] Epoch {epoch} | Avg Loss: {avg_loss:.8f}")

            if train_hydr.config.train.model_save:
                early_stopping(modelDecoder, optimizer, epoch, avg_loss) # checkpoint_epoch_{}.pth

                save_checkpoint( # last_checkpoint.pth
                    model=modelDecoder,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    rank=train_hydr.config.train.local_rank,
                    folder_path=train_hydr.config.paths.save_dir,
                    filename='.last_checkpoint.pth'
                )

                if early_stopping.early_stop:
                    if train_hydr.config.train.verbose:
                        train_hydr.logger.info(f"[EarlyStopping] Epoch {epoch} | Early stopping triggered. Stop training.")
                    break

    except Exception as e:
        raise e
    finally:
        if train_hydr.config.train.use_dist:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

