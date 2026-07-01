import dataclasses
from typing import Generator

import torch

from aurora.batch import Batch
from train_hydr.aurora_lite import Aurora
from train_hydr.decoder import Decoder
from train_hydr.data import dict_to_device


def _prepare_batch(modelAurora: Aurora, batch: Batch) -> Batch:
    """Bring a raw dataset batch into the right form/dtype/device for the model."""
    batch = modelAurora.batch_transform_hook(batch)  # This might modify the available variables.
    # Use an arbitary parameter of the model to derive the data type and device.
    p = next(modelAurora.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(modelAurora.patch_size)
    batch = batch.to(p.device)
    return batch


def rollout(modelAurora: Aurora, modelDecoder: Decoder, batch: Batch, input_dict, steps: int, lsm_criterion=0.5) -> Generator[Batch, None, None]:
    """Perform a roll-out to make long-term predictions.

    Args:
        model (:class:`aurora_lite.Aurora`): The modified model to roll out.
        batch (:class:`aurora.batch.Batch`): The batch to start the roll-out from.
        steps (int): The number of roll-out steps.

    Yields:
        :class:`aurora.batch.Batch`: The prediction of Aurora after every step.
        `dict`: The prediction of MLP after every step.
    """
    # We will need to concatenate data, so ensure that everything is already of the right form.
    p = next(modelAurora.parameters())
    batch = _prepare_batch(modelAurora, batch)
    input_dict = dict_to_device(input_dict, p.device)
    lsm_mask = batch.static_vars['lsm'] >= lsm_criterion

    for _ in range(steps):
        pred_Aurora, lat_dec = modelAurora.forward(batch)
        latent_decoder = lat_dec.detach().clone()
        pred_Decoder = modelDecoder.forward(input_dict, latent_decoder, lsm_mask)

        yield pred_Aurora, pred_Decoder
        # yield pred_Aurora, pred_Decoder, latent_decoder

        # Add the appropriate history so the model can be run on the prediction.
        batch = dataclasses.replace(
            pred_Aurora,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred_Aurora.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred_Aurora.atmos_vars.items()
            },
        )
        input_dict = pred_Decoder


def rollout_era5_forcing(modelAurora: Aurora, modelDecoder: Decoder, dataset, start_idx: int, steps: int, lsm_criterion=0.5) -> Generator[Batch, None, None]:
    """Roll-out with ERA5-forced atmosphere and autoregressive soil moisture.

    Atmospheric/surface inputs to Aurora are re-read from ERA5 at every step
    (teacher forcing) instead of being fed back from the previous prediction.
    Soil moisture (the decoder input) is still propagated autoregressively, i.e.
    the decoder prediction becomes the next decoder input.

    Args:
        modelAurora (:class:`aurora_lite.Aurora`): The modified model to roll out.
        modelDecoder (:class:`decoder.Decoder`): The hydrological decoder head.
        dataset: An indexable dataset (or ``Subset``) where ``dataset[i]`` returns
            ``((batch, input_dict), target_dict)`` for time index ``i``. Must cover
            ``start_idx + steps`` so the full horizon can be forced from ERA5.
        start_idx (int): Index of the first roll-out step.
        steps (int): The number of roll-out steps.

    Yields:
        :class:`aurora.batch.Batch`: The prediction of Aurora after every step.
        `dict`: The prediction of the decoder after every step.
    """
    p = next(modelAurora.parameters())

    # Initial state: atmosphere + soil moisture both come from ERA5 at start_idx.
    (batch, input_dict), _ = dataset[start_idx]
    batch = _prepare_batch(modelAurora, batch)
    input_dict = dict_to_device(input_dict, p.device)
    lsm_mask = batch.static_vars['lsm'] >= lsm_criterion

    for step in range(steps):
        pred_Aurora, lat_dec = modelAurora.forward(batch)
        latent_decoder = lat_dec.detach().clone()
        pred_Decoder = modelDecoder.forward(input_dict, latent_decoder, lsm_mask)

        yield pred_Aurora, pred_Decoder

        # --- Atmosphere: pull the real ERA5 batch for the next step (no feedback). ---
        (batch, _), _ = dataset[start_idx + step + 1]
        batch = _prepare_batch(modelAurora, batch)
        # --- Soil moisture: autoregressive (decoder prediction -> next decoder input). ---
        input_dict = pred_Decoder