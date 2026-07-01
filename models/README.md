# Models

Each subdirectory is a self-contained model built around the training package
(`train_hydr/`) and its Hydra `config.yaml`. They share the top-level
[`common_utils`](../common_utils) library and the pip-installed `aurora` package.

`mlp/` and `unet/` also include an example notebook (`analysis_*.ipynb`) that
walks through the full workflow — load a checkpoint, run the roll-out, and
visualize the predictions.

| Dir       | Model                                   | Role           |
|-----------|-----------------------------------------|----------------|
| `unet`    | U-Net regression                        | **Main model** |
| `mlp_enc` | Aurora-Lite-Decoder + MLP encoder head  | Variant        |
| `mlp`     | Aurora-Lite-Decoder (MLP decoder head)  | Baseline       |


## Training

```bash
cd models/<name>/train_hydr
torchrun --nproc_per_node=2 main.py exp_name='<name>'
```

All paths (inputs, checkpoints, outputs) are read from each model's
`train_hydr/config.yaml` (`paths:` block) — edit it to your local locations.
`train_hydr/log/` is kept as a placeholder.

## Metrics

`get_metrics.py` (in `mlp/` and `unet/`) computes bootstrapped skill metrics
(ACC, MAE, RMSE, bias) from the roll-out outputs.
