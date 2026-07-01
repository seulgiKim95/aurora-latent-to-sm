# Trained checkpoints

The trained model weights (`mlp_enc`, `mlp`, `unet`) are **available from the
authors on request**. They are not stored in this repository.

The models can also be reproduced from scratch with the training code
(`models/<name>/train_hydr/main.py`) and the input datasets listed in the main
[README](../README.md) — the pretrained Aurora base weights are downloaded
automatically by the `microsoft-aurora` package.

Once you have a checkpoint, point `paths.experiment_root` in that model's
`train_hydr/config.yaml` at its location.
