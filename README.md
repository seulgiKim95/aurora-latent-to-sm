# aurora-latent-to-sm

Code for the paper **"Hydrological States Emerge in the Latent Space of a Weather
Foundation Model"** — adapting a weather foundation model to global soil-moisture
forecasting with lightweight decoders trained on its frozen latent space.

📄 **Paper:** TBD *(link to be added)*

<!-- TODO(authors): add the paper title, authors, and link (DOI / arXiv / journal). -->

---

## Overview

Soil moisture (SM) regulates land–atmosphere energy and water exchange, yet
skillful forward forecasts of SM at global scale remain challenging. This work
adapts the [Aurora](https://github.com/microsoft/aurora) weather foundation
model (Microsoft, MIT-licensed) to global SM forecasting by training lightweight,
task-specific decoders on its **frozen** latent space. The backbone supplies its
own atmospheric forcing, enabling autoregressive roll-out across lead times from
the initial condition alone. Two of the decoders additionally integrate the
antecedent SM state to exploit SM memory.

Three decoders are provided:

| Directory | Paper notation | Description                                                                           | Role           |
|-----------|----------------|---------------------------------------------------------------------------------------|----------------|
| `unet`    | UNet_{z,θ}     | U-Net decoder on the latent space with antecedent SM; captures spatial structure      | **Main model** |
| `mlp`     | MLP_z          | Latent-only MLP decoder; tests whether SM is linearly decodable from the latent space | Baseline       |
| `mlp_enc` | MLP_{z,θ}      | MLP decoder with antecedent SM (memory)                                               | Variant        |

## Repository structure

```
aurora-latent-to-sm/
├── common_utils/      # shared library: data loaders, transforms, model helpers
├── models/            # the three decoders, each self-contained
│   ├── unet/          #   main model — train_hydr/ (model + training), config.yaml
│   ├── mlp/           #   baseline
│   └── mlp_enc/       #   variant
├── checkpoints/       # trained-weight download instructions
└── environment.yml    # conda environment (all dependencies)
```

> This repository releases the **model and training code**. The data-preprocessing,
> drought (eSSMI) analysis, and figure-generation notebooks used in the paper are
> maintained separately and available from the authors on request.

## Installation

```bash
conda env create -f environment.yml       # installs all dependencies (torch, microsoft-aurora, hydra, ...)
conda activate aurora
```

The scripts add the repository root to `sys.path` automatically, so the shared
`common_utils` package is importable without a separate install step.

`flash-attn` must be installed manually against your CUDA/torch build, e.g.:

```bash
python -m pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

## Data & checkpoints

The input datasets and trained checkpoints are not stored in this repository.
The inputs are publicly available from their original providers:

- **ERA5** — https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
- **Enhanced SMAP L3 soil moisture (SPL3SMP_E)** — https://nsidc.org/data/spl3smp_e/versions/6
- **ASCAT soil moisture (H121, H139)** — https://hsaf.meteoam.it/Products/Detail?prod=H121 and https://hsaf.meteoam.it/Products/Detail?prod=H139
- **GLDAS porosity** — https://ldas.gsfc.nasa.gov/gldas/soils
- **MODIS LULC (MCD12C1)** — https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd12c1-061
- **ECMWF subseasonal-to-seasonal (S2S)** — https://ecds.ecmwf.int/datasets/s2s-forecasts?tab=download

Trained weights are **available from the authors on request** (see
[`checkpoints/README.md`](checkpoints/README.md)); the models can also be
retrained from the code and the datasets above. After obtaining the data and
weights, set the paths in each model's `train_hydr/config.yaml` (the `paths:`
block) — that file is the single source of truth for all data/checkpoint/output
locations, and the training and evaluation scripts read from it.

## Training

```bash
cd models/unet/train_hydr                    # main model (or models/mlp, models/mlp_enc)
torchrun --nproc_per_node=2 main.py exp_name='unet'
```

`get_metrics.py` (where present) computes bootstrapped skill metrics from the
resulting rollouts.

## Citation

If you use this code, please cite our paper (linked at the top of this README),
the underlying Aurora model, and the ETH Aurora-Lite-Decoder (Lehmann et al.,
arXiv:2506.19088; see [`NOTICE`](NOTICE)).

## License

This project is licensed under the **GNU General Public License v3.0** — see
[`LICENSE`](LICENSE). GPL-3.0 applies because the lightweight decoder is derived
from the GPL-3.0 ETH Aurora-Lite-Decoder; third-party components and their
licenses are detailed in [`NOTICE`](NOTICE).

## Acknowledgements

Built on Microsoft's [Aurora](https://github.com/microsoft/aurora) foundation
model. The lightweight decoder is derived from the ETH-Zürich Aurora-Lite-Decoder
(GPL-3.0; Lehmann et al., arXiv:2506.19088) — see [`NOTICE`](NOTICE).
