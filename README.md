# DSAI 490 Assignment 1

This repository trains a separate Autoencoder and Variational Autoencoder for each anatomical region in a lightweight MURA-based upper-extremity radiograph dataset.

## Recommended Data

For a full medical-imaging version of the assignment, the strongest source is Stanford's MURA dataset because it already organizes upper-extremity radiographs into seven anatomical regions: elbow, finger, forearm, hand, humerus, shoulder, and wrist.

For a runnable local workflow in this repository, the code prepares a smaller public MURA-derived subset from Hugging Face:

- Official MURA overview: <https://aimi.stanford.edu/datasets/mura-msk-xrays>
- Local runnable subset used by the scripts: <https://huggingface.co/datasets/MEDIFICS/MURADATASETSU>

The preparation script downloads the subset, extracts the anatomical region from the metadata, converts every study image to grayscale PNG, and writes a local folder structure like this:

```text
data/prepared/medifics_mura/
  elbow/
    abnormal/
    normal/
  finger/
    abnormal/
    normal/
  ...
```

## Project Structure

```text
scripts/
  prepare_medifics_mura.py
  run_local_project.py
src/representation_learning/
  config.py
  data.py
  dataset_sources.py
  models.py
  training.py
  evaluation.py
  visualization.py
  runner.py
tests/
  test_pipeline.py
```

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepare The Dataset

```powershell
python .\scripts\prepare_medifics_mura.py
```

This creates the prepared per-region dataset under `data/prepared/medifics_mura/`.

## Run The Full Local Project

```powershell
python .\scripts\run_local_project.py
```

The run trains:

- one AE per anatomical region
- one VAE per anatomical region

and saves outputs to `artifacts/local_medical_runs/`.

Each region folder contains:

- dataset samples
- AE training curves
- VAE loss curves
- AE and VAE reconstruction grids
- AE vs VAE comparison plots
- 2D and 3D latent visualizations
- generated VAE samples
- latent interpolation examples
- `metrics.json`

The script also writes a summary table to:

```text
artifacts/local_medical_runs/region_results.csv
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests use a temporary synthetic dataset to validate the data pipeline and the core model code without downloading the real medical dataset.
