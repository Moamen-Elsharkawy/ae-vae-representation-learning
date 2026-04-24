# DSAI 490 Assignment 1

This project implements three representation-learning models for image reconstruction and generation:

- A convolutional Autoencoder (AE)
- A denoising Autoencoder
- A Variational Autoencoder (VAE)

The code is organized so the reusable logic lives in Python modules under `src/representation_learning`, while the experiment flow lives in the Colab notebook under `notebooks/`.

## Project Structure

```text
src/representation_learning/
  config.py
  data.py
  models.py
  training.py
  evaluation.py
  visualization.py
notebooks/
  dsai490_assignment1_colab.ipynb
report/
  technical_report.md
  video_demo_outline.md
tests/
  test_pipeline.py
```

## Expected Workflow

1. Upload the required image dataset archive to Google Drive.
2. Open the notebook in Colab.
3. Update `drive_data_root` so it points to the dataset archive or extracted dataset folder in Drive.
4. Run the notebook cells in order.
5. Export figures and metrics for the report and the video demo.

## Notes About the Data Pipeline

- The loader uses `tf.data` end to end.
- If the dataset already contains `train/`, `val/`, and `test/` folders, those splits are used directly.
- If no official split exists, the loader creates a `70/15/15` split with `seed=42`.
- Folder names are preserved as labels for visualization when they exist, but training remains unsupervised.
- If sample images are already square and at most `64x64`, the native size is kept; otherwise the loader resizes to `64x64`.

## Local Sanity Check

The repository includes a small test suite that creates a temporary image dataset and checks:

- dataset loading
- model output shapes
- VAE loss tracking
- a short overfit-style step on a single mini-batch

Run it with:

```powershell
python -m unittest discover -s tests -v
```

## Dependencies

Install the required packages with:

```powershell
python -m pip install -r requirements.txt
```
