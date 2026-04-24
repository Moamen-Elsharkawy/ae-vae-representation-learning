# Video Demonstration Outline

Target length: 2 to 5 minutes

## 1. Opening

- State the assignment goal: compare AE, denoising AE, and VAE for representation learning.
- Mention the dataset source from Google Drive and the use of `tf.data`.

## 2. Show the Pipeline

- Open the notebook and point to the dataset configuration cell.
- Briefly show the modular project structure: data loader, models, training utilities, and plotting helpers.

## 3. Training Results

- Show the AE loss curve and a few AE reconstructions.
- Show the denoising AE results with clean vs noisy vs denoised outputs.
- Show the VAE loss curves, especially reconstruction loss and KL divergence.

## 4. Latent Space and Generation

- Display the 2D and 3D latent projections.
- Show random VAE samples.
- Show latent interpolation between two test images.

## 5. Key Findings

- State which model reconstructed best.
- State which model produced the smoother latent space.
- Mention whether denoising improved robustness.
- Close with one short takeaway about the AE/VAE tradeoff.
