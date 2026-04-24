# Technical Report Draft

## Title

Representation Learning with Autoencoders and Variational Autoencoders

## 1. Model Architectures

The Autoencoder uses a convolutional encoder-decoder structure. The encoder compresses each image through three convolutional blocks and projects the feature map into a low-dimensional latent vector. The decoder mirrors that process with transposed convolutions to reconstruct the original image.

The denoising Autoencoder uses the same architecture, but it receives noisy images as input and learns to recover the clean target image. This setup measures how stable the learned representation remains under input corruption.

The Variational Autoencoder extends the encoder so it predicts a latent mean vector and a latent log-variance vector. A sampling layer applies the reparameterization trick, and the decoder reconstructs images from the sampled latent code. The training objective combines reconstruction loss with KL divergence, and KL annealing is used during the early epochs for stability.

## 2. AE vs VAE

The standard Autoencoder focuses only on minimizing reconstruction error, so it usually produces sharper reconstructions on seen patterns. However, its latent space is not explicitly regularized, which makes random sampling less reliable.

The VAE learns a smoother and more structured latent space because the KL term pushes the latent distribution toward a Gaussian prior. In practice, that usually improves interpolation and sample generation, even if reconstruction quality is slightly lower than the deterministic AE.

## 3. Latent Space Behavior

The latent vectors are projected with PCA into two and three dimensions. If labels are available through folder names, clusters can be interpreted against those categories. When labels are not available, the projection is still useful for checking whether visually similar samples occupy nearby regions of the latent space.

The expected pattern is that the AE produces compact but sometimes irregular clusters, while the VAE produces a smoother and more continuous latent arrangement because of its probabilistic regularization.

## 4. Results and Observations

Replace the placeholders below after training on the required dataset:

- AE test MSE: `[fill after run]`
- AE test SSIM: `[fill after run]`
- Denoising AE test MSE: `[fill after run]`
- Denoising AE test SSIM: `[fill after run]`
- VAE test MSE: `[fill after run]`
- VAE test SSIM: `[fill after run]`

Key observations to include:

- Which model reconstructed fine details better
- Whether denoising improved robustness on corrupted inputs
- Whether the VAE generated realistic samples
- Whether the latent interpolations changed smoothly between endpoints

## 5. Conclusion

The assignment shows the tradeoff between deterministic reconstruction and probabilistic generation. The AE is expected to perform better on direct reconstruction, while the VAE is expected to provide a more organized latent space and more meaningful sample generation. The denoising experiment adds a robustness perspective by testing whether the latent space can preserve useful structure under noisy inputs.
