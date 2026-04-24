from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .training import add_gaussian_noise


def evaluate_reconstruction(
    model: keras.Model,
    test_ds: tf.data.Dataset,
    metadata: object | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    mse_metric = keras.metrics.Mean(name="mse")
    ssim_metric = keras.metrics.Mean(name="ssim")
    model_kind = getattr(model, "model_kind", "autoencoder")
    noise_std = float(getattr(model, "noise_std", 0.15))

    example_batches: dict[str, np.ndarray] = {}

    for batch in test_ds:
        clean_images = batch["image"]
        labels = batch["label"].numpy()

        if model_kind == "denoising":
            inputs = add_gaussian_noise(clean_images, noise_std)
            example_key = "noisy"
        else:
            inputs = clean_images
            example_key = "original"

        reconstructions = model(inputs, training=False)

        batch_mse = tf.reduce_mean(tf.square(clean_images - reconstructions))
        batch_ssim = tf.reduce_mean(tf.image.ssim(clean_images, reconstructions, max_val=1.0))
        mse_metric.update_state(batch_mse)
        ssim_metric.update_state(batch_ssim)

        if not example_batches:
            example_batches["original"] = clean_images.numpy()
            example_batches["reconstructed"] = reconstructions.numpy()
            example_batches["labels"] = labels
            if example_key == "noisy":
                example_batches["noisy"] = inputs.numpy()

    metrics = {
        "mse": float(mse_metric.result().numpy()),
        "ssim": float(ssim_metric.result().numpy()),
    }
    return metrics, example_batches


def project_latent(
    encoder: keras.Model,
    test_ds: tf.data.Dataset,
    method: str = "pca",
    dims: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    latents: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for batch in test_ds:
        encoded = encoder(batch["image"], training=False)
        latent = select_latent_tensor(encoded)
        latents.append(latent.numpy())
        labels.append(batch["label"].numpy())

    latent_array = np.concatenate(latents, axis=0)
    label_array = np.concatenate(labels, axis=0)

    if method.lower() != "pca":
        raise ValueError("Only PCA projection is implemented in this project.")

    projected = pca_project(latent_array, dims=dims)
    return projected, label_array


def sample_vae(
    decoder: keras.Model,
    num_samples: int,
    latent_dim: int,
    seed: int,
) -> np.ndarray:
    generator = tf.random.Generator.from_seed(seed)
    latent_samples = generator.normal(shape=(num_samples, latent_dim))
    decoded = decoder(latent_samples, training=False)
    return decoded.numpy()


def interpolate_latent(
    encoder: keras.Model,
    decoder: keras.Model,
    image_a: np.ndarray,
    image_b: np.ndarray,
    steps: int = 8,
) -> np.ndarray:
    image_a = np.expand_dims(image_a, axis=0)
    image_b = np.expand_dims(image_b, axis=0)

    latent_a = select_latent_tensor(encoder(image_a, training=False)).numpy()[0]
    latent_b = select_latent_tensor(encoder(image_b, training=False)).numpy()[0]
    interpolation = np.linspace(latent_a, latent_b, num=steps)
    decoded = decoder(interpolation, training=False)
    return decoded.numpy()


def select_latent_tensor(encoded: tf.Tensor | tuple[tf.Tensor, ...] | list[tf.Tensor]) -> tf.Tensor:
    if isinstance(encoded, (tuple, list)):
        return encoded[0]
    return encoded


def pca_project(latent_array: np.ndarray, dims: int) -> np.ndarray:
    if dims not in {2, 3}:
        raise ValueError("PCA projection only supports 2D or 3D outputs.")

    centered = latent_array - latent_array.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh[:dims].T

    if projected.shape[1] < dims:
        padding = np.zeros((projected.shape[0], dims - projected.shape[1]), dtype=projected.dtype)
        projected = np.concatenate([projected, padding], axis=1)

    return projected
