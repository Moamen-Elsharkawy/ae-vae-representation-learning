from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import ExperimentConfig


class Sampling(layers.Layer):
    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VariationalAutoencoder(keras.Model):
    def __init__(self, encoder: keras.Model, decoder: keras.Model, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="kl_weight")
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.model_kind = "vae"

    @property
    def metrics(self) -> list[keras.metrics.Metric]:
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        _, _, z = self.encoder(inputs, training=training)
        return self.decoder(z, training=training)

    def compute_losses(
        self,
        inputs: tf.Tensor,
        reconstruction: tf.Tensor,
        z_mean: tf.Tensor,
        z_log_var: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(inputs - reconstruction), axis=(1, 2, 3))
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        total_loss = reconstruction_loss + self.kl_weight * kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, data: tf.Tensor | tuple[tf.Tensor, tf.Tensor]) -> dict[str, tf.Tensor]:
        inputs = data[0] if isinstance(data, tuple) else data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(inputs, training=True)
            reconstruction = self.decoder(z, training=True)
            total_loss, reconstruction_loss, kl_loss = self.compute_losses(
                inputs,
                reconstruction,
                z_mean,
                z_log_var,
            )

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: tf.Tensor | tuple[tf.Tensor, tf.Tensor]) -> dict[str, tf.Tensor]:
        inputs = data[0] if isinstance(data, tuple) else data
        z_mean, z_log_var, z = self.encoder(inputs, training=False)
        reconstruction = self.decoder(z, training=False)
        total_loss, reconstruction_loss, kl_loss = self.compute_losses(
            inputs,
            reconstruction,
            z_mean,
            z_log_var,
        )

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {metric.name: metric.result() for metric in self.metrics}


@dataclass(slots=True)
class EncoderArtifacts:
    model: keras.Model
    encoded_shape: tuple[int, int, int]


def build_autoencoder(
    config: ExperimentConfig,
) -> tuple[keras.Model, keras.Model, keras.Model]:
    encoder_artifacts = build_encoder(config, variational=False)
    decoder = build_decoder(config, encoded_shape=encoder_artifacts.encoded_shape)

    inputs = keras.Input(shape=config.input_shape, name="ae_input")
    latent = encoder_artifacts.model(inputs)
    outputs = decoder(latent)

    autoencoder = keras.Model(inputs, outputs, name="autoencoder")
    autoencoder.model_kind = "autoencoder"
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=keras.losses.MeanSquaredError(),
    )
    return encoder_artifacts.model, decoder, autoencoder


def build_denoising_autoencoder(
    config: ExperimentConfig,
) -> tuple[keras.Model, keras.Model, keras.Model]:
    encoder_artifacts = build_encoder(config, variational=False)
    decoder = build_decoder(config, encoded_shape=encoder_artifacts.encoded_shape)

    inputs = keras.Input(shape=config.input_shape, name="denoising_input")
    latent = encoder_artifacts.model(inputs)
    outputs = decoder(latent)

    denoising_autoencoder = keras.Model(inputs, outputs, name="denoising_autoencoder")
    denoising_autoencoder.model_kind = "denoising"
    denoising_autoencoder.noise_std = config.noise_std
    denoising_autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=keras.losses.MeanSquaredError(),
    )
    return encoder_artifacts.model, decoder, denoising_autoencoder


def build_vae(
    config: ExperimentConfig,
) -> tuple[keras.Model, keras.Model, VariationalAutoencoder]:
    encoder_artifacts = build_encoder(config, variational=True)
    decoder = build_decoder(config, encoded_shape=encoder_artifacts.encoded_shape)
    vae = VariationalAutoencoder(encoder=encoder_artifacts.model, decoder=decoder, name="vae")
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate))
    return encoder_artifacts.model, decoder, vae


def build_encoder(config: ExperimentConfig, variational: bool) -> EncoderArtifacts:
    if config.input_shape is None:
        raise ValueError("`config.input_shape` must be set before building the models.")

    encoder_inputs = keras.Input(shape=config.input_shape, name="encoder_input")
    x = encoder_inputs
    for filters in (32, 64, 128):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
        )(x)

    encoded_shape = tuple(int(dim) for dim in x.shape[1:])
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)

    if variational:
        z_mean = layers.Dense(config.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(config.latent_dim, name="z_log_var")(x)
        z = Sampling(name="sampling")((z_mean, z_log_var))
        encoder = keras.Model(
            encoder_inputs,
            [z_mean, z_log_var, z],
            name="encoder",
        )
    else:
        z = layers.Dense(config.latent_dim, name="latent_vector")(x)
        encoder = keras.Model(encoder_inputs, z, name="encoder")

    return EncoderArtifacts(model=encoder, encoded_shape=encoded_shape)


def build_decoder(config: ExperimentConfig, encoded_shape: tuple[int, int, int]) -> keras.Model:
    if config.input_shape is None:
        raise ValueError("`config.input_shape` must be set before building the decoder.")

    latent_inputs = keras.Input(shape=(config.latent_dim,), name="decoder_input")
    x = layers.Dense(int(np.prod(encoded_shape)), activation="relu")(latent_inputs)
    x = layers.Reshape(encoded_shape)(x)

    for filters in (128, 64, 32):
        x = layers.Conv2DTranspose(
            filters=filters,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
        )(x)

    x = layers.Resizing(config.input_shape[0], config.input_shape[1])(x)
    outputs = layers.Conv2DTranspose(
        filters=config.input_shape[2],
        kernel_size=3,
        padding="same",
        activation="sigmoid",
        name="decoder_output",
    )(x)

    return keras.Model(latent_inputs, outputs, name="decoder")
