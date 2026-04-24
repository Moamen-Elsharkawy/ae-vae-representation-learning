from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

from .config import ExperimentConfig

AUTOTUNE = tf.data.AUTOTUNE


class KLAnealingCallback(keras.callbacks.Callback):
    def __init__(self, anneal_epochs: int = 10) -> None:
        super().__init__()
        self.anneal_epochs = max(1, anneal_epochs)

    def on_epoch_begin(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        if self.anneal_epochs == 1:
            weight = 1.0
        else:
            weight = min(1.0, epoch / float(self.anneal_epochs - 1))

        if hasattr(self.model, "kl_weight"):
            self.model.kl_weight.assign(weight)


def add_gaussian_noise(images: tf.Tensor, noise_std: float) -> tf.Tensor:
    noise = tf.random.normal(shape=tf.shape(images), stddev=noise_std, dtype=images.dtype)
    return tf.clip_by_value(images + noise, 0.0, 1.0)


def train_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: ExperimentConfig,
) -> keras.callbacks.History:
    model_kind = getattr(model, "model_kind", "autoencoder")
    epochs = config.vae_epochs if model_kind == "vae" else config.ae_epochs

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            verbose=1,
        ),
    ]

    if model_kind == "vae":
        callbacks.insert(0, KLAnealingCallback(anneal_epochs=10))

    prepared_train = prepare_training_dataset(train_ds, model_kind=model_kind, noise_std=config.noise_std)
    prepared_val = prepare_training_dataset(val_ds, model_kind=model_kind, noise_std=config.noise_std)

    return model.fit(
        prepared_train,
        validation_data=prepared_val,
        epochs=epochs,
        callbacks=callbacks,
        verbose=config.train_verbose,
    )


def prepare_training_dataset(
    dataset: tf.data.Dataset,
    model_kind: str,
    noise_std: float,
) -> tf.data.Dataset:
    if model_kind == "vae":
        dataset = dataset.map(
            lambda sample: sample["image"],
            num_parallel_calls=AUTOTUNE,
        )
    elif model_kind == "denoising":
        dataset = dataset.map(
            lambda sample: (add_gaussian_noise(sample["image"], noise_std), sample["image"]),
            num_parallel_calls=AUTOTUNE,
        )
    else:
        dataset = dataset.map(
            lambda sample: (sample["image"], sample["image"]),
            num_parallel_calls=AUTOTUNE,
        )

    return dataset.prefetch(AUTOTUNE)
