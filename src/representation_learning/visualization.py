from __future__ import annotations

import math
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_dataset_samples(
    dataset: tf.data.Dataset,
    metadata,
    rows: int = 2,
    columns: int = 5,
    title: str = "Dataset Samples",
) -> None:
    batch = next(iter(dataset))
    images = batch["image"].numpy()
    labels = batch["label"].numpy()

    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.4, rows * 2.4))
    axes = np.atleast_1d(axes).reshape(rows, columns)
    figure.suptitle(title)

    label_names = getattr(metadata, "label_names", ())
    max_items = min(rows * columns, len(images))
    for index in range(rows * columns):
        axis = axes.flat[index]
        axis.axis("off")
        if index >= max_items:
            continue

        axis.imshow(to_display_image(images[index]))
        axis.set_title(resolve_label_title(labels[index], label_names), fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_training_curves(history, title: str = "Training History") -> None:
    history_dict = history.history
    metric_names = [name for name in history_dict if not name.startswith("val_")]

    figure, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 4))
    axes = np.atleast_1d(axes)

    for axis, metric_name in zip(axes, metric_names, strict=False):
        axis.plot(history_dict[metric_name], label=f"train {metric_name}")
        val_name = f"val_{metric_name}"
        if val_name in history_dict:
            axis.plot(history_dict[val_name], label=f"val {metric_name}")
        axis.set_title(metric_name.replace("_", " ").title())
        axis.set_xlabel("Epoch")
        axis.legend()

    figure.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_vae_losses(history, title: str = "VAE Loss Curves") -> None:
    keys = ["loss", "reconstruction_loss", "kl_loss"]
    figure, axes = plt.subplots(1, len(keys), figsize=(15, 4))

    for axis, key in zip(axes, keys, strict=False):
        axis.plot(history.history.get(key, []), label=f"train {key}")
        axis.plot(history.history.get(f"val_{key}", []), label=f"val {key}")
        axis.set_title(key.replace("_", " ").title())
        axis.set_xlabel("Epoch")
        axis.legend()

    figure.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_reconstruction_grid(
    example_batches: dict[str, np.ndarray],
    title: str = "Reconstruction Results",
    max_items: int = 6,
) -> None:
    original = example_batches["original"][:max_items]
    reconstructed = example_batches["reconstructed"][:max_items]
    noisy = example_batches.get("noisy")
    item_count = min(len(original), len(reconstructed), max_items)
    if item_count == 0:
        return

    row_count = 3 if noisy is not None else 2
    figure, axes = plt.subplots(row_count, item_count, figsize=(item_count * 2.2, row_count * 2.4))
    axes = np.array(axes, dtype=object).reshape(row_count, item_count)
    figure.suptitle(title)

    for index in range(item_count):
        axes[0, index].imshow(to_display_image(original[index]))
        axes[0, index].set_title("Original")
        axes[0, index].axis("off")

        if noisy is not None:
            axes[1, index].imshow(to_display_image(noisy[index]))
            axes[1, index].set_title("Noisy")
            axes[1, index].axis("off")
            axes[2, index].imshow(to_display_image(reconstructed[index]))
            axes[2, index].set_title("Denoised")
            axes[2, index].axis("off")
        else:
            axes[1, index].imshow(to_display_image(reconstructed[index]))
            axes[1, index].set_title("Reconstructed")
            axes[1, index].axis("off")

    plt.tight_layout()
    plt.show()


def plot_model_reconstructions(
    original: np.ndarray,
    ae_reconstructed: np.ndarray,
    vae_reconstructed: np.ndarray,
    title: str = "AE vs VAE Reconstructions",
    max_items: int = 6,
) -> None:
    max_items = min(max_items, len(original), len(ae_reconstructed), len(vae_reconstructed))
    if max_items == 0:
        return
    figure, axes = plt.subplots(3, max_items, figsize=(max_items * 2.2, 7))
    axes = np.array(axes, dtype=object).reshape(3, max_items)
    figure.suptitle(title)

    row_titles = ("Original", "AE", "VAE")
    rows = (original, ae_reconstructed, vae_reconstructed)
    for row_index, row_images in enumerate(rows):
        for col_index in range(max_items):
            axes[row_index, col_index].imshow(to_display_image(row_images[col_index]))
            axes[row_index, col_index].set_title(row_titles[row_index])
            axes[row_index, col_index].axis("off")

    plt.tight_layout()
    plt.show()


def plot_latent_projection(
    projected_points: np.ndarray,
    labels: np.ndarray | None,
    label_names: Iterable[str] | None = None,
    dims: int = 2,
    title: str = "Latent Space Projection",
) -> None:
    colors = color_values(labels, len(projected_points))

    if dims == 2:
        figure, axis = plt.subplots(figsize=(7, 6))
        scatter = axis.scatter(
            projected_points[:, 0],
            projected_points[:, 1],
            c=colors,
            cmap="tab10",
            alpha=0.75,
        )
        axis.set_xlabel("PC 1")
        axis.set_ylabel("PC 2")
        axis.set_title(title)
        maybe_add_legend(axis, scatter, labels, label_names)
        plt.tight_layout()
        plt.show()
        return

    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        projected_points[:, 0],
        projected_points[:, 1],
        projected_points[:, 2],
        c=colors,
        cmap="tab10",
        alpha=0.75,
    )
    axis.set_xlabel("PC 1")
    axis.set_ylabel("PC 2")
    axis.set_zlabel("PC 3")
    axis.set_title(title)
    maybe_add_legend(axis, scatter, labels, label_names)
    plt.tight_layout()
    plt.show()


def plot_generated_samples(
    images: np.ndarray,
    title: str = "Generated Samples",
    columns: int = 5,
) -> None:
    rows = math.ceil(len(images) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.3, rows * 2.3))
    axes = np.atleast_1d(axes).reshape(rows, columns)
    figure.suptitle(title)

    for index in range(rows * columns):
        axis = axes.flat[index]
        axis.axis("off")
        if index < len(images):
            axis.imshow(to_display_image(images[index]))

    plt.tight_layout()
    plt.show()


def plot_interpolation_grid(images: np.ndarray, title: str = "Latent Interpolation") -> None:
    figure, axes = plt.subplots(1, len(images), figsize=(len(images) * 2.2, 2.5))
    axes = np.atleast_1d(axes)
    figure.suptitle(title)

    for axis, image in zip(axes, images, strict=False):
        axis.imshow(to_display_image(image))
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def to_display_image(image: np.ndarray) -> np.ndarray:
    if image.shape[-1] == 1:
        return image[..., 0]
    return image


def resolve_label_title(label: int, label_names: Iterable[str]) -> str:
    label_names = list(label_names)
    if label < 0 or label >= len(label_names):
        return "unlabeled"
    return str(label_names[label])


def color_values(labels: np.ndarray | None, count: int) -> np.ndarray:
    if labels is None:
        return np.arange(count, dtype=float)
    if len(labels) == 0:
        return np.arange(count, dtype=float)

    color_source = labels.astype(float)
    if np.all(color_source < 0):
        color_source = np.arange(len(labels), dtype=float)
    return color_source


def maybe_add_legend(axis, scatter, labels, label_names) -> None:
    if labels is None or label_names is None:
        return

    valid_labels = sorted({int(label) for label in labels if int(label) >= 0})
    if not valid_labels:
        return

    handles = []
    names = list(label_names)
    for label in valid_labels:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=scatter.cmap(scatter.norm(label)),
                markersize=8,
            )
        )
    axis.legend(handles, [names[label] for label in valid_labels], title="Class", loc="best")
