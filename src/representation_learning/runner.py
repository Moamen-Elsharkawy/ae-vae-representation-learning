from __future__ import annotations

import gc
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
import pandas as pd
import tensorflow as tf

from .config import ExperimentConfig
from .data import build_datasets
from .evaluation import evaluate_reconstruction, interpolate_latent, project_latent, sample_vae
from .models import build_autoencoder, build_vae
from .training import train_model
from .visualization import (
    plot_dataset_samples,
    plot_generated_samples,
    plot_interpolation_grid,
    plot_latent_projection,
    plot_model_reconstructions,
    plot_reconstruction_grid,
    plot_training_curves,
    plot_vae_losses,
)

matplotlib.use("Agg")


def run_all_region_experiments(config: ExperimentConfig) -> pd.DataFrame:
    prepared_root = Path(config.resolved_data_root())
    artifact_root = config.ensure_output_dir()

    results: list[dict[str, float | int | str]] = []
    for region in config.selected_regions:
        region_root = prepared_root / region
        if not region_root.exists():
            continue
        region_results = run_region_experiment(region, region_root, artifact_root / region, config)
        results.append(region_results)

    if not results:
        raise ValueError(f"No anatomical region folders were found under {prepared_root}.")

    summary = pd.DataFrame(results).sort_values("region").reset_index(drop=True)
    summary.to_csv(artifact_root / "region_results.csv", index=False)
    return summary


def run_region_experiment(
    region: str,
    region_root: Path,
    region_output_dir: Path,
    config: ExperimentConfig,
) -> dict[str, float | int | str]:
    region_output_dir.mkdir(parents=True, exist_ok=True)
    region_config = replace(
        config,
        data_root=str(region_root),
        output_dir=str(region_output_dir),
        mount_drive=False,
        input_shape=None,
        image_size=None,
        channels=None,
    )

    train_ds, val_ds, test_ds, metadata = build_datasets(region_config)

    plot_dataset_samples(
        train_ds,
        metadata,
        rows=2,
        columns=5,
        title=f"{region.title()} Dataset Samples",
        save_path=region_output_dir / "dataset_samples.png",
        show=False,
    )

    ae_encoder, _, autoencoder = build_autoencoder(region_config)
    ae_history = train_model(autoencoder, train_ds, val_ds, region_config)
    plot_training_curves(
        ae_history,
        title=f"{region.title()} AE Training Curves",
        save_path=region_output_dir / "ae_training_curves.png",
        show=False,
    )
    ae_metrics, ae_examples = evaluate_reconstruction(autoencoder, test_ds, metadata)
    plot_reconstruction_grid(
        ae_examples,
        title=f"{region.title()} AE Reconstructions",
        save_path=region_output_dir / "ae_reconstructions.png",
        show=False,
    )

    vae_encoder, vae_decoder, vae = build_vae(region_config)
    vae_history = train_model(vae, train_ds, val_ds, region_config)
    plot_vae_losses(
        vae_history,
        title=f"{region.title()} VAE Loss Curves",
        save_path=region_output_dir / "vae_loss_curves.png",
        show=False,
    )
    vae_metrics, vae_examples = evaluate_reconstruction(vae, test_ds, metadata)
    plot_reconstruction_grid(
        vae_examples,
        title=f"{region.title()} VAE Reconstructions",
        save_path=region_output_dir / "vae_reconstructions.png",
        show=False,
    )

    plot_model_reconstructions(
        ae_examples["original"],
        ae_examples["reconstructed"],
        vae_examples["reconstructed"],
        title=f"{region.title()} AE vs VAE",
        max_items=6,
        save_path=region_output_dir / "ae_vs_vae.png",
        show=False,
    )

    projected_ae, labels = project_latent(ae_encoder, test_ds, dims=2)
    plot_latent_projection(
        projected_ae,
        labels,
        metadata.label_names,
        dims=2,
        title=f"{region.title()} AE Latent Space",
        save_path=region_output_dir / "ae_latent_2d.png",
        show=False,
    )

    projected_vae_2d, labels = project_latent(vae_encoder, test_ds, dims=2)
    plot_latent_projection(
        projected_vae_2d,
        labels,
        metadata.label_names,
        dims=2,
        title=f"{region.title()} VAE Latent Space (2D)",
        save_path=region_output_dir / "vae_latent_2d.png",
        show=False,
    )

    projected_vae_3d, labels = project_latent(vae_encoder, test_ds, dims=3)
    plot_latent_projection(
        projected_vae_3d,
        labels,
        metadata.label_names,
        dims=3,
        title=f"{region.title()} VAE Latent Space (3D)",
        save_path=region_output_dir / "vae_latent_3d.png",
        show=False,
    )

    generated_images = sample_vae(vae_decoder, num_samples=10, latent_dim=region_config.latent_dim, seed=region_config.seed)
    plot_generated_samples(
        generated_images,
        title=f"{region.title()} VAE Generated Samples",
        columns=5,
        save_path=region_output_dir / "vae_generated_samples.png",
        show=False,
    )

    test_batch = next(iter(test_ds))
    if len(test_batch["image"]) >= 2:
        interpolation = interpolate_latent(
            encoder=vae_encoder,
            decoder=vae_decoder,
            image_a=test_batch["image"][0].numpy(),
            image_b=test_batch["image"][1].numpy(),
            steps=8,
        )
        plot_interpolation_grid(
            interpolation,
            title=f"{region.title()} VAE Interpolation",
            save_path=region_output_dir / "vae_interpolation.png",
            show=False,
        )

    history_payload = {
        "ae": ae_history.history,
        "vae": vae_history.history,
        "ae_metrics": ae_metrics,
        "vae_metrics": vae_metrics,
        "metadata": metadata.to_dict(),
    }
    with (region_output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(history_payload, file, indent=2)

    tf.keras.backend.clear_session()
    gc.collect()

    return {
        "region": region,
        "train_samples": metadata.train_samples,
        "val_samples": metadata.val_samples,
        "test_samples": metadata.test_samples,
        "ae_mse": ae_metrics["mse"],
        "ae_ssim": ae_metrics["ssim"],
        "vae_mse": vae_metrics["mse"],
        "vae_ssim": vae_metrics["ssim"],
    }
