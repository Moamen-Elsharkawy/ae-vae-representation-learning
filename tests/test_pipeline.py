from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from representation_learning.config import ExperimentConfig  # noqa: E402
from representation_learning.runtime import configure_quiet_runtime  # noqa: E402

configure_quiet_runtime()

import tensorflow as tf

from representation_learning.data import build_datasets  # noqa: E402
from representation_learning.dataset_sources import flatten_conversation, infer_region, infer_status  # noqa: E402
from representation_learning.models import build_autoencoder, build_vae  # noqa: E402
from representation_learning.training import train_model  # noqa: E402


tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name) / "dataset"
        self._create_fake_dataset()

        self.config = ExperimentConfig(
            data_root=str(self.data_root),
            extract_root=str(Path(self.temp_dir.name) / "extracted"),
            output_dir=str(Path(self.temp_dir.name) / "outputs"),
            batch_size=4,
            seed=42,
            latent_dim=8,
            ae_epochs=2,
            vae_epochs=1,
            learning_rate=1e-3,
            noise_std=0.1,
            mount_drive=False,
        )

    def tearDown(self) -> None:
        tf.keras.backend.clear_session()
        self.temp_dir.cleanup()

    def test_build_datasets_uses_tf_data(self) -> None:
        train_ds, val_ds, test_ds, metadata = build_datasets(self.config)

        batch = next(iter(train_ds))
        self.assertEqual(metadata.input_shape, (32, 32, 3))
        self.assertTrue(metadata.has_labels)
        self.assertEqual(tuple(batch["image"].shape[1:]), (32, 32, 3))
        self.assertEqual(batch["image"].dtype, tf.float32)
        self.assertGreaterEqual(float(tf.reduce_min(batch["image"]).numpy()), 0.0)
        self.assertLessEqual(float(tf.reduce_max(batch["image"]).numpy()), 1.0)
        self.assertGreater(metadata.train_samples, 0)
        self.assertGreater(metadata.val_samples, 0)
        self.assertGreater(metadata.test_samples, 0)
        self.assertIsInstance(val_ds, tf.data.Dataset)
        self.assertIsInstance(test_ds, tf.data.Dataset)

    def test_autoencoder_output_shape_matches_input(self) -> None:
        train_ds, _, _, _ = build_datasets(self.config)
        _, _, autoencoder = build_autoencoder(self.config)

        batch = next(iter(train_ds))
        outputs = autoencoder(batch["image"], training=False)
        self.assertEqual(tuple(outputs.shape), tuple(batch["image"].shape))

    def test_vae_tracks_kl_loss(self) -> None:
        train_ds, val_ds, _, _ = build_datasets(self.config)
        encoder, _, vae = build_vae(self.config)

        batch = next(iter(train_ds))
        z_mean, z_log_var, z = encoder(batch["image"], training=False)
        self.assertEqual(z_mean.shape[-1], self.config.latent_dim)
        self.assertEqual(z_log_var.shape[-1], self.config.latent_dim)
        self.assertEqual(z.shape[-1], self.config.latent_dim)

        history = train_model(vae, train_ds.take(1), val_ds.take(1), self.config)
        self.assertIn("kl_loss", history.history)
        self.assertIn("val_kl_loss", history.history)
        self.assertGreaterEqual(history.history["kl_loss"][-1], 0.0)

    def test_single_batch_overfit_tendency(self) -> None:
        train_ds, _, _, _ = build_datasets(self.config)
        _, _, autoencoder = build_autoencoder(self.config)

        batch = next(iter(train_ds))
        images = batch["image"]
        losses = [float(autoencoder.train_on_batch(images, images)) for _ in range(5)]
        self.assertLess(losses[-1], losses[0])

    def test_region_and_status_extraction(self) -> None:
        conversation = {
            "data": [
                {
                    "question": "What type of upper extremity radiograph is shown?",
                    "answer": "The image shows a shoulder radiograph.",
                },
                {
                    "question": "Is it normal?",
                    "answer": "The shoulder has abnormalities.",
                },
            ]
        }
        text = flatten_conversation(conversation)
        self.assertEqual(infer_region(text), "shoulder")
        self.assertEqual(infer_status(" ".join(item["answer"] for item in conversation["data"])), "abnormal")

    def _create_fake_dataset(self) -> None:
        rng = np.random.default_rng(7)
        for split_name in ("train", "val", "test"):
            for class_name in ("class_a", "class_b"):
                split_dir = self.data_root / split_name / class_name
                split_dir.mkdir(parents=True, exist_ok=True)
                for index in range(6):
                    image = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
                    if class_name == "class_b":
                        image[:, :16] = 255 - image[:, :16]
                    image_path = split_dir / f"sample_{index}.png"
                    Image.fromarray(image).save(image_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
