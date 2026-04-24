from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentConfig:
    drive_data_root: str = "/content/drive/MyDrive/dsai490_assignment1"
    extract_root: str = "/content/datasets/dsai490_assignment1"
    output_dir: str = "/content/dsai490_outputs"
    batch_size: int = 32
    seed: int = 42
    latent_dim: int = 16
    ae_epochs: int = 30
    vae_epochs: int = 40
    learning_rate: float = 1e-3
    noise_std: float = 0.15
    val_split: float = 0.15
    test_split: float = 0.15
    resize_limit: int = 64
    shuffle_buffer: int = 1024
    mount_drive: bool = True
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")
    input_shape: tuple[int, int, int] | None = None
    image_size: tuple[int, int] | None = None
    channels: int | None = None
    latent_projection_limit: int | None = 1000

    def ensure_output_dir(self) -> Path:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
