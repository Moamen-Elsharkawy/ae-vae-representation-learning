from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentConfig:
    data_root: str = "data/prepared/medifics_mura"
    drive_data_root: str | None = None
    extract_root: str = "data/extracted"
    output_dir: str = "artifacts/local_medical_runs"
    dataset_source: str = "medifics_mura_subset"
    hf_dataset_id: str = "MEDIFICS/MURADATASETSU"
    batch_size: int = 16
    seed: int = 42
    latent_dim: int = 16
    ae_epochs: int = 5
    vae_epochs: int = 5
    learning_rate: float = 1e-3
    noise_std: float = 0.15
    val_split: float = 0.15
    test_split: float = 0.15
    resize_limit: int = 64
    shuffle_buffer: int = 1024
    mount_drive: bool = False
    train_verbose: int = 1
    conv_filters: tuple[int, int, int] = (16, 32, 64)
    dense_units: int = 64
    selected_regions: tuple[str, ...] = (
        "elbow",
        "finger",
        "forearm",
        "hand",
        "humerus",
        "shoulder",
        "wrist",
    )
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")
    input_shape: tuple[int, int, int] | None = None
    image_size: tuple[int, int] | None = None
    channels: int | None = None
    latent_projection_limit: int | None = 1000

    def ensure_output_dir(self) -> Path:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def resolved_data_root(self) -> str:
        return self.data_root or self.drive_data_root or ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
