from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from representation_learning.config import ExperimentConfig  # noqa: E402
from representation_learning.dataset_sources import prepare_medifics_mura_subset  # noqa: E402


def main() -> None:
    config = ExperimentConfig()
    prepared_root = prepare_medifics_mura_subset(
        target_root=config.data_root,
        dataset_id=config.hf_dataset_id,
    )
    print(f"Prepared dataset at: {prepared_root}")


if __name__ == "__main__":
    main()
