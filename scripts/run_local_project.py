from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from representation_learning.config import ExperimentConfig  # noqa: E402
from representation_learning.dataset_sources import prepare_medifics_mura_subset  # noqa: E402
from representation_learning.runtime import configure_quiet_runtime  # noqa: E402
from representation_learning.runner import run_all_region_experiments  # noqa: E402


def main() -> None:
    configure_quiet_runtime()

    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)

    config = ExperimentConfig()
    prepare_medifics_mura_subset(
        target_root=config.data_root,
        dataset_id=config.hf_dataset_id,
    )

    summary = run_all_region_experiments(config)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {Path(config.output_dir).resolve()}")


if __name__ == "__main__":
    main()
