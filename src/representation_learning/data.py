from __future__ import annotations

import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import tensorflow as tf

from .config import ExperimentConfig

AUTOTUNE = tf.data.AUTOTUNE
TRAIN_NAMES = {"train", "training"}
VAL_NAMES = {"val", "valid", "validation"}
TEST_NAMES = {"test", "testing"}


@dataclass(slots=True)
class ExampleRecord:
    path: str
    label_name: str | None


@dataclass(slots=True)
class DatasetMetadata:
    data_root: str
    input_shape: tuple[int, int, int]
    image_size: tuple[int, int]
    channels: int
    kept_native_size: bool
    label_names: tuple[str, ...]
    has_labels: bool
    train_samples: int
    val_samples: int
    test_samples: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_datasets(
    config: ExperimentConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, DatasetMetadata]:
    maybe_mount_google_drive(config)
    data_root = resolve_dataset_root(config)

    split_roots = detect_split_roots(data_root)
    if split_roots:
        train_records = collect_records(split_roots["train"], split_roots["train"])
        test_records = collect_records(split_roots["test"], split_roots["test"])
        if "val" in split_roots:
            val_records = collect_records(split_roots["val"], split_roots["val"])
        else:
            train_records, val_records = split_records(
                train_records,
                val_fraction=config.val_split,
                seed=config.seed,
            )
    else:
        all_records = collect_records(data_root, data_root)
        train_records, val_records, test_records = split_records_three_way(
            all_records,
            val_fraction=config.val_split,
            test_fraction=config.test_split,
            seed=config.seed,
        )

    all_records = list(train_records) + list(val_records) + list(test_records)
    if not all_records:
        raise ValueError(f"No image files were found under: {data_root}")

    label_names = tuple(
        sorted({record.label_name for record in all_records if record.label_name})
    )
    label_to_index = {label_name: index for index, label_name in enumerate(label_names)}

    probe_paths = [Path(record.path) for record in all_records[: min(16, len(all_records))]]
    channels = infer_channels(Path(all_records[0].path))
    kept_native_size, image_size = infer_image_size(
        probe_paths,
        channels=channels,
        resize_limit=config.resize_limit,
    )
    input_shape = (*image_size, channels)

    config.channels = channels
    config.image_size = image_size
    config.input_shape = input_shape

    train_ds = build_tf_dataset(
        train_records,
        label_to_index=label_to_index,
        batch_size=config.batch_size,
        image_size=image_size,
        channels=channels,
        shuffle=True,
        seed=config.seed,
        shuffle_buffer=config.shuffle_buffer,
    )
    val_ds = build_tf_dataset(
        val_records,
        label_to_index=label_to_index,
        batch_size=config.batch_size,
        image_size=image_size,
        channels=channels,
        shuffle=False,
        seed=config.seed,
        shuffle_buffer=config.shuffle_buffer,
    )
    test_ds = build_tf_dataset(
        test_records,
        label_to_index=label_to_index,
        batch_size=config.batch_size,
        image_size=image_size,
        channels=channels,
        shuffle=False,
        seed=config.seed,
        shuffle_buffer=config.shuffle_buffer,
    )

    metadata = DatasetMetadata(
        data_root=str(data_root),
        input_shape=input_shape,
        image_size=image_size,
        channels=channels,
        kept_native_size=kept_native_size,
        label_names=label_names,
        has_labels=bool(label_names),
        train_samples=len(train_records),
        val_samples=len(val_records),
        test_samples=len(test_records),
    )
    return train_ds, val_ds, test_ds, metadata


def maybe_mount_google_drive(config: ExperimentConfig) -> None:
    if not config.mount_drive:
        return

    drive_root = Path(config.drive_data_root)
    if drive_root.exists():
        return

    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        return

    drive.mount("/content/drive", force_remount=False)


def resolve_dataset_root(config: ExperimentConfig) -> Path:
    source_root = Path(config.drive_data_root)
    extract_root = Path(config.extract_root)

    if source_root.is_dir():
        if contains_images(source_root, config.image_extensions):
            return source_root

        archives = find_archives(source_root)
        if len(archives) == 1:
            return extract_archive_once(archives[0], extract_root, config.image_extensions)

    if source_root.is_file():
        return extract_archive_once(source_root, extract_root, config.image_extensions)

    if extract_root.exists() and contains_images(extract_root, config.image_extensions):
        return extract_root

    raise FileNotFoundError(
        "Could not resolve the dataset path. Point `drive_data_root` to either the "
        "extracted dataset folder or the archive file stored in Google Drive."
    )


def find_archives(root: Path) -> list[Path]:
    archives: list[Path] = []
    for pattern in ("*.zip", "*.tar", "*.tar.gz", "*.tgz"):
        archives.extend(root.rglob(pattern))
    return sorted(set(archives))


def contains_images(root: Path, extensions: Iterable[str]) -> bool:
    extension_set = {suffix.lower() for suffix in extensions}
    return any(path.suffix.lower() in extension_set for path in root.rglob("*") if path.is_file())


def extract_archive_once(archive_path: Path, extract_root: Path, extensions: Iterable[str]) -> Path:
    if extract_root.exists() and contains_images(extract_root, extensions):
        return detect_dataset_root(extract_root, extensions)

    extract_root.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive_path), str(extract_root))

    candidate_root = detect_dataset_root(extract_root, extensions)
    if candidate_root is None:
        raise ValueError(f"Archive extracted but no image files were found in: {extract_root}")

    return candidate_root


def detect_dataset_root(root: Path, extensions: Iterable[str]) -> Path:
    if detect_split_roots(root):
        return root

    image_root = first_image_root(root, extensions)
    if image_root is None:
        raise ValueError(f"No image files were found under: {root}")

    return image_root


def first_image_root(root: Path, extensions: Iterable[str]) -> Path | None:
    extension_set = {suffix.lower() for suffix in extensions}
    image_paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extension_set
    )
    if not image_paths:
        return None

    first_image = image_paths[0]
    relative_parts = first_image.relative_to(root).parts
    if len(relative_parts) <= 1:
        return root

    top_level_dirs = [child for child in root.iterdir() if child.is_dir()]
    if len(top_level_dirs) == 1:
        return top_level_dirs[0]

    return root


def detect_split_roots(root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for child in root.iterdir():
        if not child.is_dir():
            continue

        name = child.name.lower()
        if name in TRAIN_NAMES:
            mapping["train"] = child
        elif name in VAL_NAMES:
            mapping["val"] = child
        elif name in TEST_NAMES:
            mapping["test"] = child

    if "train" in mapping and "test" in mapping:
        return mapping
    return {}


def collect_records(root: Path, label_root: Path) -> list[ExampleRecord]:
    image_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    records: list[ExampleRecord] = []
    for image_path in image_paths:
        relative_path = image_path.relative_to(label_root)
        label_name = relative_path.parts[0] if len(relative_path.parts) > 1 else None
        records.append(ExampleRecord(path=str(image_path), label_name=label_name))
    return records


def split_records(
    records: list[ExampleRecord],
    val_fraction: float,
    seed: int,
) -> tuple[list[ExampleRecord], list[ExampleRecord]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_count = min(len(shuffled) - 1, max(1, int(len(shuffled) * val_fraction)))
    return shuffled[val_count:], shuffled[:val_count]


def split_records_three_way(
    records: list[ExampleRecord],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[ExampleRecord], list[ExampleRecord], list[ExampleRecord]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    total_count = len(shuffled)
    if total_count < 3:
        raise ValueError("At least three images are required to create train/val/test splits.")

    test_count = min(total_count - 2, max(1, int(total_count * test_fraction)))
    remaining_after_test = total_count - test_count
    val_count = min(remaining_after_test - 1, max(1, int(total_count * val_fraction)))

    test_records = shuffled[:test_count]
    val_records = shuffled[test_count : test_count + val_count]
    train_records = shuffled[test_count + val_count :]

    if not train_records:
        raise ValueError("The dataset split left no training samples. Reduce val/test fractions.")

    return train_records, val_records, test_records


def infer_channels(image_path: Path) -> int:
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(image_bytes, channels=0, expand_animations=False)
    shape = tuple(image.numpy().shape)
    if len(shape) != 3:
        raise ValueError(f"Expected an image tensor with rank 3, got shape {shape} for {image_path}.")
    return int(shape[2])


def infer_image_size(
    probe_paths: list[Path],
    channels: int,
    resize_limit: int,
) -> tuple[bool, tuple[int, int]]:
    observed_sizes: list[tuple[int, int]] = []
    for image_path in probe_paths:
        image_bytes = tf.io.read_file(str(image_path))
        image = tf.io.decode_image(image_bytes, channels=channels, expand_animations=False)
        height, width = tuple(image.numpy().shape[:2])
        observed_sizes.append((int(height), int(width)))

    unique_sizes = set(observed_sizes)
    if len(unique_sizes) == 1:
        height, width = observed_sizes[0]
        if height == width and min(height, width) <= resize_limit:
            return True, (height, width)

    return False, (resize_limit, resize_limit)


def build_tf_dataset(
    records: list[ExampleRecord],
    label_to_index: dict[str, int],
    batch_size: int,
    image_size: tuple[int, int],
    channels: int,
    shuffle: bool,
    seed: int,
    shuffle_buffer: int,
) -> tf.data.Dataset:
    if not records:
        raise ValueError("A dataset split is empty. Check the dataset path and split logic.")

    paths = [record.path for record in records]
    labels = [label_to_index.get(record.label_name, -1) for record in records]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=max(shuffle_buffer, len(paths)),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda path, label: load_example(path, label, image_size, channels),
        num_parallel_calls=AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def load_example(
    path: tf.Tensor,
    label: tf.Tensor,
    image_size: tuple[int, int],
    channels: int,
) -> dict[str, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=channels, expand_animations=False)
    image.set_shape([None, None, channels])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size, antialias=True)
    return {"image": image, "label": label, "path": path}
