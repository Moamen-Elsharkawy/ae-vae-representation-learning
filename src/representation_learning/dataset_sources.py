from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pandas as pd

REGIONS = ("elbow", "finger", "forearm", "hand", "humerus", "shoulder", "wrist")
ABNORMAL_PATTERN = re.compile(r"\babnormal(?:ity|ities)?\b", re.IGNORECASE)
NORMAL_PATTERN = re.compile(r"\bnormal\b", re.IGNORECASE)
NEGATIVE_PATTERN = re.compile(r"\bnegative\b", re.IGNORECASE)
POSITIVE_PATTERN = re.compile(r"\bpositive\b", re.IGNORECASE)


def prepare_medifics_mura_subset(
    target_root: str | Path,
    dataset_id: str = "MEDIFICS/MURADATASETSU",
    force: bool = False,
) -> Path:
    target_root = Path(target_root)
    metadata_path = target_root / "metadata.csv"
    if metadata_path.exists() and not force:
        return target_root

    if force and target_root.exists():
        shutil.rmtree(target_root)

    target_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    patch_multiprocess_resource_tracker()

    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split="train")

    rows: list[dict[str, str]] = []
    for row in dataset:
        conversation_text = flatten_conversation(row["conversation"])
        answer_text = flatten_answers(row["conversation"])
        region = infer_region(conversation_text)
        if region is None:
            continue

        status = infer_status(answer_text)
        image = row["image"].convert("L")

        image_dir = target_root / region / status
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{row['id']}.png"
        image.save(image_path)

        rows.append(
            {
                "id": str(row["id"]),
                "region": region,
                "status": status,
                "path": str(image_path),
                "source_dataset": dataset_id,
            }
        )

    if not rows:
        raise ValueError(f"No usable images were prepared from {dataset_id}.")

    pd.DataFrame(rows).sort_values(["region", "status", "id"]).to_csv(metadata_path, index=False)
    return target_root


def patch_multiprocess_resource_tracker() -> None:
    try:
        from multiprocess import resource_tracker
    except ImportError:
        return

    resource_tracker.ResourceTracker.__del__ = lambda self: None


def flatten_conversation(conversation: dict[str, list[dict[str, str]]]) -> str:
    parts: list[str] = []
    for item in conversation.get("data", []):
        question = item.get("question", "")
        answer = item.get("answer", "")
        parts.append(f"{question} {answer}".strip())
    return " ".join(parts).strip()


def flatten_answers(conversation: dict[str, list[dict[str, str]]]) -> str:
    return " ".join(item.get("answer", "").strip() for item in conversation.get("data", [])).strip()


def infer_region(text: str) -> str | None:
    lowered = text.lower()
    for region in REGIONS:
        if re.search(rf"\b{re.escape(region)}\b", lowered):
            return region
    return None


def infer_status(text: str) -> str:
    if POSITIVE_PATTERN.search(text):
        return "abnormal"
    if NEGATIVE_PATTERN.search(text):
        return "normal"
    if re.search(r"\bno abnormal(?:ity|ities)\b", text, re.IGNORECASE):
        return "normal"
    if ABNORMAL_PATTERN.search(text):
        return "abnormal"
    if NORMAL_PATTERN.search(text):
        return "normal"
    return "normal"
