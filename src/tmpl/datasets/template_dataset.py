from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(slots=True)
class GroundTruthRecord:
    """
    Ground truth for a single captured frame.
    """

    name: str
    x: float
    y: float
    theta: float
    image_path: Path
    target_msec: float


@dataclass(slots=True)
class TemplateDataset:
    """
    Dataset bundle exposing template and ground-truth frames.
    """

    template_path: Path
    records: List[GroundTruthRecord]
    root: Path


def load_template_dataset(
    root: Path | str,
    csv_name: str = "result0.csv",
    template_name: str | None = None,
) -> TemplateDataset:
    """
    Load a template-matching dataset with standard folder layout.

    Expected directory structure:
        root/
            imgs/
            model/
            csv/
    """
    root_path = Path(root)
    csv_path = root_path / "csv" / csv_name
    images_dir = root_path / "imgs"
    model_dir = root_path / "model"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    template_path = _resolve_template_path(model_dir, template_name)

    records = _load_records(csv_path, images_dir)
    if not records:
        raise ValueError(f"No samples found in {csv_path}")

    return TemplateDataset(template_path=template_path, records=records, root=root_path)


def _resolve_template_path(model_dir: Path, template_name: Optional[str]) -> Path:
    if template_name:
        template_path = model_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template image not found: {template_path}")
        return template_path

    candidates = [path for path in model_dir.iterdir() if path.is_file()]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No template files found under {model_dir}")
    raise ValueError(
        f"Multiple templates found under {model_dir}. Specify which one with --template-name."
    )


def _load_records(csv_path: Path, images_dir: Path) -> List[GroundTruthRecord]:
    records: List[GroundTruthRecord] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file must include a header row.")

        supports_name = "name" in reader.fieldnames
        supports_path = "path" in reader.fieldnames
        if not supports_name and not supports_path:
            raise ValueError("CSV header must contain either 'name' or 'path' columns.")

        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
                theta = float(row["theta"])
                target_msec_raw = row.get("msec")
                target_msec = float(target_msec_raw) if target_msec_raw not in (None, "") else 0.0
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric values in row: {row}") from exc

            if supports_name and row.get("name"):
                relative = row["name"].strip()
                image_path = images_dir / relative
                name = relative
            elif supports_path and row.get("path"):
                raw_path = Path(row["path"].strip())
                image_path = raw_path if raw_path.is_absolute() else images_dir / raw_path
                name = image_path.name
            else:
                raise ValueError(f"Row missing image reference: {row}")

            if not image_path.exists():
                raise FileNotFoundError(f"Image referenced in CSV missing: {image_path}")

            records.append(
                GroundTruthRecord(
                    name=name,
                    x=x,
                    y=y,
                    theta=theta,
                    image_path=image_path,
                    target_msec=target_msec,
                )
            )

    return records


__all__ = ["GroundTruthRecord", "TemplateDataset", "load_template_dataset"]
