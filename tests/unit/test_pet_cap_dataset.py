from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from tmpl.datasets import load_template_dataset


def test_load_template_dataset_with_name_column(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    imgs_dir = root / "imgs"
    csv_dir = root / "csv"
    model_dir = root / "model"
    imgs_dir.mkdir(parents=True)
    csv_dir.mkdir()
    model_dir.mkdir()

    sample_template = np.zeros((16, 16), dtype=np.uint8)
    cv2.circle(sample_template, (8, 8), 6, 255, thickness=-1)
    cv2.imwrite(str(model_dir / "0.png"), sample_template)

    image = np.zeros((32, 32), dtype=np.uint8)
    cv2.rectangle(image, (8, 12), (20, 24), 255, -1)
    cv2.imwrite(str(imgs_dir / "frame0.png"), image)

    csv_path = csv_dir / "result0.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "x", "y", "theta", "msec"])
        writer.writerow(["frame0.png", "16.0", "20.0", "0.0", "5.0"])

    dataset = load_template_dataset(root=root)

    assert dataset.template_path.exists()
    assert len(dataset.records) == 1
    record = dataset.records[0]
    assert record.image_path.exists()
    assert record.name == "frame0.png"
    assert record.x == pytest.approx(16.0)
    assert record.y == pytest.approx(20.0)
    assert record.theta == pytest.approx(0.0)
    assert record.target_msec == pytest.approx(5.0)


def test_load_template_dataset_with_path_column(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    imgs_dir = root / "imgs"
    csv_dir = root / "csv"
    model_dir = root / "model"
    imgs_dir.mkdir(parents=True)
    csv_dir.mkdir()
    model_dir.mkdir()

    (model_dir / "template.png").write_bytes(b"binary")

    absolute_image_path = imgs_dir / "frame0.png"
    absolute_image_path.write_bytes(b"img")

    with (csv_dir / "result0.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "x", "y", "theta", "msec"])
        writer.writerow([str(absolute_image_path), "1.0", "2.0", "3.0", ""])

    dataset = load_template_dataset(root=root)

    assert dataset.template_path.name == "template.png"
    assert len(dataset.records) == 1
    record = dataset.records[0]
    assert record.image_path == absolute_image_path
    assert record.name == "frame0.png"
    assert record.x == pytest.approx(1.0)
    assert record.y == pytest.approx(2.0)
    assert record.theta == pytest.approx(3.0)
    assert record.target_msec == pytest.approx(0.0)
