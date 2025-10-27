from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path
from typing import Sequence
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import cv2
import numpy as np

from tmpl.datasets import load_template_dataset
from tmpl.io import load_grayscale
from tmpl.matching.shape import ShapeMatchResult, ShapeTemplateMatcher


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate shape-based template matching accuracy and throughput.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("pet_cap", "kiban"),
        default="pet_cap",
        help="Named dataset preset to evaluate (controls defaults for paths and templates).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory containing imgs/, csv/, and model/ folders. Defaults depend on --dataset.",
    )
    parser.add_argument(
        "--angle-step",
        type=float,
        default=3.0,
        help="Step size (degrees) for generating template rotations.",
    )
    parser.add_argument(
        "--canny-low",
        type=int,
        default=40,
        help="Lower threshold for Canny edge detector.",
    )
    parser.add_argument(
        "--canny-high",
        type=int,
        default=120,
        help="Upper threshold for Canny edge detector.",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=5,
        help="Odd kernel size for Gaussian blur applied before Canny.",
    )
    parser.add_argument(
        "--dilation-iterations",
        type=int,
        default=1,
        help="Number of morphological dilation passes applied to edge maps.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="result0.csv",
        help="CSV filename that stores ground-truth coordinates.",
    )
    parser.add_argument(
        "--template-name",
        type=str,
        default=None,
        help="Template filename located inside the model/ directory. "
        "If omitted, defaults depend on --dataset or the loader will auto-detect when a single file is present.",
    )
    parser.add_argument(
        "--angle-min",
        type=float,
        default=-180.0,
        help="Minimum rotation angle to evaluate (inclusive).",
    )
    parser.add_argument(
        "--angle-max",
        type=float,
        default=180.0,
        help="Maximum rotation angle to evaluate (exclusive).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/match_viz"),
        help="Directory where annotated match visuals will be written.",
    )
    parser.add_argument(
        "--downscale-factor",
        type=float,
        default=None,
        help="Downscale factor (0 < f <= 1) for coarse localization.",
    )
    parser.add_argument(
        "--refine-angle-window",
        type=float,
        default=None,
        help="Angle window (degrees) around coarse pose to search in fine stage.",
    )
    parser.add_argument(
        "--refine-roi-scale",
        type=float,
        default=None,
        help="Multiplier applied to template size to define the fine-search ROI.",
    )
    parser.add_argument(
        "--refine-roi-padding",
        type=int,
        default=None,
        help="Extra padding (pixels) added to the ROI half-dimensions.",
    )
    return parser.parse_args()


def build_angles(angle_min: float, angle_max: float, angle_step: float) -> Sequence[float]:
    if angle_step <= 0:
        raise ValueError("angle_step must be positive")
    values = []
    current = angle_min
    # Use strict < to avoid floating point accumulation from overshooting.
    while current < angle_max:
        values.append(round(current, 6))
        current += angle_step
    return values


def angle_error(pred: float, target: float) -> float:
    diff = (pred - target + 180.0) % 360.0 - 180.0
    return abs(diff)


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-180, 180) range for reporting.
    """
    return (angle + 180.0) % 360.0 - 180.0


def render_visualization(
    image: np.ndarray,
    result: ShapeMatchResult,
    record_name: str,
    target_x: float,
    target_y: float,
    target_theta: float,
    predicted_theta: float,
    pixel_error: float,
    theta_error: float,
) -> np.ndarray:
    """
    Overlay template edges, predicted pose, and ground truth markers."""
    annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = annotated.copy()

    x, y = map(int, result.top_left)
    width, height = result.template_size
    mask = result.template_edges > 0.1

    if 0 <= y < annotated.shape[0] and 0 <= x < annotated.shape[1]:
        y_end = min(y + height, annotated.shape[0])
        x_end = min(x + width, annotated.shape[1])
        roi = overlay[y:y_end, x:x_end]
        mask_crop = mask[: y_end - y, : x_end - x]
        if roi.shape[:2] == mask_crop.shape:
            roi[mask_crop.astype(bool)] = (0, 0, 255)

    annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)

    pred_center = (int(round(result.center[0])), int(round(result.center[1])))
    target_center = (int(round(target_x)), int(round(target_y)))
    cv2.drawMarker(annotated, pred_center, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
    cv2.circle(annotated, target_center, 6, (0, 215, 255), 2)

    mask_points = np.column_stack(np.where(mask))
    if mask_points.size >= 6:
        points = mask_points[:, ::-1].astype(np.float32)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box[:, 0] += x
        box[:, 1] += y
        box = box.astype(int)
        cv2.polylines(annotated, [box], True, (255, 0, 0), 2)

    label = (
        f"{record_name} | score={result.score:.3f} | "
        f"pred=({result.center[0]:.1f},{result.center[1]:.1f},{predicted_theta:.1f}°) | "
        f"target=({target_x:.1f},{target_y:.1f},{target_theta:.1f}°)"
    )
    cv2.putText(annotated, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(
        annotated,
        f"err_px={pixel_error:.2f} err_deg={theta_error:.2f}",
        (8, annotated.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (240, 240, 240),
        2,
        lineType=cv2.LINE_AA,
    )

    return annotated


def evaluate() -> None:
    args = parse_arguments()
    dataset_defaults = {
        "pet_cap": {
            "root": Path("data/tmpl_imgs/pet_cap"),
            "template": "0.png",
            "downscale_factor": 0.5,
            "refine_angle_window": 10.0,
            "refine_roi_scale": 2.0,
            "refine_roi_padding": 12,
        },
        "kiban": {
            "root": Path("data/tmpl_imgs/kiban"),
            "template": None,
            "downscale_factor": 1.0,
            "refine_angle_window": 180.0,
            "refine_roi_scale": 2.0,
            "refine_roi_padding": 12,
        },
    }

    defaults = dataset_defaults[args.dataset]
    data_root = Path(args.data_root) if args.data_root else defaults["root"]
    template_name = args.template_name if args.template_name is not None else defaults["template"]
    downscale_factor = args.downscale_factor if args.downscale_factor is not None else defaults["downscale_factor"]
    refine_angle_window = (
        args.refine_angle_window if args.refine_angle_window is not None else defaults["refine_angle_window"]
    )
    refine_roi_scale = args.refine_roi_scale if args.refine_roi_scale is not None else defaults["refine_roi_scale"]
    refine_roi_padding = (
        args.refine_roi_padding if args.refine_roi_padding is not None else defaults["refine_roi_padding"]
    )

    dataset = load_template_dataset(
        root=data_root,
        csv_name=args.csv_name,
        template_name=template_name,
    )

    template_image = load_grayscale(dataset.template_path)
    matcher = ShapeTemplateMatcher(
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        blur_kernel=args.blur_kernel,
        dilation_iterations=args.dilation_iterations,
        downscale_factor=downscale_factor,
        refine_angle_window=refine_angle_window,
        refine_roi_scale=refine_roi_scale,
        refine_roi_padding=refine_roi_padding,
    )

    angles = build_angles(args.angle_min, args.angle_max, args.angle_step)
    compiled = matcher.compile_template(template_image, angles)

    pixel_errors: list[float] = []
    angle_errors: list[float] = []
    durations_ms: list[float] = []
    latency_slack_ms: list[float] = []
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for record in dataset.records:
        image = load_grayscale(record.image_path)

        start = time.perf_counter()
        result = matcher.match(image, compiled)
        end = time.perf_counter()

        center_x, center_y = result.center
        pixel_error = math.hypot(center_x - record.x, center_y - record.y)
        predicted_theta_raw = -result.angle
        predicted_theta = normalize_angle(predicted_theta_raw)
        theta_error = angle_error(predicted_theta, record.theta)
        duration_ms = (end - start) * 1000.0
        target_msec = record.target_msec
        slack_ms = duration_ms - target_msec

        pixel_errors.append(pixel_error)
        angle_errors.append(theta_error)
        durations_ms.append(duration_ms)
        latency_slack_ms.append(slack_ms)

        annotated = render_visualization(
            image,
            result,
            record.name,
            record.x,
            record.y,
            record.theta,
            predicted_theta,
            pixel_error,
            theta_error,
        )
        output_path = output_dir / f"{Path(record.name).stem}_viz.png"
        cv2.imwrite(str(output_path), annotated)

        if target_msec > 0:
            target_info = f"time={duration_ms:7.2f}ms (target={target_msec:5.1f}ms, +{slack_ms:6.2f})"
        else:
            target_info = f"time={duration_ms:7.2f}ms"

        print(
            f"{record.name:35s} | "
            f"score={result.score: .4f} | "
            f"pred=({center_x:7.2f},{center_y:7.2f},{predicted_theta:7.2f}) | "
            f"target=({record.x:7.2f},{record.y:7.2f},{record.theta:7.2f}) | "
            f"err_px={pixel_error:6.3f} | "
            f"err_deg={theta_error:6.3f} | "
            f"{target_info} | "
            f"viz={output_path.name}"
        )

    print("\nSummary")
    print("-" * 72)
    print(f"Frames evaluated : {len(dataset.records)}")
    print(f"Angles searched  : {len(angles)} ({angles[0]}° -> {angles[-1]}° step {args.angle_step}°)")
    print(f"Pixel error (px) : mean={statistics.fmean(pixel_errors):.3f}, median={statistics.median(pixel_errors):.3f}, max={max(pixel_errors):.3f}")
    print(f"Angle error (°)  : mean={statistics.fmean(angle_errors):.3f}, median={statistics.median(angle_errors):.3f}, max={max(angle_errors):.3f}")
    print(f"Latency (ms)     : mean={statistics.fmean(durations_ms):.2f}, median={statistics.median(durations_ms):.2f}, min={min(durations_ms):.2f}, max={max(durations_ms):.2f}")
    if latency_slack_ms:
        print(f"Latency slack    : mean={statistics.fmean(latency_slack_ms):.2f}ms, median={statistics.median(latency_slack_ms):.2f}ms, max={max(latency_slack_ms):.2f}ms")


if __name__ == "__main__":
    evaluate()
