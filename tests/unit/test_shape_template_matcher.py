from __future__ import annotations

import cv2
import numpy as np
import pytest

from tmpl.matching.shape import ShapeTemplateMatcher


def test_shape_matcher_locates_template_without_rotation() -> None:
    matcher = ShapeTemplateMatcher(canny_low=30, canny_high=90, blur_kernel=3, dilation_iterations=0)

    template = np.zeros((15, 15), dtype=np.uint8)
    cv2.rectangle(template, (3, 3), (12, 12), 255, thickness=-1)

    image = np.zeros((60, 80), dtype=np.uint8)
    image[22:37, 34:49] = template

    compiled = matcher.compile_template(template, angles=[0.0])
    result = matcher.match(image, compiled)

    assert pytest.approx(result.center[0], abs=0.5) == 41.5
    assert pytest.approx(result.center[1], abs=0.5) == 29.5
    assert pytest.approx(result.score, abs=1e-4) == result.score  # ensure finite
    assert result.angle == 0.0
    assert result.template_size == (template.shape[1], template.shape[0])
    assert result.template_edges.shape == (template.shape[0], template.shape[1])


def test_shape_matcher_detects_rotated_shape() -> None:
    matcher = ShapeTemplateMatcher(canny_low=20, canny_high=80, blur_kernel=3, dilation_iterations=1)

    template = np.zeros((20, 20), dtype=np.uint8)
    cv2.rectangle(template, (8, 2), (12, 18), 255, thickness=-1)
    cv2.rectangle(template, (2, 14), (18, 18), 255, thickness=-1)

    rotated = matcher._rotate_image(template, 45.0)  # type: ignore[attr-defined]
    canvas = np.zeros((80, 80), dtype=np.uint8)
    offset_y, offset_x = 30, 28
    h, w = rotated.shape
    canvas[offset_y : offset_y + h, offset_x : offset_x + w] = rotated

    compiled = matcher.compile_template(template, angles=[0.0, 15.0, 30.0, 45.0, 60.0])
    result = matcher.match(canvas, compiled)

    assert pytest.approx(result.angle, abs=1.0) == 45.0
    expected_center = (offset_x + w / 2.0, offset_y + h / 2.0)
    assert pytest.approx(result.center[0], abs=1.0) == expected_center[0]
    assert pytest.approx(result.center[1], abs=1.0) == expected_center[1]
    h_edges, w_edges = result.template_edges.shape
    assert (w_edges, h_edges) == result.template_size
    assert h_edges > 0 and w_edges > 0


def test_compile_template_rejects_empty_angles() -> None:
    matcher = ShapeTemplateMatcher()
    template = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        matcher.compile_template(template, angles=[])
