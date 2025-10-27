import numpy as np
import pytest

import cv2

from tmpl import TemplateMatcher


def test_match_returns_expected_location_for_exact_match() -> None:
    base = np.zeros((10, 10), dtype=np.uint8)
    base[3:6, 4:7] = 255
    template = base[3:6, 4:7]

    matcher = TemplateMatcher(method=cv2.TM_SQDIFF_NORMED)
    location, score, heatmap = matcher.match(base, template)

    assert location == (4, 3)
    assert score == pytest.approx(1.0, abs=1e-6)
    assert heatmap.shape == (base.shape[0] - template.shape[0] + 1, base.shape[1] - template.shape[1] + 1)
