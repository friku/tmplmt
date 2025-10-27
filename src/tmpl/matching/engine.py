from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


MatchResult = Tuple[Tuple[int, int], float, np.ndarray]


@dataclass(slots=True)
class TemplateMatcher:
    """
    Thin wrapper around OpenCV's matchTemplate with convenience scoring.
    """

    method: int = cv2.TM_CCOEFF_NORMED

    def match(self, image: np.ndarray, template: np.ndarray) -> MatchResult:
        """
        Locate the best matching region.

        Returns ((x, y), score, heatmap).
        """
        if image.ndim != template.ndim:
            raise ValueError("image and template dimensionality must match")

        result = cv2.matchTemplate(image, template, self.method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if self.method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            location = min_loc
            score = 1.0 - min_val
        else:
            location = max_loc
            score = max_val

        return location, float(score), result

