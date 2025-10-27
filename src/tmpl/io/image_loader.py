from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

PathLike = Union[str, Path]


def load_grayscale(path: PathLike) -> np.ndarray:
    """
    Load an image as a single-channel array suitable for template matching.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at {path}")
    return image

