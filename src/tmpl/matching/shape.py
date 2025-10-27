from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import math


@dataclass(slots=True)
class TemplateOrientation:
    """
    Stores a single orientation of a prepared template.
    """

    angle: float
    edges: np.ndarray
    width: int
    height: int


@dataclass(slots=True)
class CompiledTemplate:
    """
    Holds the precomputed edge masks for every orientation.
    """

    orientations: List[TemplateOrientation]

    @property
    def angles(self) -> List[float]:
        return [orientation.angle for orientation in self.orientations]


@dataclass(slots=True)
class ShapeMatchResult:
    """
    Result returned by the shape-based matcher.
    """

    top_left: Tuple[int, int]
    center: Tuple[float, float]
    score: float
    angle: float
    heatmap: np.ndarray
    template_size: Tuple[int, int]
    template_edges: np.ndarray


class ShapeTemplateMatcher:
    """
    Edge-driven template matcher resilient against illumination changes.
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        blur_kernel: int = 5,
        dilation_iterations: int = 1,
        method: int = cv2.TM_CCOEFF_NORMED,
        downscale_factor: float = 1.0,
        refine_angle_window: float = 15.0,
        refine_roi_scale: float = 2.0,
        refine_roi_padding: int = 12,
        coarse_min_score: float = 0.2,
    ) -> None:
        if canny_low <= 0 or canny_high <= 0:
            raise ValueError("Canny thresholds must be positive")
        if canny_low >= canny_high:
            raise ValueError("canny_low must be smaller than canny_high")
        if blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be odd")
        if dilation_iterations < 0:
            raise ValueError("dilation_iterations must be >= 0")
        if not (0 < downscale_factor <= 1.0):
            raise ValueError("downscale_factor must be in (0, 1]")
        if refine_angle_window < 0:
            raise ValueError("refine_angle_window must be >= 0")
        if refine_roi_scale < 1.0:
            raise ValueError("refine_roi_scale must be >= 1.0")
        if refine_roi_padding < 0:
            raise ValueError("refine_roi_padding must be >= 0")
        if coarse_min_score < 0 or coarse_min_score > 1:
            raise ValueError("coarse_min_score must be between 0 and 1")
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_kernel = blur_kernel
        self.dilation_iterations = dilation_iterations
        self.method = method
        self.downscale_factor = downscale_factor
        self.refine_angle_window = refine_angle_window
        self.refine_roi_scale = refine_roi_scale
        self.refine_roi_padding = refine_roi_padding
        self.coarse_min_score = coarse_min_score

    def compile_template(
        self,
        template: np.ndarray,
        angles: Sequence[float],
    ) -> CompiledTemplate:
        """
        Precompute edge masks for the template across requested rotations.
        """
        if template.ndim != 2:
            raise ValueError("template must be a single-channel grayscale image")
        if not angles:
            raise ValueError("angles must contain at least one rotation")

        base_edges = self._to_edge_mask(template)
        orientations: List[TemplateOrientation] = []
        for angle in angles:
            rotated = self._rotate_image(base_edges, angle)
            if not np.any(rotated):
                raise ValueError(f"empty edge mask after rotating template by {angle} degrees")
            orientations.append(
                TemplateOrientation(
                    angle=angle,
                    edges=rotated.astype(np.float32) / 255.0,
                    width=int(rotated.shape[1]),
                    height=int(rotated.shape[0]),
                )
            )
        return CompiledTemplate(orientations=orientations)

    def match(
        self,
        image: np.ndarray,
        compiled_template: CompiledTemplate,
    ) -> ShapeMatchResult:
        """
        Locate the best match for any orientation of the compiled template.
        """
        if image.ndim != 2:
            raise ValueError("image must be a single-channel grayscale image")

        edge_image = self._to_edge_mask(image).astype(np.float32) / 255.0

        search_bounds = (0, 0, edge_image.shape[1], edge_image.shape[0])
        candidate_indices = list(range(len(compiled_template.orientations)))

        coarse_result = None
        if self.downscale_factor < 1.0 and compiled_template.orientations:
            coarse_result = self._coarse_localize(edge_image, compiled_template)
            if coarse_result is not None:
                candidate_indices = coarse_result["indices"]
                search_bounds = coarse_result["bounds"]

        fine_result = self._fine_search(edge_image, compiled_template, candidate_indices, search_bounds)

        if fine_result is None:
            fine_result = self._fine_search(
                edge_image,
                compiled_template,
                list(range(len(compiled_template.orientations))),
                (0, 0, edge_image.shape[1], edge_image.shape[0]),
            )

        if fine_result is None:
            raise ValueError("template is larger than the search image for all orientations")

        best_location, best_score, best_angle, best_heatmap, best_edges, best_size = fine_result

        center = (
            best_location[0] + best_size[0] / 2.0,
            best_location[1] + best_size[1] / 2.0,
        )

        return ShapeMatchResult(
            top_left=best_location,
            center=center,
            score=best_score,
            angle=best_angle,
            heatmap=best_heatmap,
            template_size=best_size,
            template_edges=best_edges,
        )

    def _to_edge_mask(self, image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (self.blur_kernel, self.blur_kernel), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        if self.dilation_iterations > 0:
            kernel = np.ones((3, 3), dtype=np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=self.dilation_iterations)
        return edges

    @staticmethod
    def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        transform = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(transform[0, 0])
        sin = abs(transform[0, 1])

        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        transform[0, 2] += (new_width / 2.0) - center[0]
        transform[1, 2] += (new_height / 2.0) - center[1]

        rotated = cv2.warpAffine(
            image,
            transform,
            (new_width, new_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return rotated

    def _coarse_localize(
        self,
        edge_image: np.ndarray,
        compiled_template: CompiledTemplate,
    ) -> dict | None:
        factor = self.downscale_factor
        new_width = max(1, int(round(edge_image.shape[1] * factor)))
        new_height = max(1, int(round(edge_image.shape[0] * factor)))
        if new_width < 1 or new_height < 1:
            return None

        resized_image = cv2.resize(edge_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        best_score = -np.inf
        best_idx: int | None = None
        best_loc_small: Tuple[int, int] = (0, 0)
        scaled_templates: List[np.ndarray] = []
        scaled_dims: List[Tuple[int, int]] = []

        for orientation in compiled_template.orientations:
            width = max(1, int(round(orientation.width * factor)))
            height = max(1, int(round(orientation.height * factor)))
            scaled = cv2.resize(orientation.edges, (width, height), interpolation=cv2.INTER_AREA)
            scaled = (scaled > 0.05).astype(np.float32)
            scaled_templates.append(scaled)
            scaled_dims.append((width, height))

        for idx, template_small in enumerate(scaled_templates):
            if resized_image.shape[0] < template_small.shape[0] or resized_image.shape[1] < template_small.shape[1]:
                continue
            heatmap = cv2.matchTemplate(resized_image, template_small, self.method)
            _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)
            if max_val > best_score:
                best_score = float(max_val)
                best_idx = idx
                best_loc_small = (int(max_loc[0]), int(max_loc[1]))

        if best_idx is None or best_score < self.coarse_min_score:
            return None

        orientation = compiled_template.orientations[best_idx]
        scaled_width, scaled_height = scaled_dims[best_idx]
        inv_factor = 1.0 / factor
        center_x = (best_loc_small[0] + scaled_width / 2.0) * inv_factor
        center_y = (best_loc_small[1] + scaled_height / 2.0) * inv_factor

        half_width = (orientation.width * self.refine_roi_scale) / 2.0 + self.refine_roi_padding
        half_height = (orientation.height * self.refine_roi_scale) / 2.0 + self.refine_roi_padding

        x0 = max(0, int(math.floor(center_x - half_width)))
        x1 = min(edge_image.shape[1], int(math.ceil(center_x + half_width)))
        y0 = max(0, int(math.floor(center_y - half_height)))
        y1 = min(edge_image.shape[0], int(math.ceil(center_y + half_height)))

        min_width = orientation.width + 2
        min_height = orientation.height + 2
        if x1 - x0 < min_width:
            delta = min_width - (x1 - x0)
            x0 = max(0, x0 - delta // 2)
            x1 = min(edge_image.shape[1], x1 + delta - delta // 2)
        if y1 - y0 < min_height:
            delta = min_height - (y1 - y0)
            y0 = max(0, y0 - delta // 2)
            y1 = min(edge_image.shape[0], y1 + delta - delta // 2)

        candidate_indices = list(range(len(compiled_template.orientations)))
        if self.refine_angle_window > 0 and self.refine_angle_window < 180.0:
            best_angle = orientation.angle
            candidate_indices = [
                idx
                for idx, other in enumerate(compiled_template.orientations)
                if self._angle_distance(other.angle, best_angle) <= self.refine_angle_window
            ]
            if best_idx not in candidate_indices:
                candidate_indices.append(best_idx)

        candidate_indices = sorted(set(candidate_indices))

        return {"indices": candidate_indices, "bounds": (x0, y0, x1, y1)}

    def _fine_search(
        self,
        edge_image: np.ndarray,
        compiled_template: CompiledTemplate,
        indices: List[int],
        bounds: Tuple[int, int, int, int],
    ) -> Tuple[Tuple[int, int], float, float, np.ndarray, np.ndarray, Tuple[int, int]] | None:
        x0, y0, x1, y1 = bounds
        x0 = max(0, min(edge_image.shape[1], x0))
        x1 = max(0, min(edge_image.shape[1], x1))
        y0 = max(0, min(edge_image.shape[0], y0))
        y1 = max(0, min(edge_image.shape[0], y1))

        if x1 <= x0 or y1 <= y0:
            return None

        search_image = edge_image[y0:y1, x0:x1]

        best_score = -np.inf
        best_location: Tuple[int, int] | None = None
        best_angle = 0.0
        best_heatmap: np.ndarray | None = None
        best_edges: np.ndarray | None = None
        best_size: Tuple[int, int] = (0, 0)

        for idx in indices:
            orientation = compiled_template.orientations[idx]
            if search_image.shape[0] < orientation.height or search_image.shape[1] < orientation.width:
                continue
            heatmap = cv2.matchTemplate(search_image, orientation.edges, self.method)
            _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)

            if max_val > best_score:
                best_score = float(max_val)
                best_location = (int(max_loc[0] + x0), int(max_loc[1] + y0))
                best_angle = float(orientation.angle)
                best_heatmap = heatmap
                best_edges = orientation.edges
                best_size = (orientation.width, orientation.height)

        if best_heatmap is None or best_edges is None or best_location is None:
            return None

        return best_location, best_score, best_angle, best_heatmap, best_edges, best_size

    @staticmethod
    def _angle_distance(a: float, b: float) -> float:
        diff = (a - b + 180.0) % 360.0 - 180.0
        return abs(diff)
