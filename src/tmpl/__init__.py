"""
Core package for high-performance template matching utilities.
"""

from .matching.engine import TemplateMatcher
from .matching.shape import (
    CompiledTemplate,
    ShapeMatchResult,
    ShapeTemplateMatcher,
)

__all__ = ["TemplateMatcher", "ShapeTemplateMatcher", "ShapeMatchResult", "CompiledTemplate"]
