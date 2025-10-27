"""
Matching subpackage exposes high-level template matching APIs.
"""

from .engine import TemplateMatcher
from .shape import CompiledTemplate, ShapeMatchResult, ShapeTemplateMatcher

__all__ = [
    "CompiledTemplate",
    "ShapeMatchResult",
    "ShapeTemplateMatcher",
    "TemplateMatcher",
]
