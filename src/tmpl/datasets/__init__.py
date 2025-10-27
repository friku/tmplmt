"""
Dataset helpers for benchmark suites bundled with the project.
"""

from .template_dataset import GroundTruthRecord, TemplateDataset, load_template_dataset

__all__ = ["GroundTruthRecord", "TemplateDataset", "load_template_dataset"]
