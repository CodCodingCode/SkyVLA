"""Perception for object-goal navigation: VLM detection + 2D->3D lifting."""
from .lift import bbox_to_world
from .detect import OpenAIDetector, SyntheticDetector

__all__ = ["bbox_to_world", "OpenAIDetector", "SyntheticDetector"]
