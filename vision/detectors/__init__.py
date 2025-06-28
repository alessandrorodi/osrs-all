"""
OSRS Bot Framework - Vision Detectors Module

This module contains advanced detection classes for Phase 2 implementation.
"""

from .yolo_detector import YOLODetector, GameStateDetection
from .ocr_detector import OCRDetector, TextDetection

__all__ = [
    'YOLODetector',
    'GameStateDetection', 
    'OCRDetector',
    'TextDetection'
] 