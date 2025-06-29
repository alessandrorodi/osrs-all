"""
OSRS Bot Framework - Vision Module

This module contains computer vision components and detectors.
Phase 2 includes advanced AI vision capabilities with YOLO and OCR.
"""

from .intelligent_vision import IntelligentVision, UIState, VisualElement, intelligent_vision

# Import detectors if available
try:
    from .detectors import YOLODetector, OCRDetector
    detectors_available = True
except ImportError:
    detectors_available = False

__all__ = [
    'IntelligentVision',
    'UIState',
    'VisualElement', 
    'intelligent_vision'
]

if detectors_available:
    __all__.extend(['YOLODetector', 'OCRDetector']) 