"""
Navigation Package for OSRS Bot Framework

This package provides advanced navigation capabilities including:
- Minimap analysis with YOLOv8 integration
- A* pathfinding algorithm
- Multi-floor navigation support
- Danger zone detection
- Real-time processing optimization

Modules:
- pathfinding: Core pathfinding algorithms and navigation logic
"""

__version__ = "2.0.0"
__author__ = "OSRS Bot Framework"

# Import main classes for easy access
try:
    from .pathfinding import OSRSPathfinder, NavigationGoal, PathResult, PathNode, MovementType
    __all__ = ['OSRSPathfinder', 'NavigationGoal', 'PathResult', 'PathNode', 'MovementType']
except ImportError:
    # Handle case where dependencies aren't available
    __all__ = []