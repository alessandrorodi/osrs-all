"""
GUI Widgets Package for OSRS Bot Framework

This package provides specialized GUI widgets for the OSRS bot interface including:
- Navigation panel with live minimap display
- Path visualization and planning tools
- Performance metrics and debug visualization
- Real-time analysis displays

Modules:
- navigation_panel: Comprehensive navigation control panel
"""

__version__ = "2.0.0"
__author__ = "OSRS Bot Framework"

# Import main widgets for easy access
try:
    from .navigation_panel import NavigationPanel
    __all__ = ['NavigationPanel']
except ImportError:
    # Handle case where dependencies aren't available
    __all__ = []