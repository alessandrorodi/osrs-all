"""
GUI tabs package

This package contains individual tab implementations for the OSRS Bot GUI.
"""

# Import all functions from the main tab functions module
from ..tab_functions import (
    create_dashboard_tab,
    create_bots_tab,
    create_vision_tab,
    create_performance_tab,
    create_logs_tab,
    create_templates_tab
)

# Import performance monitor tab
from .performance_monitor_tab import create_performance_monitor_tab, PerformanceMonitorTab

__all__ = [
    # Main tab functions
    'create_dashboard_tab',
    'create_bots_tab', 
    'create_vision_tab',
    'create_performance_tab',
    'create_logs_tab',
    'create_templates_tab',
    # Performance monitor tab
    'create_performance_monitor_tab',
    'PerformanceMonitorTab'
] 