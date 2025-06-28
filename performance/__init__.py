"""
Performance Monitoring and Optimization System

This package implements comprehensive performance monitoring and optimization
capabilities for the OSRS Bot Framework, including:

- System performance profiling
- YOLOv8 inference timing breakdown
- GPU utilization monitoring (RTX 4090 specific)
- Adaptive optimization algorithms
- Real-time performance analytics
"""

from .profiler import (
    SystemProfiler, 
    PerformanceProfiler, 
    GPUProfiler,
    YOLOProfiler,
    FrameProcessingProfiler
)

from .optimization_engine import (
    OptimizationEngine,
    AdaptiveOptimizer,
    RTX4090Optimizer,
    PerformanceOptimizer
)

__version__ = "1.0.0"
__all__ = [
    "SystemProfiler",
    "PerformanceProfiler", 
    "GPUProfiler",
    "YOLOProfiler",
    "FrameProcessingProfiler",
    "OptimizationEngine",
    "AdaptiveOptimizer", 
    "RTX4090Optimizer",
    "PerformanceOptimizer"
]