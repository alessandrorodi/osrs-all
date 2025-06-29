"""
Adaptive Optimization Engine

Provides intelligent optimization capabilities including:
- Dynamic model switching (speed vs accuracy)
- Smart frame skipping algorithms  
- Resource allocation optimization
- Thermal throttling prevention
- RTX 4090 specific optimizations
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import numpy as np

from .profiler import PerformanceProfiler, YOLOMetrics, SystemMetrics, GPUMetrics

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Optimization levels"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM_PERFORMANCE = "maximum_performance"


class ModelProfile(Enum):
    """YOLO model profiles for different performance requirements"""
    NANO = "yolov8n.pt"  # Fastest, lowest accuracy
    SMALL = "yolov8s.pt"  # Fast, good accuracy
    MEDIUM = "yolov8m.pt"  # Balanced
    LARGE = "yolov8l.pt"  # Slower, better accuracy
    XLARGE = "yolov8x.pt"  # Slowest, best accuracy


@dataclass
class OptimizationStrategy:
    """Optimization strategy configuration"""
    target_fps: float = 60.0
    max_gpu_usage: float = 85.0
    max_gpu_temp: float = 80.0
    max_cpu_usage: float = 70.0
    max_memory_usage: float = 80.0
    quality_threshold: float = 0.7
    frame_skip_enabled: bool = True
    dynamic_resolution: bool = True
    batch_processing: bool = True
    tensorrt_enabled: bool = True


@dataclass
class OptimizationState:
    """Current optimization state"""
    timestamp: float
    current_model: ModelProfile
    current_resolution: Tuple[int, int]
    frame_skip_ratio: float
    batch_size: int
    gpu_power_limit: Optional[float]
    performance_score: float
    quality_score: float
    efficiency_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'current_model': self.current_model.value,
            'current_resolution': self.current_resolution,
            'frame_skip_ratio': self.frame_skip_ratio,
            'batch_size': self.batch_size,
            'gpu_power_limit': self.gpu_power_limit,
            'performance_score': self.performance_score,
            'quality_score': self.quality_score,
            'efficiency_score': self.efficiency_score
        }


class SmartFrameSkipper:
    """Intelligent frame skipping algorithm"""
    
    def __init__(self, target_fps: float = 60.0):
        self.target_fps = target_fps
        self.frame_times = []
        self.skip_ratio = 0.0
        self.last_decision_time = time.time()
        self.decision_interval = 1.0  # seconds
        
        # Frame importance scoring
        self.importance_weights = {
            'motion_detected': 1.5,
            'new_objects': 1.3,
            'ui_changes': 1.2,
            'combat_active': 2.0,
            'critical_event': 3.0
        }
    
    def should_skip_frame(self, frame_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Determine if current frame should be skipped"""
        current_time = time.time()
        
        # Update skip ratio periodically
        if current_time - self.last_decision_time > self.decision_interval:
            self._update_skip_ratio()
            self.last_decision_time = current_time
        
        # No skipping if we're not behind target FPS
        if self.skip_ratio <= 0.0:
            return False
        
        # Calculate frame importance
        importance_score = self._calculate_frame_importance(frame_metadata)
        
        # Higher importance = less likely to skip
        skip_threshold = self.skip_ratio * (2.0 - importance_score)
        
        return np.random.random() < skip_threshold
    
    def record_frame_time(self, processing_time: float) -> None:
        """Record frame processing time"""
        self.frame_times.append(processing_time)
        
        # Keep only recent measurements
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
    
    def _update_skip_ratio(self) -> None:
        """Update frame skip ratio based on performance"""
        if not self.frame_times:
            return
        
        avg_frame_time = np.mean(self.frame_times)
        current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Calculate how much we're behind target FPS
        fps_deficit = max(0, self.target_fps - current_fps)
        fps_ratio = fps_deficit / self.target_fps
        
        # Adjust skip ratio (more aggressive if further behind)
        if fps_ratio > 0.3:  # More than 30% behind
            self.skip_ratio = min(0.5, fps_ratio * 1.5)
        elif fps_ratio > 0.1:  # 10-30% behind
            self.skip_ratio = fps_ratio
        else:
            self.skip_ratio = 0.0
        
        logger.debug(f"Frame skipper: FPS {current_fps:.1f}, target {self.target_fps}, skip ratio {self.skip_ratio:.2f}")
    
    def _calculate_frame_importance(self, metadata: Optional[Dict[str, Any]]) -> float:
        """Calculate frame importance score (0.5 = low, 1.0 = normal, 2.0+ = high)"""
        if not metadata:
            return 1.0
        
        importance = 1.0
        
        for factor, weight in self.importance_weights.items():
            if metadata.get(factor, False):
                importance *= weight
        
        return min(3.0, importance)  # Cap at 3.0


class DynamicResolutionController:
    """Dynamic resolution adjustment for performance optimization"""
    
    def __init__(self, base_resolution: Tuple[int, int] = (640, 640)):
        self.base_resolution = base_resolution
        self.current_resolution = base_resolution
        self.resolution_levels = [
            (416, 416),   # Ultra-fast
            (512, 512),   # Fast
            (640, 640),   # Balanced
            (768, 768),   # Quality
            (896, 896),   # High quality
        ]
        self.performance_history = []
        self.adjustment_interval = 5.0  # seconds
        self.last_adjustment = time.time()
    
    def get_optimal_resolution(self, current_fps: float, target_fps: float, 
                              gpu_usage: float) -> Tuple[int, int]:
        """Get optimal resolution based on current performance"""
        current_time = time.time()
        
        # Only adjust periodically
        if current_time - self.last_adjustment < self.adjustment_interval:
            return self.current_resolution
        
        self.last_adjustment = current_time
        
        # Record current performance
        self.performance_history.append({
            'timestamp': current_time,
            'fps': current_fps,
            'gpu_usage': gpu_usage,
            'resolution': self.current_resolution
        })
        
        # Keep only recent history
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)
        
        # Determine if we need to adjust resolution
        fps_ratio = current_fps / target_fps if target_fps > 0 else 1.0
        
        current_level = self._get_resolution_level(self.current_resolution)
        
        if fps_ratio < 0.8 and gpu_usage > 80:  # Underperforming
            # Reduce resolution for better performance
            new_level = max(0, current_level - 1)
        elif fps_ratio > 1.2 and gpu_usage < 60:  # Overperforming
            # Increase resolution for better quality
            new_level = min(len(self.resolution_levels) - 1, current_level + 1)
        else:
            # Keep current resolution
            new_level = current_level
        
        new_resolution = self.resolution_levels[new_level]
        
        if new_resolution != self.current_resolution:
            logger.info(f"Resolution adjusted from {self.current_resolution} to {new_resolution}")
            self.current_resolution = new_resolution
        
        return self.current_resolution
    
    def _get_resolution_level(self, resolution: Tuple[int, int]) -> int:
        """Get resolution level index"""
        try:
            return self.resolution_levels.index(resolution)
        except ValueError:
            # Find closest resolution
            distances = [
                abs(res[0] - resolution[0]) + abs(res[1] - resolution[1])
                for res in self.resolution_levels
            ]
            return distances.index(min(distances))


class RTX4090Optimizer:
    """RTX 4090 specific optimizations"""
    
    def __init__(self):
        self.gpu_detected = False
        self.max_power_limit = 450  # RTX 4090 max power (watts)
        self.thermal_throttle_temp = 83  # RTX 4090 thermal throttle temp
        self.memory_bandwidth = 1008  # GB/s
        self.cuda_cores = 16384
        
        # TensorRT optimization
        self.tensorrt_enabled = False
        self.tensorrt_precision = "fp16"  # fp32, fp16, int8
        
        # Multi-stream processing
        self.cuda_streams = 4
        self.async_processing = True
        
        self._detect_rtx4090()
    
    def _detect_rtx4090(self) -> None:
        """Detect RTX 4090 GPU"""
        try:
            # This would use GPU detection logic from profiler
            # For now, assume it's available
            self.gpu_detected = True
            logger.info("RTX 4090 optimizer initialized")
        except Exception as e:
            logger.warning(f"RTX 4090 detection failed: {e}")
    
    def optimize_for_yolo(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply RTX 4090 specific YOLO optimizations"""
        if not self.gpu_detected:
            return model_config
        
        optimized_config = model_config.copy()
        
        # Enable TensorRT if available
        if self.tensorrt_enabled:
            optimized_config.update({
                'engine': 'tensorrt',
                'precision': self.tensorrt_precision,
                'workspace_size': 4 * 1024**3,  # 4GB workspace
                'max_batch_size': 8,
                'optimize_for_throughput': True
            })
        
        # CUDA optimizations
        optimized_config.update({
            'device': 'cuda:0',
            'cuda_streams': self.cuda_streams,
            'async_processing': self.async_processing,
            'memory_growth': True,
            'allow_growth': True
        })
        
        # Memory optimizations
        optimized_config.update({
            'pin_memory': True,
            'non_blocking': True,
            'prefetch_factor': 2,
            'persistent_workers': True
        })
        
        return optimized_config
    
    def adjust_power_limit(self, target_temp: float, current_temp: float, 
                          current_power: float) -> Optional[float]:
        """Adjust GPU power limit based on temperature"""
        if not self.gpu_detected:
            return None
        
        # Prevent thermal throttling
        if current_temp > target_temp:
            # Reduce power limit
            reduction_factor = min(0.2, (current_temp - target_temp) / 10.0)
            new_power_limit = current_power * (1.0 - reduction_factor)
            return max(200, new_power_limit)  # Minimum 200W
        
        elif current_temp < target_temp - 5:
            # Can increase power limit
            increase_factor = min(0.1, (target_temp - current_temp) / 20.0)
            new_power_limit = current_power * (1.0 + increase_factor)
            return min(self.max_power_limit, new_power_limit)
        
        return None  # No change needed
    
    def get_optimal_batch_size(self, model_size: str, input_resolution: Tuple[int, int]) -> int:
        """Get optimal batch size for RTX 4090"""
        if not self.gpu_detected:
            return 1
        
        # Estimate memory usage per sample
        width, height = input_resolution
        channels = 3
        sample_memory_mb = (width * height * channels * 4) / (1024**2)  # 4 bytes per float32
        
        # Model memory estimates (rough)
        model_memory = {
            'nano': 6,    # MB
            'small': 22,  # MB
            'medium': 52, # MB
            'large': 88,  # MB
            'xlarge': 136 # MB
        }
        
        model_mb = model_memory.get(model_size.lower(), 50)
        
        # RTX 4090 has 24GB VRAM, leave some headroom
        available_memory_mb = 20 * 1024  # 20GB usable
        memory_per_batch = model_mb + (sample_memory_mb * 2)  # 2x for processing overhead
        
        max_batch_size = int(available_memory_mb / memory_per_batch)
        
        # Practical limits
        return min(max_batch_size, 16)  # Cap at 16 for stability


class AdaptiveOptimizer:
    """Main adaptive optimization controller"""
    
    def __init__(self, profiler: PerformanceProfiler, strategy: OptimizationStrategy):
        self.profiler = profiler
        self.strategy = strategy
        
        # Optimization components
        self.frame_skipper = SmartFrameSkipper(strategy.target_fps)
        self.resolution_controller = DynamicResolutionController()
        self.rtx4090_optimizer = RTX4090Optimizer()
        
        # State tracking
        self.current_state = OptimizationState(
            timestamp=time.time(),
            current_model=ModelProfile.MEDIUM,
            current_resolution=(640, 640),
            frame_skip_ratio=0.0,
            batch_size=1,
            gpu_power_limit=None,
            performance_score=0.0,
            quality_score=0.0,
            efficiency_score=0.0
        )
        
        # Decision making
        self.optimization_interval = 2.0  # seconds
        self.last_optimization = time.time()
        self.optimization_history = []
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'model_changed': [],
            'resolution_changed': [],
            'optimization_applied': []
        }
        
        logger.info("Adaptive optimizer initialized")
    
    def optimize(self, force: bool = False) -> Dict[str, Any]:
        """Perform optimization step"""
        current_time = time.time()
        
        # Only optimize periodically unless forced
        if not force and current_time - self.last_optimization < self.optimization_interval:
            return {}
        
        self.last_optimization = current_time
        
        # Get current performance snapshot
        snapshot = self.profiler.get_complete_snapshot()
        optimizations_applied = {}
        
        try:
            # Analyze current performance
            performance_analysis = self._analyze_performance(snapshot)
            
            # Model optimization
            if self.strategy.dynamic_resolution:
                model_optimization = self._optimize_model_selection(performance_analysis)
                optimizations_applied.update(model_optimization)
            
            # Resolution optimization
            if self.strategy.dynamic_resolution:
                resolution_optimization = self._optimize_resolution(performance_analysis)
                optimizations_applied.update(resolution_optimization)
            
            # Frame skipping optimization
            if self.strategy.frame_skip_enabled:
                frame_skip_optimization = self._optimize_frame_skipping(performance_analysis)
                optimizations_applied.update(frame_skip_optimization)
            
            # GPU power optimization
            gpu_optimization = self._optimize_gpu_power(performance_analysis)
            optimizations_applied.update(gpu_optimization)
            
            # Update current state
            self._update_optimization_state(performance_analysis, optimizations_applied)
            
            # Record optimization
            self.optimization_history.append({
                'timestamp': current_time,
                'performance_analysis': performance_analysis,
                'optimizations_applied': optimizations_applied,
                'state': self.current_state.to_dict()
            })
            
            # Trigger callbacks
            self._trigger_callbacks('optimization_applied', optimizations_applied)
            
            logger.debug(f"Optimization applied: {optimizations_applied}")
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
        
        return optimizations_applied
    
    def _analyze_performance(self, snapshot) -> Dict[str, Any]:
        """Analyze current performance metrics"""
        analysis = {
            'timestamp': snapshot.timestamp,
            'fps': 0.0,
            'gpu_usage': 0.0,
            'gpu_temperature': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'bottlenecks': [],
            'performance_score': 0.0
        }
        
        # System metrics
        if snapshot.system:
            analysis['cpu_usage'] = snapshot.system.cpu_usage
            analysis['memory_usage'] = snapshot.system.memory_usage
        
        # GPU metrics
        if snapshot.gpu and len(snapshot.gpu) > 0:
            gpu = snapshot.gpu[0]  # Primary GPU
            analysis['gpu_usage'] = gpu.gpu_usage
            analysis['gpu_temperature'] = gpu.temperature
        
        # YOLO performance
        if snapshot.yolo and len(snapshot.yolo) > 0:
            recent_yolo = snapshot.yolo[-10:]  # Last 10 inferences
            fps_values = [m.fps for m in recent_yolo]
            analysis['fps'] = np.mean(fps_values) if fps_values else 0.0
        
        # Frame processing performance
        if snapshot.frame_processing and len(snapshot.frame_processing) > 0:
            recent_frames = snapshot.frame_processing[-10:]  # Last 10 frames
            processing_times = [f.total_time for f in recent_frames]
            if processing_times:
                avg_time = np.mean(processing_times)
                analysis['pipeline_fps'] = 1000.0 / avg_time if avg_time > 0 else 0.0
        
        # Identify bottlenecks
        if analysis['gpu_usage'] > 90:
            analysis['bottlenecks'].append('gpu_compute')
        if analysis['gpu_temperature'] > self.strategy.max_gpu_temp:
            analysis['bottlenecks'].append('gpu_thermal')
        if analysis['cpu_usage'] > self.strategy.max_cpu_usage:
            analysis['bottlenecks'].append('cpu')
        if analysis['memory_usage'] > self.strategy.max_memory_usage:
            analysis['bottlenecks'].append('memory')
        
        # Calculate performance score
        fps_score = min(1.0, analysis['fps'] / self.strategy.target_fps)
        efficiency_score = 1.0 - (analysis['gpu_usage'] / 100.0)
        analysis['performance_score'] = (fps_score + efficiency_score) / 2.0
        
        return analysis
    
    def _optimize_model_selection(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize YOLO model selection"""
        optimizations = {}
        current_fps = analysis['fps']
        target_fps = self.strategy.target_fps
        
        # Determine if model change is needed
        if current_fps < target_fps * 0.8:  # Significantly under target
            # Consider faster model
            current_idx = list(ModelProfile).index(self.current_state.current_model)
            if current_idx > 0:  # Can go faster
                new_model = list(ModelProfile)[current_idx - 1]
                optimizations['model_change'] = {
                    'from': self.current_state.current_model.value,
                    'to': new_model.value,
                    'reason': 'performance'
                }
                self.current_state.current_model = new_model
                self._trigger_callbacks('model_changed', new_model)
        
        elif current_fps > target_fps * 1.3 and analysis['gpu_usage'] < 60:  # Over target with headroom
            # Consider more accurate model
            current_idx = list(ModelProfile).index(self.current_state.current_model)
            if current_idx < len(ModelProfile) - 1:  # Can go more accurate
                new_model = list(ModelProfile)[current_idx + 1]
                optimizations['model_change'] = {
                    'from': self.current_state.current_model.value,
                    'to': new_model.value,
                    'reason': 'quality'
                }
                self.current_state.current_model = new_model
                self._trigger_callbacks('model_changed', new_model)
        
        return optimizations
    
    def _optimize_resolution(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize input resolution"""
        optimizations = {}
        
        current_fps = analysis['fps']
        target_fps = self.strategy.target_fps
        gpu_usage = analysis['gpu_usage']
        
        new_resolution = self.resolution_controller.get_optimal_resolution(
            current_fps, target_fps, gpu_usage
        )
        
        if new_resolution != self.current_state.current_resolution:
            optimizations['resolution_change'] = {
                'from': self.current_state.current_resolution,
                'to': new_resolution,
                'reason': 'performance'
            }
            self.current_state.current_resolution = new_resolution
            self._trigger_callbacks('resolution_changed', new_resolution)
        
        return optimizations
    
    def _optimize_frame_skipping(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize frame skipping"""
        optimizations = {}
        
        # Update frame skipper with current performance
        if analysis['fps'] > 0:
            frame_time = 1.0 / analysis['fps']
            self.frame_skipper.record_frame_time(frame_time)
        
        # Get current skip ratio
        old_skip_ratio = self.current_state.frame_skip_ratio
        self.frame_skipper._update_skip_ratio()
        new_skip_ratio = self.frame_skipper.skip_ratio
        
        if abs(new_skip_ratio - old_skip_ratio) > 0.05:  # Significant change
            optimizations['frame_skip_change'] = {
                'from': old_skip_ratio,
                'to': new_skip_ratio,
                'reason': 'fps_optimization'
            }
            self.current_state.frame_skip_ratio = new_skip_ratio
        
        return optimizations
    
    def _optimize_gpu_power(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize GPU power limit"""
        optimizations = {}
        
        if 'gpu_thermal' in analysis['bottlenecks']:
            current_temp = analysis['gpu_temperature']
            target_temp = self.strategy.max_gpu_temp
            current_power = 400  # Would get from GPU monitoring
            
            new_power_limit = self.rtx4090_optimizer.adjust_power_limit(
                target_temp, current_temp, current_power
            )
            
            if new_power_limit and new_power_limit != self.current_state.gpu_power_limit:
                optimizations['power_limit_change'] = {
                    'from': self.current_state.gpu_power_limit,
                    'to': new_power_limit,
                    'reason': 'thermal_management'
                }
                self.current_state.gpu_power_limit = new_power_limit
        
        return optimizations
    
    def _update_optimization_state(self, analysis: Dict[str, Any], 
                                  optimizations: Dict[str, Any]) -> None:
        """Update optimization state"""
        self.current_state.timestamp = time.time()
        self.current_state.performance_score = analysis['performance_score']
        
        # Calculate quality score (placeholder - would be based on detection accuracy)
        self.current_state.quality_score = 0.8  # Would calculate from actual metrics
        
        # Calculate efficiency score
        fps_efficiency = min(1.0, analysis['fps'] / self.strategy.target_fps)
        resource_efficiency = 1.0 - (analysis['gpu_usage'] / 100.0)
        self.current_state.efficiency_score = (fps_efficiency + resource_efficiency) / 2.0
    
    def _trigger_callbacks(self, event: str, data: Any) -> None:
        """Trigger event callbacks"""
        for callback in self.callbacks[event]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    def add_callback(self, event: str, callback: Callable) -> None:
        """Add optimization callback"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations"""
        recommendations = []
        
        if not self.optimization_history:
            return recommendations
        
        recent_state = self.current_state
        
        # Performance recommendations
        if recent_state.performance_score < 0.7:
            recommendations.append("Consider reducing model complexity or resolution")
        
        if recent_state.frame_skip_ratio > 0.3:
            recommendations.append("High frame skip ratio - system may be overloaded")
        
        if recent_state.efficiency_score < 0.6:
            recommendations.append("Low efficiency - optimize resource usage")
        
        # RTX 4090 specific recommendations
        if self.rtx4090_optimizer.gpu_detected:
            if not self.rtx4090_optimizer.tensorrt_enabled:
                recommendations.append("Enable TensorRT for RTX 4090 optimization")
            
            recommendations.append("Consider multi-stream processing for RTX 4090")
        
        return recommendations


class OptimizationEngine:
    """Main optimization engine coordinating all optimization systems"""
    
    def __init__(self, profiler: PerformanceProfiler, 
                 optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.profiler = profiler
        self.optimization_level = optimization_level
        
        # Create optimization strategy based on level
        self.strategy = self._create_strategy(optimization_level)
        
        # Initialize optimizer
        self.adaptive_optimizer = AdaptiveOptimizer(profiler, self.strategy)
        
        # State management
        self._running = False
        self._optimization_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.optimization_stats = {
            'optimizations_applied': 0,
            'performance_improvements': 0,
            'average_fps_gain': 0.0,
            'total_runtime': 0.0
        }
        
        logger.info(f"Optimization engine initialized with level: {optimization_level.value}")
    
    def _create_strategy(self, level: OptimizationLevel) -> OptimizationStrategy:
        """Create optimization strategy based on level"""
        strategies = {
            OptimizationLevel.CONSERVATIVE: OptimizationStrategy(
                target_fps=30.0,
                max_gpu_usage=70.0,
                max_gpu_temp=75.0,
                frame_skip_enabled=False,
                dynamic_resolution=False,
                tensorrt_enabled=False
            ),
            OptimizationLevel.BALANCED: OptimizationStrategy(
                target_fps=60.0,
                max_gpu_usage=85.0,
                max_gpu_temp=80.0,
                frame_skip_enabled=True,
                dynamic_resolution=True,
                tensorrt_enabled=True
            ),
            OptimizationLevel.AGGRESSIVE: OptimizationStrategy(
                target_fps=90.0,
                max_gpu_usage=95.0,
                max_gpu_temp=85.0,
                frame_skip_enabled=True,
                dynamic_resolution=True,
                tensorrt_enabled=True
            ),
            OptimizationLevel.MAXIMUM_PERFORMANCE: OptimizationStrategy(
                target_fps=120.0,
                max_gpu_usage=98.0,
                max_gpu_temp=88.0,
                frame_skip_enabled=True,
                dynamic_resolution=True,
                batch_processing=True,
                tensorrt_enabled=True
            )
        }
        
        return strategies[level]
    
    def start(self) -> None:
        """Start optimization engine"""
        if self._running:
            return
        
        self._running = True
        self._optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self._optimization_thread.start()
        
        logger.info("Optimization engine started")
    
    def stop(self) -> None:
        """Stop optimization engine"""
        self._running = False
        
        if self._optimization_thread and self._optimization_thread.is_alive():
            self._optimization_thread.join(timeout=2.0)
        
        logger.info("Optimization engine stopped")
    
    def _optimization_loop(self) -> None:
        """Main optimization loop"""
        start_time = time.time()
        
        while self._running:
            try:
                # Perform optimization
                optimizations = self.adaptive_optimizer.optimize()
                
                if optimizations:
                    self.optimization_stats['optimizations_applied'] += 1
                    logger.debug(f"Applied optimizations: {optimizations}")
                
                # Sleep before next optimization cycle
                time.sleep(self.adaptive_optimizer.optimization_interval)
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                time.sleep(1.0)
        
        self.optimization_stats['total_runtime'] = time.time() - start_time
    
    def force_optimization(self) -> Dict[str, Any]:
        """Force immediate optimization"""
        return self.adaptive_optimizer.optimize(force=True)
    
    def get_current_state(self) -> OptimizationState:
        """Get current optimization state"""
        return self.adaptive_optimizer.current_state
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return self.optimization_stats.copy()
    
    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations"""
        return self.adaptive_optimizer.get_optimization_recommendations()
    
    def update_strategy(self, new_strategy: OptimizationStrategy) -> None:
        """Update optimization strategy"""
        self.strategy = new_strategy
        self.adaptive_optimizer.strategy = new_strategy
        logger.info("Optimization strategy updated")
    
    def export_optimization_data(self, filepath: Optional[str] = None) -> str:
        """Export optimization data"""
        if not filepath:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"optimization_data_{timestamp}.json"
        
        export_data = {
            'timestamp': time.time(),
            'optimization_level': self.optimization_level.value,
            'strategy': {
                'target_fps': self.strategy.target_fps,
                'max_gpu_usage': self.strategy.max_gpu_usage,
                'max_gpu_temp': self.strategy.max_gpu_temp,
                'frame_skip_enabled': self.strategy.frame_skip_enabled,
                'dynamic_resolution': self.strategy.dynamic_resolution,
                'tensorrt_enabled': self.strategy.tensorrt_enabled
            },
            'current_state': self.adaptive_optimizer.current_state.to_dict(),
            'optimization_history': self.adaptive_optimizer.optimization_history[-100:],  # Last 100
            'statistics': self.optimization_stats,
            'recommendations': self.get_recommendations()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Optimization data exported to {filepath}")
        return filepath


class PerformanceOptimizer:
    """High-level performance optimizer interface"""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.optimization_engine = OptimizationEngine(profiler)
        
        # Quick optimization presets
        self.presets = {
            'battery_saver': OptimizationLevel.CONSERVATIVE,
            'balanced': OptimizationLevel.BALANCED,
            'performance': OptimizationLevel.AGGRESSIVE,
            'maximum': OptimizationLevel.MAXIMUM_PERFORMANCE
        }
    
    def apply_preset(self, preset_name: str) -> bool:
        """Apply optimization preset"""
        if preset_name not in self.presets:
            logger.error(f"Unknown preset: {preset_name}")
            return False
        
        level = self.presets[preset_name]
        strategy = self.optimization_engine._create_strategy(level)
        self.optimization_engine.update_strategy(strategy)
        
        logger.info(f"Applied optimization preset: {preset_name}")
        return True
    
    def auto_optimize(self) -> Dict[str, Any]:
        """Perform automatic optimization"""
        return self.optimization_engine.force_optimization()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        snapshot = self.profiler.get_complete_snapshot()
        state = self.optimization_engine.get_current_state()
        stats = self.optimization_engine.get_optimization_stats()
        
        return {
            'timestamp': time.time(),
            'system_performance': {
                'cpu_usage': snapshot.system.cpu_usage if snapshot.system else 0,
                'memory_usage': snapshot.system.memory_usage if snapshot.system else 0,
                'gpu_usage': snapshot.gpu[0].gpu_usage if snapshot.gpu else 0,
                'gpu_temperature': snapshot.gpu[0].temperature if snapshot.gpu else 0,
            },
            'optimization_state': state.to_dict(),
            'optimization_stats': stats,
            'recommendations': self.optimization_engine.get_recommendations()
        }


# Global optimization engine instance
def create_optimization_engine(profiler: PerformanceProfiler, 
                              level: OptimizationLevel = OptimizationLevel.BALANCED) -> OptimizationEngine:
    """Create and configure optimization engine"""
    return OptimizationEngine(profiler, level)