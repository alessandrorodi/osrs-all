"""
Performance Profiling System

Comprehensive profiling capabilities for monitoring system performance,
YOLOv8 inference timing, GPU utilization, and frame processing pipeline.
"""

import time
import threading
import queue
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from pathlib import Path
import json
import psutil
import numpy as np
from enum import Enum

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("pynvml not available - GPU monitoring disabled")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.warning("GPUtil not available - alternative GPU monitoring disabled")

logger = logging.getLogger(__name__)


class ProfilerState(Enum):
    """Profiler states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_usage: float  # Percentage
    memory_usage: float  # Percentage
    memory_used_gb: float
    memory_total_gb: float
    disk_io_read: float  # MB/s
    disk_io_write: float  # MB/s
    network_io_sent: float  # MB/s
    network_io_recv: float  # MB/s
    process_count: int
    thread_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GPUMetrics:
    """GPU performance metrics"""
    timestamp: float
    gpu_id: int
    gpu_name: str
    gpu_usage: float  # Percentage
    memory_usage: float  # Percentage
    memory_used_mb: float
    memory_total_mb: float
    temperature: float  # Celsius
    power_usage: float  # Watts
    clock_core: float  # MHz
    clock_memory: float  # MHz
    fan_speed: float  # Percentage
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class YOLOMetrics:
    """YOLO inference performance metrics"""
    timestamp: float
    model_name: str
    image_size: Tuple[int, int]
    preprocessing_time: float  # milliseconds
    inference_time: float  # milliseconds
    postprocessing_time: float  # milliseconds
    total_time: float  # milliseconds
    detections_count: int
    batch_size: int
    fps: float
    gpu_memory_used: Optional[float] = None  # MB
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrameProcessingMetrics:
    """Frame processing pipeline metrics"""
    timestamp: float
    frame_id: int
    capture_time: float  # milliseconds
    vision_time: float  # milliseconds
    yolo_time: float  # milliseconds
    ocr_time: float  # milliseconds
    minimap_time: float  # milliseconds
    classification_time: float  # milliseconds
    total_time: float  # milliseconds
    queue_size: int
    frame_drops: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceSnapshot:
    """Complete performance snapshot"""
    timestamp: float
    system: SystemMetrics
    gpu: Optional[List[GPUMetrics]] = None
    yolo: Optional[List[YOLOMetrics]] = None
    frame_processing: Optional[List[FrameProcessingMetrics]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SystemProfiler:
    """System-level performance profiler"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.metrics_history: deque = deque(maxlen=3600)  # 1 hour of data
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Initialize psutil counters
        self._last_disk_io = psutil.disk_io_counters()
        self._last_network_io = psutil.net_io_counters()
        self._last_time = time.time()
    
    def start(self) -> None:
        """Start system profiling"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._profile_loop, daemon=True)
        self._thread.start()
        logger.info("System profiler started")
    
    def stop(self) -> None:
        """Stop system profiling"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("System profiler stopped")
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        current_time = time.time()
        
        # CPU and Memory
        cpu_usage = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        time_delta = current_time - self._last_time
        
        if time_delta > 0 and self._last_disk_io and disk_io:
            disk_read = (disk_io.read_bytes - self._last_disk_io.read_bytes) / (1024**2) / time_delta
            disk_write = (disk_io.write_bytes - self._last_disk_io.write_bytes) / (1024**2) / time_delta
        else:
            disk_read = disk_write = 0.0
        
        # Network I/O
        network_io = psutil.net_io_counters()
        if time_delta > 0 and self._last_network_io:
            network_sent = (network_io.bytes_sent - self._last_network_io.bytes_sent) / (1024**2) / time_delta
            network_recv = (network_io.bytes_recv - self._last_network_io.bytes_recv) / (1024**2) / time_delta
        else:
            network_sent = network_recv = 0.0
        
        # Process information
        process_count = len(psutil.pids())
        thread_count = threading.active_count()
        
        # Update last values
        self._last_disk_io = disk_io
        self._last_network_io = network_io
        self._last_time = current_time
        
        return SystemMetrics(
            timestamp=current_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            disk_io_read=disk_read,
            disk_io_write=disk_write,
            network_io_sent=network_sent,
            network_io_recv=network_recv,
            process_count=process_count,
            thread_count=thread_count
        )
    
    def get_metrics_history(self, duration_seconds: int = 300) -> List[SystemMetrics]:
        """Get recent metrics history"""
        cutoff_time = time.time() - duration_seconds
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]
    
    def _profile_loop(self) -> None:
        """Main profiling loop"""
        while self._running:
            try:
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"System profiling error: {e}")
                time.sleep(self.update_interval)


class GPUProfiler:
    """GPU performance profiler with RTX 4090 optimizations"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.metrics_history: deque = deque(maxlen=3600)  # 1 hour of data
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Initialize GPU monitoring
        self._gpu_available = self._initialize_gpu_monitoring()
        self._rtx4090_detected = self._detect_rtx4090()
        
        if self._rtx4090_detected:
            logger.info("RTX 4090 detected - enabling advanced GPU profiling")
    
    def _initialize_gpu_monitoring(self) -> bool:
        """Initialize GPU monitoring libraries"""
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                return True
            except Exception as e:
                logger.warning(f"NVML initialization failed: {e}")
        
        return GPUTIL_AVAILABLE
    
    def _detect_rtx4090(self) -> bool:
        """Detect if RTX 4090 is present"""
        try:
            if NVML_AVAILABLE:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    if "RTX 4090" in name:
                        return True
            
            elif GPUTIL_AVAILABLE:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    if "RTX 4090" in gpu.name:
                        return True
        
        except Exception as e:
            logger.error(f"GPU detection error: {e}")
        
        return False
    
    def start(self) -> None:
        """Start GPU profiling"""
        if self._running or not self._gpu_available:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._profile_loop, daemon=True)
        self._thread.start()
        logger.info("GPU profiler started")
    
    def stop(self) -> None:
        """Stop GPU profiling"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("GPU profiler stopped")
    
    def get_current_metrics(self) -> List[GPUMetrics]:
        """Get current GPU metrics for all GPUs"""
        if not self._gpu_available:
            return []
        
        metrics = []
        current_time = time.time()
        
        try:
            if NVML_AVAILABLE:
                device_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Basic info
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    
                    # Utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage = util.gpu
                    memory_usage = util.memory
                    
                    # Memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_used_mb = mem_info.used / (1024**2)
                    memory_total_mb = mem_info.total / (1024**2)
                    
                    # Temperature
                    try:
                        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        temperature = 0.0
                    
                    # Power
                    try:
                        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    except:
                        power_usage = 0.0
                    
                    # Clock speeds
                    try:
                        clock_core = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                        clock_memory = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                    except:
                        clock_core = clock_memory = 0.0
                    
                    # Fan speed
                    try:
                        fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                    except:
                        fan_speed = 0.0
                    
                    metrics.append(GPUMetrics(
                        timestamp=current_time,
                        gpu_id=i,
                        gpu_name=name,
                        gpu_usage=gpu_usage,
                        memory_usage=memory_usage,
                        memory_used_mb=memory_used_mb,
                        memory_total_mb=memory_total_mb,
                        temperature=temperature,
                        power_usage=power_usage,
                        clock_core=clock_core,
                        clock_memory=clock_memory,
                        fan_speed=fan_speed
                    ))
            
            elif GPUTIL_AVAILABLE:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    metrics.append(GPUMetrics(
                        timestamp=current_time,
                        gpu_id=i,
                        gpu_name=gpu.name,
                        gpu_usage=gpu.load * 100,
                        memory_usage=gpu.memoryUtil * 100,
                        memory_used_mb=gpu.memoryUsed,
                        memory_total_mb=gpu.memoryTotal,
                        temperature=gpu.temperature,
                        power_usage=0.0,  # Not available in GPUtil
                        clock_core=0.0,
                        clock_memory=0.0,
                        fan_speed=0.0
                    ))
        
        except Exception as e:
            logger.error(f"GPU metrics collection error: {e}")
        
        return metrics
    
    def get_metrics_history(self, duration_seconds: int = 300) -> List[List[GPUMetrics]]:
        """Get recent GPU metrics history"""
        cutoff_time = time.time() - duration_seconds
        return [m for m in self.metrics_history if m[0].timestamp > cutoff_time]
    
    def _profile_loop(self) -> None:
        """Main GPU profiling loop"""
        while self._running:
            try:
                metrics = self.get_current_metrics()
                if metrics:
                    self.metrics_history.append(metrics)
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"GPU profiling error: {e}")
                time.sleep(self.update_interval)


class YOLOProfiler:
    """YOLO inference performance profiler"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self._active_inferences: Dict[str, Dict[str, Any]] = {}
        
    def start_inference(self, model_name: str, image_size: Tuple[int, int], 
                       batch_size: int = 1) -> str:
        """Start timing a YOLO inference"""
        inference_id = f"{model_name}_{time.time()}_{id(threading.current_thread())}"
        
        self._active_inferences[inference_id] = {
            'start_time': time.time(),
            'model_name': model_name,
            'image_size': image_size,
            'batch_size': batch_size,
            'preprocessing_time': 0.0,
            'inference_time': 0.0,
            'postprocessing_time': 0.0
        }
        
        return inference_id
    
    def record_preprocessing(self, inference_id: str, duration: float) -> None:
        """Record preprocessing time"""
        if inference_id in self._active_inferences:
            self._active_inferences[inference_id]['preprocessing_time'] = duration * 1000  # Convert to ms
    
    def record_inference(self, inference_id: str, duration: float, 
                        gpu_memory_used: Optional[float] = None) -> None:
        """Record inference time"""
        if inference_id in self._active_inferences:
            self._active_inferences[inference_id]['inference_time'] = duration * 1000  # Convert to ms
            if gpu_memory_used:
                self._active_inferences[inference_id]['gpu_memory_used'] = gpu_memory_used
    
    def record_postprocessing(self, inference_id: str, duration: float) -> None:
        """Record postprocessing time"""
        if inference_id in self._active_inferences:
            self._active_inferences[inference_id]['postprocessing_time'] = duration * 1000  # Convert to ms
    
    def finish_inference(self, inference_id: str, detections_count: int) -> YOLOMetrics:
        """Finish timing and record complete metrics"""
        if inference_id not in self._active_inferences:
            raise ValueError(f"Inference ID {inference_id} not found")
        
        inference_data = self._active_inferences.pop(inference_id)
        
        total_time = (
            inference_data['preprocessing_time'] +
            inference_data['inference_time'] +
            inference_data['postprocessing_time']
        )
        
        fps = 1000.0 / total_time if total_time > 0 else 0.0
        
        metrics = YOLOMetrics(
            timestamp=time.time(),
            model_name=inference_data['model_name'],
            image_size=inference_data['image_size'],
            preprocessing_time=inference_data['preprocessing_time'],
            inference_time=inference_data['inference_time'],
            postprocessing_time=inference_data['postprocessing_time'],
            total_time=total_time,
            detections_count=detections_count,
            batch_size=inference_data['batch_size'],
            fps=fps,
            gpu_memory_used=inference_data.get('gpu_memory_used')
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_metrics_history(self, duration_seconds: int = 300) -> List[YOLOMetrics]:
        """Get recent YOLO metrics history"""
        cutoff_time = time.time() - duration_seconds
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.get_metrics_history(300)  # Last 5 minutes
        
        inference_times = [m.inference_time for m in recent_metrics]
        total_times = [m.total_time for m in recent_metrics]
        fps_values = [m.fps for m in recent_metrics]
        
        return {
            'total_inferences': len(self.metrics_history),
            'recent_inferences': len(recent_metrics),
            'avg_inference_time': np.mean(inference_times) if inference_times else 0,
            'min_inference_time': np.min(inference_times) if inference_times else 0,
            'max_inference_time': np.max(inference_times) if inference_times else 0,
            'avg_total_time': np.mean(total_times) if total_times else 0,
            'avg_fps': np.mean(fps_values) if fps_values else 0,
            'max_fps': np.max(fps_values) if fps_values else 0,
            'bottleneck_analysis': self._analyze_bottlenecks(recent_metrics)
        }
    
    def _analyze_bottlenecks(self, metrics: List[YOLOMetrics]) -> Dict[str, Any]:
        """Analyze performance bottlenecks"""
        if not metrics:
            return {}
        
        preprocessing_times = [m.preprocessing_time for m in metrics]
        inference_times = [m.inference_time for m in metrics]
        postprocessing_times = [m.postprocessing_time for m in metrics]
        
        avg_preprocessing = np.mean(preprocessing_times)
        avg_inference = np.mean(inference_times)
        avg_postprocessing = np.mean(postprocessing_times)
        
        total_avg = avg_preprocessing + avg_inference + avg_postprocessing
        
        return {
            'preprocessing_percentage': (avg_preprocessing / total_avg * 100) if total_avg > 0 else 0,
            'inference_percentage': (avg_inference / total_avg * 100) if total_avg > 0 else 0,
            'postprocessing_percentage': (avg_postprocessing / total_avg * 100) if total_avg > 0 else 0,
            'primary_bottleneck': max([
                ('preprocessing', avg_preprocessing),
                ('inference', avg_inference),
                ('postprocessing', avg_postprocessing)
            ], key=lambda x: x[1])[0]
        }


class FrameProcessingProfiler:
    """Frame processing pipeline profiler"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self._active_frames: Dict[int, Dict[str, Any]] = {}
        self._frame_counter = 0
        self._drop_counter = 0
        
    def start_frame_processing(self, frame_size: Tuple[int, int]) -> int:
        """Start timing frame processing"""
        frame_id = self._frame_counter
        self._frame_counter += 1
        
        self._active_frames[frame_id] = {
            'start_time': time.time(),
            'frame_size': frame_size,
            'capture_time': 0.0,
            'vision_time': 0.0,
            'yolo_time': 0.0,
            'ocr_time': 0.0,
            'minimap_time': 0.0,
            'classification_time': 0.0,
            'queue_size': 0
        }
        
        return frame_id
    
    def record_stage_time(self, frame_id: int, stage: str, duration: float) -> None:
        """Record time for a specific processing stage"""
        if frame_id in self._active_frames and stage in self._active_frames[frame_id]:
            self._active_frames[frame_id][stage] = duration * 1000  # Convert to ms
    
    def record_queue_size(self, frame_id: int, queue_size: int) -> None:
        """Record processing queue size"""
        if frame_id in self._active_frames:
            self._active_frames[frame_id]['queue_size'] = queue_size
    
    def record_frame_drop(self) -> None:
        """Record a dropped frame"""
        self._drop_counter += 1
    
    def finish_frame_processing(self, frame_id: int) -> FrameProcessingMetrics:
        """Finish timing and record complete metrics"""
        if frame_id not in self._active_frames:
            raise ValueError(f"Frame ID {frame_id} not found")
        
        frame_data = self._active_frames.pop(frame_id)
        
        total_time = (
            frame_data['capture_time'] +
            frame_data['vision_time'] +
            frame_data['yolo_time'] +
            frame_data['ocr_time'] +
            frame_data['minimap_time'] +
            frame_data['classification_time']
        )
        
        metrics = FrameProcessingMetrics(
            timestamp=time.time(),
            frame_id=frame_id,
            capture_time=frame_data['capture_time'],
            vision_time=frame_data['vision_time'],
            yolo_time=frame_data['yolo_time'],
            ocr_time=frame_data['ocr_time'],
            minimap_time=frame_data['minimap_time'],
            classification_time=frame_data['classification_time'],
            total_time=total_time,
            queue_size=frame_data['queue_size'],
            frame_drops=self._drop_counter
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_metrics_history(self, duration_seconds: int = 300) -> List[FrameProcessingMetrics]:
        """Get recent frame processing metrics history"""
        cutoff_time = time.time() - duration_seconds
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.get_metrics_history(300)  # Last 5 minutes
        
        total_times = [m.total_time for m in recent_metrics]
        fps_values = [1000.0 / m.total_time if m.total_time > 0 else 0 for m in recent_metrics]
        
        return {
            'total_frames_processed': len(self.metrics_history),
            'recent_frames_processed': len(recent_metrics),
            'total_frames_dropped': self._drop_counter,
            'avg_processing_time': np.mean(total_times) if total_times else 0,
            'avg_fps': np.mean(fps_values) if fps_values else 0,
            'max_fps': np.max(fps_values) if fps_values else 0,
            'pipeline_analysis': self._analyze_pipeline_bottlenecks(recent_metrics)
        }
    
    def _analyze_pipeline_bottlenecks(self, metrics: List[FrameProcessingMetrics]) -> Dict[str, Any]:
        """Analyze pipeline bottlenecks"""
        if not metrics:
            return {}
        
        stages = ['capture_time', 'vision_time', 'yolo_time', 'ocr_time', 'minimap_time', 'classification_time']
        stage_times = {}
        
        for stage in stages:
            times = [getattr(m, stage) for m in metrics]
            stage_times[stage] = np.mean(times) if times else 0
        
        total_avg = sum(stage_times.values())
        
        bottleneck_analysis = {}
        for stage, avg_time in stage_times.items():
            percentage = (avg_time / total_avg * 100) if total_avg > 0 else 0
            bottleneck_analysis[stage] = {
                'avg_time_ms': avg_time,
                'percentage': percentage
            }
        
        primary_bottleneck = max(stage_times.items(), key=lambda x: x[1])[0]
        
        return {
            'stage_breakdown': bottleneck_analysis,
            'primary_bottleneck': primary_bottleneck,
            'efficiency_score': min(100, (1000.0 / total_avg)) if total_avg > 0 else 0
        }


class PerformanceProfiler:
    """Main performance profiler coordinating all profiling systems"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize sub-profilers
        self.system_profiler = SystemProfiler(
            update_interval=self.config.get('system_update_interval', 1.0)
        )
        
        self.gpu_profiler = GPUProfiler(
            update_interval=self.config.get('gpu_update_interval', 1.0)
        )
        
        self.yolo_profiler = YOLOProfiler(
            max_history=self.config.get('yolo_max_history', 1000)
        )
        
        self.frame_profiler = FrameProcessingProfiler(
            max_history=self.config.get('frame_max_history', 1000)
        )
        
        # State management
        self.state = ProfilerState.STOPPED
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Export settings
        self.export_dir = Path(self.config.get('export_dir', 'performance_data'))
        self.export_dir.mkdir(exist_ok=True)
        
        logger.info("Performance profiler initialized")
    
    def start(self) -> None:
        """Start all profiling systems"""
        if self.state == ProfilerState.RUNNING:
            return
        
        self.state = ProfilerState.STARTING
        
        try:
            self.system_profiler.start()
            self.gpu_profiler.start()
            
            self.state = ProfilerState.RUNNING
            self._trigger_callbacks('started')
            logger.info("Performance profiler started")
            
        except Exception as e:
            logger.error(f"Failed to start performance profiler: {e}")
            self.state = ProfilerState.STOPPED
            raise
    
    def stop(self) -> None:
        """Stop all profiling systems"""
        if self.state == ProfilerState.STOPPED:
            return
        
        self.state = ProfilerState.STOPPING
        
        try:
            self.system_profiler.stop()
            self.gpu_profiler.stop()
            
            self.state = ProfilerState.STOPPED
            self._trigger_callbacks('stopped')
            logger.info("Performance profiler stopped")
            
        except Exception as e:
            logger.error(f"Error while stopping performance profiler: {e}")
    
    def pause(self) -> None:
        """Pause profiling"""
        if self.state == ProfilerState.RUNNING:
            self.state = ProfilerState.PAUSED
            self._trigger_callbacks('paused')
    
    def resume(self) -> None:
        """Resume profiling"""
        if self.state == ProfilerState.PAUSED:
            self.state = ProfilerState.RUNNING
            self._trigger_callbacks('resumed')
    
    def get_complete_snapshot(self) -> PerformanceSnapshot:
        """Get complete performance snapshot"""
        timestamp = time.time()
        
        # Collect current metrics from all profilers
        system_metrics = self.system_profiler.get_current_metrics()
        gpu_metrics = self.gpu_profiler.get_current_metrics()
        
        # Get recent metrics from time-based profilers
        yolo_metrics = self.yolo_profiler.get_metrics_history(60)  # Last minute
        frame_metrics = self.frame_profiler.get_metrics_history(60)  # Last minute
        
        return PerformanceSnapshot(
            timestamp=timestamp,
            system=system_metrics,
            gpu=gpu_metrics if gpu_metrics else None,
            yolo=yolo_metrics if yolo_metrics else None,
            frame_processing=frame_metrics if frame_metrics else None
        )
    
    def export_performance_data(self, filename: Optional[str] = None) -> str:
        """Export performance data to JSON file"""
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"performance_data_{timestamp}.json"
        
        filepath = self.export_dir / filename
        
        # Collect comprehensive data
        export_data = {
            'export_timestamp': time.time(),
            'system_metrics': [m.to_dict() for m in self.system_profiler.get_metrics_history(3600)],
            'gpu_metrics': [
                [gpu.to_dict() for gpu in gpu_list] 
                for gpu_list in self.gpu_profiler.get_metrics_history(3600)
            ],
            'yolo_metrics': [m.to_dict() for m in self.yolo_profiler.get_metrics_history(3600)],
            'frame_metrics': [m.to_dict() for m in self.frame_profiler.get_metrics_history(3600)],
            'statistics': {
                'yolo': self.yolo_profiler.get_performance_statistics(),
                'pipeline': self.frame_profiler.get_pipeline_statistics()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Performance data exported to {filepath}")
        return str(filepath)
    
    def add_callback(self, event: str, callback: Callable) -> None:
        """Add event callback"""
        self._callbacks[event].append(callback)
    
    def remove_callback(self, event: str, callback: Callable) -> None:
        """Remove event callback"""
        if callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
    
    def _trigger_callbacks(self, event: str) -> None:
        """Trigger event callbacks"""
        for callback in self._callbacks[event]:
            try:
                callback()
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Get comprehensive bottleneck analysis"""
        yolo_stats = self.yolo_profiler.get_performance_statistics()
        pipeline_stats = self.frame_profiler.get_pipeline_statistics()
        
        # System bottlenecks
        system_metrics = self.system_profiler.get_metrics_history(300)
        gpu_metrics_list = self.gpu_profiler.get_metrics_history(300)
        
        bottlenecks = {
            'yolo_bottlenecks': yolo_stats.get('bottleneck_analysis', {}),
            'pipeline_bottlenecks': pipeline_stats.get('pipeline_analysis', {}),
            'system_bottlenecks': self._analyze_system_bottlenecks(system_metrics, gpu_metrics_list),
            'recommendations': self._generate_optimization_recommendations(yolo_stats, pipeline_stats)
        }
        
        return bottlenecks
    
    def _analyze_system_bottlenecks(self, system_metrics: List[SystemMetrics], 
                                   gpu_metrics_list: List[List[GPUMetrics]]) -> Dict[str, Any]:
        """Analyze system-level bottlenecks"""
        if not system_metrics:
            return {}
        
        # Calculate averages
        avg_cpu = np.mean([m.cpu_usage for m in system_metrics])
        avg_memory = np.mean([m.memory_usage for m in system_metrics])
        
        bottlenecks = {}
        
        if avg_cpu > 80:
            bottlenecks['cpu'] = {'severity': 'high', 'usage': avg_cpu}
        elif avg_cpu > 60:
            bottlenecks['cpu'] = {'severity': 'medium', 'usage': avg_cpu}
        
        if avg_memory > 85:
            bottlenecks['memory'] = {'severity': 'high', 'usage': avg_memory}
        elif avg_memory > 70:
            bottlenecks['memory'] = {'severity': 'medium', 'usage': avg_memory}
        
        # GPU analysis
        if gpu_metrics_list:
            for gpu_metrics in gpu_metrics_list:
                if gpu_metrics:
                    avg_gpu_usage = np.mean([gpu.gpu_usage for gpu in gpu_metrics])
                    avg_gpu_memory = np.mean([gpu.memory_usage for gpu in gpu_metrics])
                    
                    if avg_gpu_usage > 90:
                        bottlenecks['gpu_compute'] = {'severity': 'high', 'usage': avg_gpu_usage}
                    
                    if avg_gpu_memory > 85:
                        bottlenecks['gpu_memory'] = {'severity': 'high', 'usage': avg_gpu_memory}
        
        return bottlenecks
    
    def _generate_optimization_recommendations(self, yolo_stats: Dict[str, Any], 
                                             pipeline_stats: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # YOLO optimizations
        bottleneck_analysis = yolo_stats.get('bottleneck_analysis', {})
        primary_bottleneck = bottleneck_analysis.get('primary_bottleneck')
        
        if primary_bottleneck == 'inference':
            recommendations.append("Consider using a smaller YOLO model or reducing input resolution")
            recommendations.append("Enable TensorRT optimization for RTX 4090")
        elif primary_bottleneck == 'preprocessing':
            recommendations.append("Optimize image preprocessing pipeline")
            recommendations.append("Use GPU-accelerated image processing")
        elif primary_bottleneck == 'postprocessing':
            recommendations.append("Optimize NMS and detection filtering")
            recommendations.append("Reduce maximum detection count")
        
        # Pipeline optimizations
        pipeline_analysis = pipeline_stats.get('pipeline_analysis', {})
        if pipeline_analysis:
            primary_pipeline_bottleneck = pipeline_analysis.get('primary_bottleneck')
            
            if primary_pipeline_bottleneck == 'yolo_time':
                recommendations.append("YOLO inference is the pipeline bottleneck")
            elif primary_pipeline_bottleneck == 'capture_time':
                recommendations.append("Screen capture optimization needed")
            elif primary_pipeline_bottleneck == 'ocr_time':
                recommendations.append("OCR processing optimization needed")
        
        # Frame rate recommendations
        avg_fps = yolo_stats.get('avg_fps', 0)
        if avg_fps < 30:
            recommendations.append("Frame rate below 30 FPS - consider optimization")
        elif avg_fps < 60:
            recommendations.append("Frame rate could be optimized for better performance")
        
        return recommendations


# Global profiler instance
profiler = PerformanceProfiler()