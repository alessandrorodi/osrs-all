"""
Comprehensive Tests for Performance Monitoring and Optimization System

Tests all major components:
- System profiling
- GPU monitoring  
- YOLO profiling
- Frame processing profiling
- Optimization engine
- Performance analytics
"""

import pytest
import time
import threading
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np

# Import performance modules
from performance.profiler import (
    SystemProfiler, GPUProfiler, YOLOProfiler, FrameProcessingProfiler,
    PerformanceProfiler, SystemMetrics, GPUMetrics, YOLOMetrics,
    FrameProcessingMetrics, PerformanceSnapshot
)

from performance.optimization_engine import (
    OptimizationEngine, AdaptiveOptimizer, RTX4090Optimizer,
    SmartFrameSkipper, DynamicResolutionController,
    OptimizationLevel, ModelProfile, OptimizationStrategy,
    OptimizationState, PerformanceOptimizer
)


class TestSystemProfiler:
    """Test system performance profiler"""
    
    def test_initialization(self):
        """Test profiler initialization"""
        profiler = SystemProfiler(update_interval=0.1)
        
        assert profiler.update_interval == 0.1
        assert len(profiler.metrics_history) == 0
        assert not profiler._running
    
    def test_current_metrics(self):
        """Test current metrics collection"""
        profiler = SystemProfiler()
        metrics = profiler.get_current_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert 0 <= metrics.cpu_usage <= 100
        assert 0 <= metrics.memory_usage <= 100
        assert metrics.memory_used_gb >= 0
        assert metrics.memory_total_gb > 0
        assert metrics.process_count > 0
        assert metrics.thread_count > 0
    
    def test_start_stop(self):
        """Test profiler start/stop functionality"""
        profiler = SystemProfiler(update_interval=0.1)
        
        # Start profiler
        profiler.start()
        assert profiler._running
        assert profiler._thread is not None
        
        # Let it collect some data
        time.sleep(0.3)
        
        # Stop profiler
        profiler.stop()
        assert not profiler._running
        assert len(profiler.metrics_history) > 0
    
    def test_metrics_history(self):
        """Test metrics history functionality"""
        profiler = SystemProfiler(update_interval=0.1)
        profiler.start()
        
        time.sleep(0.3)
        profiler.stop()
        
        # Test history retrieval
        all_history = profiler.get_metrics_history(300)
        recent_history = profiler.get_metrics_history(1)
        
        assert len(all_history) >= len(recent_history)
        assert all(isinstance(m, SystemMetrics) for m in all_history)


class TestGPUProfiler:
    """Test GPU performance profiler"""
    
    def test_initialization(self):
        """Test GPU profiler initialization"""
        profiler = GPUProfiler(update_interval=0.1)
        
        assert profiler.update_interval == 0.1
        assert len(profiler.metrics_history) == 0
        assert not profiler._running
    
    def test_current_metrics(self):
        """Test current GPU metrics collection"""
        profiler = GPUProfiler()
        metrics = profiler.get_current_metrics()
        
        # Should return empty list if no GPU available in test environment
        assert isinstance(metrics, list)
    
    @patch('performance.profiler.NVML_AVAILABLE', True)
    @patch('performance.profiler.pynvml')
    def test_nvml_metrics(self, mock_pynvml):
        """Test NVML GPU metrics collection"""
        # Mock NVML responses
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        
        mock_handle = Mock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = b"RTX 4090"
        
        # Mock utilization
        mock_util = Mock()
        mock_util.gpu = 75.0
        mock_util.memory = 50.0
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
        
        # Mock memory info
        mock_mem = Mock()
        mock_mem.used = 8 * 1024**3  # 8GB
        mock_mem.total = 24 * 1024**3  # 24GB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        
        # Mock other metrics
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 65.0
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 350000  # mW
        mock_pynvml.nvmlDeviceGetClockInfo.return_value = 1800.0
        mock_pynvml.nvmlDeviceGetFanSpeed.return_value = 60.0
        
        profiler = GPUProfiler()
        metrics = profiler.get_current_metrics()
        
        assert len(metrics) == 1
        gpu_metric = metrics[0]
        assert gpu_metric.gpu_name == "RTX 4090"
        assert gpu_metric.gpu_usage == 75.0
        assert gpu_metric.memory_usage == 50.0
        assert gpu_metric.temperature == 65.0


class TestYOLOProfiler:
    """Test YOLO inference profiler"""
    
    def test_initialization(self):
        """Test YOLO profiler initialization"""
        profiler = YOLOProfiler(max_history=100)
        
        assert profiler.max_history == 100
        assert len(profiler.metrics_history) == 0
        assert len(profiler._active_inferences) == 0
    
    def test_inference_timing(self):
        """Test YOLO inference timing"""
        profiler = YOLOProfiler()
        
        # Start inference
        inference_id = profiler.start_inference("yolov8n", (640, 640), batch_size=1)
        assert inference_id in profiler._active_inferences
        
        # Record timing stages
        profiler.record_preprocessing(inference_id, 0.01)  # 10ms
        profiler.record_inference(inference_id, 0.05, gpu_memory_used=1024)  # 50ms
        profiler.record_postprocessing(inference_id, 0.005)  # 5ms
        
        # Finish inference
        metrics = profiler.finish_inference(inference_id, detections_count=5)
        
        assert isinstance(metrics, YOLOMetrics)
        assert metrics.model_name == "yolov8n"
        assert metrics.image_size == (640, 640)
        assert metrics.preprocessing_time == 10.0  # Converted to ms
        assert metrics.inference_time == 50.0
        assert metrics.postprocessing_time == 5.0
        assert metrics.total_time == 65.0
        assert metrics.detections_count == 5
        assert metrics.gpu_memory_used == 1024
        assert metrics.fps > 0
        
        # Check metrics stored in history
        assert len(profiler.metrics_history) == 1
    
    def test_performance_statistics(self):
        """Test performance statistics calculation"""
        profiler = YOLOProfiler()
        
        # Generate multiple inference measurements
        for i in range(10):
            inference_id = profiler.start_inference("yolov8n", (640, 640))
            profiler.record_preprocessing(inference_id, 0.01)
            profiler.record_inference(inference_id, 0.05 + i * 0.001)  # Varying inference time
            profiler.record_postprocessing(inference_id, 0.005)
            profiler.finish_inference(inference_id, detections_count=5)
        
        stats = profiler.get_performance_statistics()
        
        assert stats['total_inferences'] == 10
        assert stats['recent_inferences'] == 10
        assert stats['avg_inference_time'] > 0
        assert stats['avg_fps'] > 0
        assert 'bottleneck_analysis' in stats
    
    def test_bottleneck_analysis(self):
        """Test bottleneck analysis"""
        profiler = YOLOProfiler()
        
        # Create inference with preprocessing bottleneck
        inference_id = profiler.start_inference("yolov8n", (640, 640))
        profiler.record_preprocessing(inference_id, 0.1)  # High preprocessing time
        profiler.record_inference(inference_id, 0.02)
        profiler.record_postprocessing(inference_id, 0.005)
        metrics = profiler.finish_inference(inference_id, detections_count=5)
        
        bottlenecks = profiler._analyze_bottlenecks([metrics])
        
        assert bottlenecks['primary_bottleneck'] == 'preprocessing'
        assert bottlenecks['preprocessing_percentage'] > 50


class TestFrameProcessingProfiler:
    """Test frame processing pipeline profiler"""
    
    def test_initialization(self):
        """Test frame profiler initialization"""
        profiler = FrameProcessingProfiler(max_history=100)
        
        assert profiler.max_history == 100
        assert len(profiler.metrics_history) == 0
        assert profiler._frame_counter == 0
        assert profiler._drop_counter == 0
    
    def test_frame_processing_timing(self):
        """Test frame processing timing"""
        profiler = FrameProcessingProfiler()
        
        # Start frame processing
        frame_id = profiler.start_frame_processing((1920, 1080))
        assert frame_id == 0  # First frame
        assert frame_id in profiler._active_frames
        
        # Record stage timings
        profiler.record_stage_time(frame_id, 'capture_time', 0.01)
        profiler.record_stage_time(frame_id, 'vision_time', 0.02)
        profiler.record_stage_time(frame_id, 'yolo_time', 0.05)
        profiler.record_stage_time(frame_id, 'ocr_time', 0.01)
        profiler.record_stage_time(frame_id, 'minimap_time', 0.005)
        profiler.record_stage_time(frame_id, 'classification_time', 0.003)
        profiler.record_queue_size(frame_id, 3)
        
        # Finish frame processing
        metrics = profiler.finish_frame_processing(frame_id)
        
        assert isinstance(metrics, FrameProcessingMetrics)
        assert metrics.frame_id == 0
        assert metrics.capture_time == 10.0  # Converted to ms
        assert metrics.yolo_time == 50.0
        assert metrics.total_time > 0
        assert metrics.queue_size == 3
        
        # Check metrics stored
        assert len(profiler.metrics_history) == 1
    
    def test_frame_drops(self):
        """Test frame drop tracking"""
        profiler = FrameProcessingProfiler()
        
        # Record some frame drops
        profiler.record_frame_drop()
        profiler.record_frame_drop()
        
        # Process a frame
        frame_id = profiler.start_frame_processing((640, 640))
        profiler.record_stage_time(frame_id, 'capture_time', 0.01)
        metrics = profiler.finish_frame_processing(frame_id)
        
        assert metrics.frame_drops == 2
    
    def test_pipeline_statistics(self):
        """Test pipeline statistics calculation"""
        profiler = FrameProcessingProfiler()
        
        # Process multiple frames
        for i in range(5):
            frame_id = profiler.start_frame_processing((640, 640))
            profiler.record_stage_time(frame_id, 'capture_time', 0.01)
            profiler.record_stage_time(frame_id, 'yolo_time', 0.05 + i * 0.01)  # Varying YOLO time
            profiler.record_stage_time(frame_id, 'ocr_time', 0.01)
            profiler.finish_frame_processing(frame_id)
        
        stats = profiler.get_pipeline_statistics()
        
        assert stats['total_frames_processed'] == 5
        assert stats['avg_processing_time'] > 0
        assert stats['avg_fps'] > 0
        assert 'pipeline_analysis' in stats


class TestSmartFrameSkipper:
    """Test smart frame skipping algorithm"""
    
    def test_initialization(self):
        """Test frame skipper initialization"""
        skipper = SmartFrameSkipper(target_fps=60.0)
        
        assert skipper.target_fps == 60.0
        assert skipper.skip_ratio == 0.0
        assert len(skipper.frame_times) == 0
    
    def test_frame_importance_calculation(self):
        """Test frame importance scoring"""
        skipper = SmartFrameSkipper()
        
        # Normal frame
        importance = skipper._calculate_frame_importance({})
        assert importance == 1.0
        
        # High importance frame (combat)
        importance = skipper._calculate_frame_importance({'combat_active': True})
        assert importance > 1.0
        
        # Multiple importance factors
        importance = skipper._calculate_frame_importance({
            'combat_active': True,
            'new_objects': True
        })
        assert importance > 2.0
    
    def test_skip_ratio_update(self):
        """Test skip ratio calculation"""
        skipper = SmartFrameSkipper(target_fps=60.0)
        
        # Add slow frame times (below target FPS)
        slow_frame_time = 1.0 / 30.0  # 30 FPS
        for _ in range(10):
            skipper.record_frame_time(slow_frame_time)
        
        skipper._update_skip_ratio()
        assert skipper.skip_ratio > 0  # Should start skipping frames
        
        # Add fast frame times (above target FPS)
        fast_frame_time = 1.0 / 120.0  # 120 FPS
        for _ in range(20):
            skipper.record_frame_time(fast_frame_time)
        
        skipper._update_skip_ratio()
        assert skipper.skip_ratio == 0  # Should stop skipping frames
    
    def test_should_skip_frame(self):
        """Test frame skipping decision"""
        skipper = SmartFrameSkipper(target_fps=60.0)
        
        # With zero skip ratio, should never skip
        skipper.skip_ratio = 0.0
        assert not skipper.should_skip_frame()
        
        # With high skip ratio, should skip some frames
        skipper.skip_ratio = 0.5
        skip_count = sum(skipper.should_skip_frame() for _ in range(100))
        assert 0 < skip_count < 100  # Should skip some but not all
        
        # High importance frame should be less likely to skip
        skipper.skip_ratio = 0.8
        normal_skips = sum(skipper.should_skip_frame() for _ in range(100))
        important_skips = sum(
            skipper.should_skip_frame({'combat_active': True}) 
            for _ in range(100)
        )
        assert important_skips < normal_skips


class TestDynamicResolutionController:
    """Test dynamic resolution controller"""
    
    def test_initialization(self):
        """Test resolution controller initialization"""
        controller = DynamicResolutionController(base_resolution=(640, 640))
        
        assert controller.base_resolution == (640, 640)
        assert controller.current_resolution == (640, 640)
        assert len(controller.resolution_levels) > 0
    
    def test_resolution_adjustment(self):
        """Test resolution adjustment logic"""
        controller = DynamicResolutionController()
        
        # Underperforming scenario (low FPS, high GPU usage)
        new_res = controller.get_optimal_resolution(
            current_fps=30.0, target_fps=60.0, gpu_usage=90.0
        )
        # Should recommend lower resolution for better performance
        assert new_res[0] <= controller.base_resolution[0]
        
        # Reset controller
        controller = DynamicResolutionController()
        
        # Overperforming scenario (high FPS, low GPU usage)
        new_res = controller.get_optimal_resolution(
            current_fps=100.0, target_fps=60.0, gpu_usage=40.0
        )
        # May recommend higher resolution for better quality
        # (or stay the same if already at optimal)
    
    def test_resolution_level_detection(self):
        """Test resolution level detection"""
        controller = DynamicResolutionController()
        
        # Test known resolution
        level = controller._get_resolution_level((640, 640))
        assert isinstance(level, int)
        assert 0 <= level < len(controller.resolution_levels)
        
        # Test unknown resolution (should find closest)
        level = controller._get_resolution_level((650, 650))
        assert isinstance(level, int)


class TestRTX4090Optimizer:
    """Test RTX 4090 specific optimizer"""
    
    def test_initialization(self):
        """Test RTX 4090 optimizer initialization"""
        optimizer = RTX4090Optimizer()
        
        assert optimizer.max_power_limit == 450
        assert optimizer.thermal_throttle_temp == 83
        assert optimizer.cuda_cores == 16384
        assert isinstance(optimizer.tensorrt_enabled, bool)
    
    def test_yolo_optimization(self):
        """Test YOLO configuration optimization"""
        optimizer = RTX4090Optimizer()
        optimizer.gpu_detected = True  # Force GPU detection
        optimizer.tensorrt_enabled = True
        
        base_config = {'device': 'cpu', 'batch_size': 1}
        optimized = optimizer.optimize_for_yolo(base_config)
        
        assert optimized['device'] == 'cuda:0'
        assert 'engine' in optimized
        assert optimized['cuda_streams'] == 4
        assert optimized['pin_memory'] is True
    
    def test_power_limit_adjustment(self):
        """Test GPU power limit adjustment"""
        optimizer = RTX4090Optimizer()
        optimizer.gpu_detected = True
        
        # High temperature scenario
        new_limit = optimizer.adjust_power_limit(
            target_temp=80.0, current_temp=85.0, current_power=400.0
        )
        assert new_limit < 400.0  # Should reduce power
        
        # Low temperature scenario
        new_limit = optimizer.adjust_power_limit(
            target_temp=80.0, current_temp=70.0, current_power=300.0
        )
        assert new_limit > 300.0  # Should increase power
    
    def test_optimal_batch_size(self):
        """Test optimal batch size calculation"""
        optimizer = RTX4090Optimizer()
        optimizer.gpu_detected = True
        
        # Small model, small resolution
        batch_size = optimizer.get_optimal_batch_size("nano", (416, 416))
        assert batch_size >= 1
        
        # Large model, large resolution
        batch_size = optimizer.get_optimal_batch_size("xlarge", (896, 896))
        assert batch_size >= 1
        assert batch_size <= 16  # Should cap at reasonable limit


class TestAdaptiveOptimizer:
    """Test adaptive optimization controller"""
    
    def test_initialization(self):
        """Test adaptive optimizer initialization"""
        profiler = Mock()
        strategy = OptimizationStrategy()
        
        optimizer = AdaptiveOptimizer(profiler, strategy)
        
        assert optimizer.profiler == profiler
        assert optimizer.strategy == strategy
        assert isinstance(optimizer.current_state, OptimizationState)
        assert len(optimizer.optimization_history) == 0
    
    def test_performance_analysis(self):
        """Test performance analysis"""
        # Mock profiler and snapshot
        profiler = Mock()
        snapshot = Mock()
        snapshot.timestamp = time.time()
        snapshot.system = Mock()
        snapshot.system.cpu_usage = 50.0
        snapshot.system.memory_usage = 60.0
        snapshot.gpu = [Mock()]
        snapshot.gpu[0].gpu_usage = 70.0
        snapshot.gpu[0].temperature = 75.0
        snapshot.yolo = []
        snapshot.frame_processing = []
        
        strategy = OptimizationStrategy()
        optimizer = AdaptiveOptimizer(profiler, strategy)
        
        analysis = optimizer._analyze_performance(snapshot)
        
        assert analysis['cpu_usage'] == 50.0
        assert analysis['gpu_usage'] == 70.0
        assert analysis['gpu_temperature'] == 75.0
        assert isinstance(analysis['bottlenecks'], list)
        assert 0 <= analysis['performance_score'] <= 1
    
    def test_optimization_callbacks(self):
        """Test optimization callbacks"""
        profiler = Mock()
        strategy = OptimizationStrategy()
        optimizer = AdaptiveOptimizer(profiler, strategy)
        
        # Add callback
        callback_called = False
        def test_callback(data):
            nonlocal callback_called
            callback_called = True
        
        optimizer.add_callback('optimization_applied', test_callback)
        optimizer._trigger_callbacks('optimization_applied', {})
        
        assert callback_called


class TestOptimizationEngine:
    """Test main optimization engine"""
    
    def test_initialization(self):
        """Test optimization engine initialization"""
        profiler = Mock()
        
        engine = OptimizationEngine(profiler, OptimizationLevel.BALANCED)
        
        assert engine.profiler == profiler
        assert engine.optimization_level == OptimizationLevel.BALANCED
        assert isinstance(engine.strategy, OptimizationStrategy)
        assert isinstance(engine.adaptive_optimizer, AdaptiveOptimizer)
        assert not engine._running
    
    def test_strategy_creation(self):
        """Test optimization strategy creation"""
        profiler = Mock()
        engine = OptimizationEngine(profiler, OptimizationLevel.AGGRESSIVE)
        
        strategy = engine._create_strategy(OptimizationLevel.CONSERVATIVE)
        assert strategy.target_fps == 30.0
        assert not strategy.frame_skip_enabled
        
        strategy = engine._create_strategy(OptimizationLevel.MAXIMUM_PERFORMANCE)
        assert strategy.target_fps == 120.0
        assert strategy.frame_skip_enabled
    
    def test_start_stop(self):
        """Test engine start/stop"""
        profiler = Mock()
        engine = OptimizationEngine(profiler)
        
        # Start engine
        engine.start()
        assert engine._running
        assert engine._optimization_thread is not None
        
        # Stop engine
        engine.stop()
        assert not engine._running
    
    def test_force_optimization(self):
        """Test force optimization"""
        profiler = Mock()
        engine = OptimizationEngine(profiler)
        
        # Mock the adaptive optimizer
        engine.adaptive_optimizer.optimize = Mock(return_value={'test': 'optimization'})
        
        result = engine.force_optimization()
        
        assert result == {'test': 'optimization'}
        engine.adaptive_optimizer.optimize.assert_called_once_with(force=True)


class TestPerformanceOptimizer:
    """Test high-level performance optimizer"""
    
    def test_initialization(self):
        """Test performance optimizer initialization"""
        profiler = Mock()
        
        optimizer = PerformanceOptimizer(profiler)
        
        assert optimizer.profiler == profiler
        assert isinstance(optimizer.optimization_engine, OptimizationEngine)
        assert len(optimizer.presets) > 0
    
    def test_preset_application(self):
        """Test optimization preset application"""
        profiler = Mock()
        optimizer = PerformanceOptimizer(profiler)
        
        # Mock strategy update
        optimizer.optimization_engine.update_strategy = Mock()
        
        result = optimizer.apply_preset('balanced')
        assert result is True
        
        # Test invalid preset
        result = optimizer.apply_preset('invalid_preset')
        assert result is False
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        profiler = Mock()
        
        # Mock snapshot
        snapshot = Mock()
        snapshot.system = Mock()
        snapshot.system.cpu_usage = 50.0
        snapshot.system.memory_usage = 60.0
        snapshot.gpu = [Mock()]
        snapshot.gpu[0].gpu_usage = 70.0
        snapshot.gpu[0].temperature = 75.0
        
        profiler.get_complete_snapshot.return_value = snapshot
        
        optimizer = PerformanceOptimizer(profiler)
        
        # Mock optimization engine methods
        optimizer.optimization_engine.get_current_state = Mock(return_value=Mock())
        optimizer.optimization_engine.get_optimization_stats = Mock(return_value={})
        optimizer.optimization_engine.get_recommendations = Mock(return_value=[])
        
        summary = optimizer.get_performance_summary()
        
        assert 'timestamp' in summary
        assert 'system_performance' in summary
        assert 'optimization_state' in summary


class TestPerformanceProfiler:
    """Test main performance profiler"""
    
    def test_initialization(self):
        """Test performance profiler initialization"""
        profiler = PerformanceProfiler()
        
        assert isinstance(profiler.system_profiler, SystemProfiler)
        assert isinstance(profiler.gpu_profiler, GPUProfiler)
        assert isinstance(profiler.yolo_profiler, YOLOProfiler)
        assert isinstance(profiler.frame_profiler, FrameProcessingProfiler)
        assert profiler.state.value == "stopped"
    
    def test_start_stop(self):
        """Test profiler start/stop"""
        profiler = PerformanceProfiler()
        
        # Start profiler
        profiler.start()
        assert profiler.state.value == "running"
        
        # Stop profiler
        profiler.stop()
        assert profiler.state.value == "stopped"
    
    def test_complete_snapshot(self):
        """Test complete performance snapshot"""
        profiler = PerformanceProfiler()
        
        snapshot = profiler.get_complete_snapshot()
        
        assert isinstance(snapshot, PerformanceSnapshot)
        assert isinstance(snapshot.system, SystemMetrics)
        assert snapshot.timestamp > 0
    
    def test_export_data(self):
        """Test performance data export"""
        profiler = PerformanceProfiler()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            profiler.export_dir = Path(temp_dir)
            
            exported_file = profiler.export_performance_data("test_export.json")
            
            assert Path(exported_file).exists()
            
            # Verify JSON structure
            with open(exported_file, 'r') as f:
                data = json.load(f)
            
            assert 'export_timestamp' in data
            assert 'system_metrics' in data
            assert 'statistics' in data
    
    def test_bottleneck_analysis(self):
        """Test bottleneck analysis"""
        profiler = PerformanceProfiler()
        
        bottlenecks = profiler.get_bottleneck_analysis()
        
        assert isinstance(bottlenecks, dict)
        assert 'yolo_bottlenecks' in bottlenecks
        assert 'pipeline_bottlenecks' in bottlenecks
        assert 'system_bottlenecks' in bottlenecks
        assert 'recommendations' in bottlenecks


class TestIntegration:
    """Integration tests for the complete performance system"""
    
    def test_profiler_optimizer_integration(self):
        """Test profiler and optimizer working together"""
        # Create profiler
        profiler = PerformanceProfiler()
        
        # Start profiling
        profiler.start()
        time.sleep(0.1)  # Collect some data
        
        # Create optimization engine
        optimizer = OptimizationEngine(profiler, OptimizationLevel.BALANCED)
        
        # Start optimization
        optimizer.start()
        time.sleep(0.1)  # Let it run briefly
        
        # Force optimization
        optimizations = optimizer.force_optimization()
        assert isinstance(optimizations, dict)
        
        # Stop everything
        optimizer.stop()
        profiler.stop()
    
    def test_yolo_profiling_integration(self):
        """Test YOLO profiling in realistic scenario"""
        profiler = PerformanceProfiler()
        
        # Simulate YOLO inference
        yolo_profiler = profiler.yolo_profiler
        
        inference_id = yolo_profiler.start_inference("yolov8n", (640, 640))
        
        # Simulate preprocessing
        time.sleep(0.001)
        yolo_profiler.record_preprocessing(inference_id, 0.01)
        
        # Simulate inference
        time.sleep(0.001)
        yolo_profiler.record_inference(inference_id, 0.05)
        
        # Simulate postprocessing
        time.sleep(0.001)
        yolo_profiler.record_postprocessing(inference_id, 0.005)
        
        # Finish inference
        metrics = yolo_profiler.finish_inference(inference_id, detections_count=3)
        
        assert isinstance(metrics, YOLOMetrics)
        assert metrics.total_time > 0
        assert metrics.fps > 0
        
        # Get performance statistics
        stats = yolo_profiler.get_performance_statistics()
        assert stats['total_inferences'] == 1


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks to ensure system efficiency"""
    
    def test_profiler_overhead(self):
        """Test that profiling doesn't significantly impact performance"""
        # Measure baseline performance
        start_time = time.time()
        for _ in range(1000):
            pass  # Minimal work
        baseline_time = time.time() - start_time
        
        # Measure with profiling
        profiler = PerformanceProfiler()
        profiler.start()
        
        start_time = time.time()
        for _ in range(1000):
            pass  # Same minimal work
        profiled_time = time.time() - start_time
        
        profiler.stop()
        
        # Profiling overhead should be minimal
        overhead = (profiled_time - baseline_time) / baseline_time
        assert overhead < 0.1  # Less than 10% overhead
    
    def test_optimization_speed(self):
        """Test that optimization decisions are made quickly"""
        profiler = PerformanceProfiler()
        optimizer = OptimizationEngine(profiler)
        
        start_time = time.time()
        optimizer.force_optimization()
        optimization_time = time.time() - start_time
        
        # Optimization should be fast
        assert optimization_time < 0.1  # Less than 100ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])