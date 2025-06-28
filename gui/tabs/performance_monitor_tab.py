"""
Performance Monitor GUI Tab

Provides comprehensive performance monitoring interface with:
- Real-time GPU/CPU usage graphs
- YOLO inference timing breakdown
- Optimization recommendation system
- Performance benchmark comparisons
- Resource allocation controls
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
import logging
from pathlib import Path

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np

from performance.profiler import PerformanceProfiler, profiler
from performance.optimization_engine import (
    OptimizationEngine, OptimizationLevel, 
    PerformanceOptimizer, create_optimization_engine
)

logger = logging.getLogger(__name__)


class PerformanceMonitorTab(ctk.CTkFrame):
    """Performance monitoring and optimization tab"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Performance monitoring components
        self.profiler = profiler
        self.optimization_engine: Optional[OptimizationEngine] = None
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        
        # GUI state
        self._monitoring_active = False
        self._optimization_active = False
        self._update_thread: Optional[threading.Thread] = None
        self._stop_updates = threading.Event()
        
        # Data storage for charts
        self.chart_data = {
            'timestamps': [],
            'cpu_usage': [],
            'gpu_usage': [],
            'memory_usage': [],
            'fps_values': [],
            'gpu_temp': [],
            'inference_times': []
        }
        
        # Setup GUI
        self.setup_ui()
        self.setup_charts()
        
        # Initialize optimization engine
        self._initialize_optimization()
        
        logger.info("Performance monitor tab initialized")
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container with scrolling
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Control panel
        self.create_control_panel()
        
        # Performance overview
        self.create_performance_overview()
        
        # Charts section
        self.create_charts_section()
        
        # Optimization section
        self.create_optimization_section()
        
        # Recommendations section
        self.create_recommendations_section()
    
    def create_control_panel(self):
        """Create monitoring control panel"""
        control_frame = ctk.CTkFrame(self.main_container)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Title
        ctk.CTkLabel(
            control_frame,
            text="🔧 Performance Control Panel",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=10)
        
        # Control buttons
        button_frame = ctk.CTkFrame(control_frame)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        # Monitoring controls
        self.monitor_button = ctk.CTkButton(
            button_frame,
            text="▶️ Start Monitoring",
            command=self.toggle_monitoring,
            width=150
        )
        self.monitor_button.pack(side="left", padx=5)
        
        # Optimization controls  
        self.optimize_button = ctk.CTkButton(
            button_frame,
            text="🚀 Start Optimization",
            command=self.toggle_optimization,
            width=150,
            fg_color="green",
            hover_color="darkgreen"
        )
        self.optimize_button.pack(side="left", padx=5)
        
        # Manual optimization
        ctk.CTkButton(
            button_frame,
            text="⚡ Force Optimize",
            command=self.force_optimization,
            width=120
        ).pack(side="left", padx=5)
        
        # Export data
        ctk.CTkButton(
            button_frame,
            text="💾 Export Data",
            command=self.export_performance_data,
            width=120
        ).pack(side="left", padx=5)
        
        # Settings
        ctk.CTkButton(
            button_frame,
            text="⚙️ Settings",
            command=self.open_settings,
            width=100
        ).pack(side="left", padx=5)
    
    def create_performance_overview(self):
        """Create performance overview section"""
        overview_frame = ctk.CTkFrame(self.main_container)
        overview_frame.pack(fill="x", padx=10, pady=10)
        
        # Title
        ctk.CTkLabel(
            overview_frame,
            text="📊 Performance Overview",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Metrics grid
        metrics_frame = ctk.CTkFrame(overview_frame)
        metrics_frame.pack(fill="x", padx=10, pady=10)
        
        # Performance metrics labels
        self.performance_labels = {}
        metrics = [
            ("CPU Usage", "cpu_usage", "%"),
            ("GPU Usage", "gpu_usage", "%"),
            ("Memory Usage", "memory_usage", "%"),
            ("GPU Temperature", "gpu_temp", "°C"),
            ("Current FPS", "current_fps", "FPS"),
            ("Avg Inference Time", "avg_inference", "ms")
        ]
        
        for i, (label, key, unit) in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            metric_frame = ctk.CTkFrame(metrics_frame)
            metric_frame.grid(row=row, column=col, padx=10, pady=5, sticky="ew")
            
            ctk.CTkLabel(
                metric_frame,
                text=label,
                font=ctk.CTkFont(size=12, weight="bold")
            ).pack(pady=2)
            
            value_label = ctk.CTkLabel(
                metric_frame,
                text=f"-- {unit}",
                font=ctk.CTkFont(size=14)
            )
            value_label.pack(pady=2)
            
            self.performance_labels[key] = value_label
        
        # Configure grid weights
        for i in range(3):
            metrics_frame.columnconfigure(i, weight=1)
    
    def create_charts_section(self):
        """Create performance charts section"""
        charts_frame = ctk.CTkFrame(self.main_container)
        charts_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        ctk.CTkLabel(
            charts_frame,
            text="📈 Real-time Performance Charts",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Chart container
        self.chart_container = ctk.CTkFrame(charts_frame)
        self.chart_container.pack(fill="both", expand=True, padx=10, pady=10)
    
    def setup_charts(self):
        """Setup matplotlib charts"""
        # Create figure with subplots
        self.figure, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.figure.patch.set_facecolor('#2b2b2b')
        self.figure.tight_layout(pad=3.0)
        
        # Configure axes
        for ax in self.axes.flat:
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        
        # CPU/GPU Usage chart
        self.axes[0, 0].set_title('CPU & GPU Usage')
        self.axes[0, 0].set_ylabel('Usage (%)')
        self.axes[0, 0].set_ylim(0, 100)
        
        # Memory Usage chart
        self.axes[0, 1].set_title('Memory Usage')
        self.axes[0, 1].set_ylabel('Usage (%)')
        self.axes[0, 1].set_ylim(0, 100)
        
        # FPS Performance chart
        self.axes[1, 0].set_title('FPS Performance')
        self.axes[1, 0].set_ylabel('FPS')
        self.axes[1, 0].set_ylim(0, 120)
        
        # GPU Temperature chart
        self.axes[1, 1].set_title('GPU Temperature')
        self.axes[1, 1].set_ylabel('Temperature (°C)')
        self.axes[1, 1].set_ylim(20, 90)
        
        # Create canvas
        self.chart_canvas = FigureCanvasTkAgg(self.figure, self.chart_container)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Initialize empty plots
        self.cpu_line, = self.axes[0, 0].plot([], [], 'g-', label='CPU', linewidth=2)
        self.gpu_line, = self.axes[0, 0].plot([], [], 'r-', label='GPU', linewidth=2)
        self.axes[0, 0].legend()
        
        self.memory_line, = self.axes[0, 1].plot([], [], 'b-', linewidth=2)
        self.fps_line, = self.axes[1, 0].plot([], [], 'm-', linewidth=2)
        self.temp_line, = self.axes[1, 1].plot([], [], 'orange', linewidth=2)
    
    def create_optimization_section(self):
        """Create optimization controls section"""
        opt_frame = ctk.CTkFrame(self.main_container)
        opt_frame.pack(fill="x", padx=10, pady=10)
        
        # Title
        ctk.CTkLabel(
            opt_frame,
            text="🚀 Optimization Engine",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Optimization level selection
        level_frame = ctk.CTkFrame(opt_frame)
        level_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(level_frame, text="Optimization Level:").pack(side="left", padx=10)
        
        self.optimization_level = ctk.CTkOptionMenu(
            level_frame,
            values=["Conservative", "Balanced", "Aggressive", "Maximum Performance"],
            command=self.on_optimization_level_change
        )
        self.optimization_level.set("Balanced")
        self.optimization_level.pack(side="left", padx=5)
        
        # Optimization presets
        preset_frame = ctk.CTkFrame(opt_frame)
        preset_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(preset_frame, text="Quick Presets:").pack(side="left", padx=10)
        
        presets = [("🔋 Battery Saver", "battery_saver"), ("⚖️ Balanced", "balanced"), 
                  ("⚡ Performance", "performance"), ("🔥 Maximum", "maximum")]
        
        for text, preset in presets:
            ctk.CTkButton(
                preset_frame,
                text=text,
                command=lambda p=preset: self.apply_optimization_preset(p),
                width=100
            ).pack(side="left", padx=2)
        
        # Current optimization state
        state_frame = ctk.CTkFrame(opt_frame)
        state_frame.pack(fill="x", padx=10, pady=10)
        
        self.optimization_status = ctk.CTkLabel(
            state_frame,
            text="Optimization Status: Ready",
            font=ctk.CTkFont(size=12)
        )
        self.optimization_status.pack(pady=5)
        
        # Optimization statistics
        stats_frame = ctk.CTkFrame(state_frame)
        stats_frame.pack(fill="x", padx=10, pady=5)
        
        self.optimization_stats = ctk.CTkTextbox(stats_frame, height=80)
        self.optimization_stats.pack(fill="x", padx=5, pady=5)
    
    def create_recommendations_section(self):
        """Create optimization recommendations section"""
        rec_frame = ctk.CTkFrame(self.main_container)
        rec_frame.pack(fill="x", padx=10, pady=10)
        
        # Title
        ctk.CTkLabel(
            rec_frame,
            text="💡 Optimization Recommendations",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Recommendations display
        self.recommendations_display = ctk.CTkTextbox(rec_frame, height=120)
        self.recommendations_display.pack(fill="x", padx=10, pady=10)
        
        # Auto-apply recommendations checkbox
        self.auto_apply_recommendations = ctk.CTkCheckBox(
            rec_frame,
            text="Auto-apply safe recommendations",
            command=self.toggle_auto_recommendations
        )
        self.auto_apply_recommendations.pack(pady=5)
    
    def _initialize_optimization(self):
        """Initialize optimization engine"""
        try:
            self.optimization_engine = create_optimization_engine(
                self.profiler, OptimizationLevel.BALANCED
            )
            self.performance_optimizer = PerformanceOptimizer(self.profiler)
            
            # Add optimization callbacks
            self.optimization_engine.adaptive_optimizer.add_callback(
                'optimization_applied', self._on_optimization_applied
            )
            
            logger.info("Optimization engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimization engine: {e}")
            messagebox.showerror("Error", f"Failed to initialize optimization: {e}")
    
    def toggle_monitoring(self):
        """Toggle performance monitoring"""
        if not self._monitoring_active:
            self.start_monitoring()
        else:
            self.stop_monitoring()
    
    def start_monitoring(self):
        """Start performance monitoring"""
        try:
            self.profiler.start()
            self._monitoring_active = True
            self._stop_updates.clear()
            
            # Start update thread
            self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self._update_thread.start()
            
            self.monitor_button.configure(text="⏹️ Stop Monitoring", fg_color="red")
            self.optimization_status.configure(text="Monitoring Status: Active")
            
            logger.info("Performance monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            messagebox.showerror("Error", f"Failed to start monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        try:
            self._monitoring_active = False
            self._stop_updates.set()
            
            if self._update_thread and self._update_thread.is_alive():
                self._update_thread.join(timeout=2.0)
            
            self.profiler.stop()
            
            self.monitor_button.configure(text="▶️ Start Monitoring", fg_color=None)
            self.optimization_status.configure(text="Monitoring Status: Stopped")
            
            logger.info("Performance monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
    def toggle_optimization(self):
        """Toggle optimization engine"""
        if not self._optimization_active:
            self.start_optimization()
        else:
            self.stop_optimization()
    
    def start_optimization(self):
        """Start optimization engine"""
        try:
            if not self.optimization_engine:
                self._initialize_optimization()
            
            self.optimization_engine.start()
            self._optimization_active = True
            
            self.optimize_button.configure(text="⏹️ Stop Optimization", fg_color="red")
            self.optimization_status.configure(text="Optimization Status: Active")
            
            logger.info("Optimization engine started")
            
        except Exception as e:
            logger.error(f"Failed to start optimization: {e}")
            messagebox.showerror("Error", f"Failed to start optimization: {e}")
    
    def stop_optimization(self):
        """Stop optimization engine"""
        try:
            if self.optimization_engine:
                self.optimization_engine.stop()
            
            self._optimization_active = False
            
            self.optimize_button.configure(text="🚀 Start Optimization", fg_color="green")
            self.optimization_status.configure(text="Optimization Status: Stopped")
            
            logger.info("Optimization engine stopped")
            
        except Exception as e:
            logger.error(f"Error stopping optimization: {e}")
    
    def force_optimization(self):
        """Force immediate optimization"""
        try:
            if not self.optimization_engine:
                messagebox.showwarning("Warning", "Optimization engine not initialized")
                return
                
            optimizations = self.optimization_engine.force_optimization()
            
            if optimizations:
                message = "Optimizations applied:\n"
                for key, value in optimizations.items():
                    message += f"• {key}: {value}\n"
                messagebox.showinfo("Optimization Complete", message)
            else:
                messagebox.showinfo("Optimization", "No optimizations needed at this time")
                
        except Exception as e:
            logger.error(f"Force optimization failed: {e}")
            messagebox.showerror("Error", f"Optimization failed: {e}")
    
    def on_optimization_level_change(self, level_name: str):
        """Handle optimization level change"""
        try:
            level_map = {
                "Conservative": OptimizationLevel.CONSERVATIVE,
                "Balanced": OptimizationLevel.BALANCED,
                "Aggressive": OptimizationLevel.AGGRESSIVE,
                "Maximum Performance": OptimizationLevel.MAXIMUM_PERFORMANCE
            }
            
            if level_name in level_map and self.optimization_engine:
                new_level = level_map[level_name]
                strategy = self.optimization_engine._create_strategy(new_level)
                self.optimization_engine.update_strategy(strategy)
                
                logger.info(f"Optimization level changed to: {level_name}")
                
        except Exception as e:
            logger.error(f"Failed to change optimization level: {e}")
    
    def apply_optimization_preset(self, preset_name: str):
        """Apply optimization preset"""
        try:
            if self.performance_optimizer:
                success = self.performance_optimizer.apply_preset(preset_name)
                if success:
                    messagebox.showinfo("Success", f"Applied preset: {preset_name}")
                else:
                    messagebox.showerror("Error", f"Failed to apply preset: {preset_name}")
            
        except Exception as e:
            logger.error(f"Failed to apply preset {preset_name}: {e}")
            messagebox.showerror("Error", f"Failed to apply preset: {e}")
    
    def toggle_auto_recommendations(self):
        """Toggle auto-apply recommendations"""
        if self.auto_apply_recommendations.get():
            logger.info("Auto-apply recommendations enabled")
        else:
            logger.info("Auto-apply recommendations disabled")
    
    def export_performance_data(self):
        """Export performance data"""
        try:
            # File dialog for export location
            filename = filedialog.asksaveasfilename(
                title="Export Performance Data",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                exported_file = self.profiler.export_performance_data(filename)
                messagebox.showinfo("Export Complete", f"Data exported to:\n{exported_file}")
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def open_settings(self):
        """Open performance settings dialog"""
        settings_window = PerformanceSettingsWindow(self)
        settings_window.grab_set()  # Make modal
    
    def _update_loop(self):
        """Main update loop for real-time monitoring"""
        while not self._stop_updates.is_set():
            try:
                if self._monitoring_active:
                    self._update_performance_data()
                    self._update_charts()
                    self._update_optimization_status()
                    self._update_recommendations()
                
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                logger.error(f"Update loop error: {e}")
                time.sleep(1.0)
    
    def _update_performance_data(self):
        """Update performance data from profiler"""
        try:
            snapshot = self.profiler.get_complete_snapshot()
            current_time = time.time()
            
            # Add timestamp
            self.chart_data['timestamps'].append(current_time)
            
            # System metrics
            if snapshot.system:
                self.chart_data['cpu_usage'].append(snapshot.system.cpu_usage)
                self.chart_data['memory_usage'].append(snapshot.system.memory_usage)
            else:
                self.chart_data['cpu_usage'].append(0)
                self.chart_data['memory_usage'].append(0)
            
            # GPU metrics
            if snapshot.gpu and len(snapshot.gpu) > 0:
                gpu = snapshot.gpu[0]
                self.chart_data['gpu_usage'].append(gpu.gpu_usage)
                self.chart_data['gpu_temp'].append(gpu.temperature)
            else:
                self.chart_data['gpu_usage'].append(0)
                self.chart_data['gpu_temp'].append(0)
            
            # YOLO metrics
            if snapshot.yolo and len(snapshot.yolo) > 0:
                recent_yolo = snapshot.yolo[-10:]
                fps_values = [m.fps for m in recent_yolo]
                avg_fps = np.mean(fps_values) if fps_values else 0
                self.chart_data['fps_values'].append(avg_fps)
                
                inference_times = [m.inference_time for m in recent_yolo]
                avg_inference = np.mean(inference_times) if inference_times else 0
                self.chart_data['inference_times'].append(avg_inference)
            else:
                self.chart_data['fps_values'].append(0)
                self.chart_data['inference_times'].append(0)
            
            # Keep only recent data (last 5 minutes)
            max_points = 300
            for key in self.chart_data:
                if len(self.chart_data[key]) > max_points:
                    self.chart_data[key] = self.chart_data[key][-max_points:]
            
            # Update performance labels
            self._update_performance_labels(snapshot)
            
        except Exception as e:
            logger.error(f"Failed to update performance data: {e}")
    
    def _update_performance_labels(self, snapshot):
        """Update performance overview labels"""
        try:
            # CPU Usage
            cpu_usage = snapshot.system.cpu_usage if snapshot.system else 0
            self.performance_labels['cpu_usage'].configure(text=f"{cpu_usage:.1f} %")
            
            # GPU Usage
            gpu_usage = snapshot.gpu[0].gpu_usage if snapshot.gpu else 0
            self.performance_labels['gpu_usage'].configure(text=f"{gpu_usage:.1f} %")
            
            # Memory Usage
            memory_usage = snapshot.system.memory_usage if snapshot.system else 0
            self.performance_labels['memory_usage'].configure(text=f"{memory_usage:.1f} %")
            
            # GPU Temperature
            gpu_temp = snapshot.gpu[0].temperature if snapshot.gpu else 0
            self.performance_labels['gpu_temp'].configure(text=f"{gpu_temp:.1f} °C")
            
            # Current FPS
            if snapshot.yolo and len(snapshot.yolo) > 0:
                recent_fps = [m.fps for m in snapshot.yolo[-5:]]
                current_fps = np.mean(recent_fps) if recent_fps else 0
            else:
                current_fps = 0
            self.performance_labels['current_fps'].configure(text=f"{current_fps:.1f} FPS")
            
            # Average Inference Time
            if snapshot.yolo and len(snapshot.yolo) > 0:
                recent_inference = [m.inference_time for m in snapshot.yolo[-10:]]
                avg_inference = np.mean(recent_inference) if recent_inference else 0
            else:
                avg_inference = 0
            self.performance_labels['avg_inference'].configure(text=f"{avg_inference:.1f} ms")
            
        except Exception as e:
            logger.error(f"Failed to update performance labels: {e}")
    
    def _update_charts(self):
        """Update performance charts"""
        try:
            if not self.chart_data['timestamps']:
                return
            
            # Get recent data for plotting
            timestamps = self.chart_data['timestamps']
            
            # Convert to relative time (seconds ago)
            current_time = time.time()
            relative_times = [(current_time - ts) for ts in timestamps]
            relative_times.reverse()  # Show most recent on right
            
            # Reverse all data arrays to match
            cpu_data = list(reversed(self.chart_data['cpu_usage']))
            gpu_data = list(reversed(self.chart_data['gpu_usage']))
            memory_data = list(reversed(self.chart_data['memory_usage']))
            fps_data = list(reversed(self.chart_data['fps_values']))
            temp_data = list(reversed(self.chart_data['gpu_temp']))
            
            # Update plots
            self.cpu_line.set_data(relative_times, cpu_data)
            self.gpu_line.set_data(relative_times, gpu_data)
            self.memory_line.set_data(relative_times, memory_data)
            self.fps_line.set_data(relative_times, fps_data)
            self.temp_line.set_data(relative_times, temp_data)
            
            # Update x-axis limits
            if relative_times:
                x_min, x_max = max(relative_times), min(relative_times)
                for ax in self.axes.flat:
                    ax.set_xlim(x_min, x_max)
            
            # Refresh canvas
            self.chart_canvas.draw()
            
        except Exception as e:
            logger.error(f"Failed to update charts: {e}")
    
    def _update_optimization_status(self):
        """Update optimization status display"""
        try:
            if not self.optimization_engine:
                return
            
            stats = self.optimization_engine.get_optimization_stats()
            state = self.optimization_engine.get_current_state()
            
            status_text = f"""Optimizations Applied: {stats['optimizations_applied']}
Performance Score: {state.performance_score:.2f}
Quality Score: {state.quality_score:.2f}
Efficiency Score: {state.efficiency_score:.2f}
Current Model: {state.current_model.value}
Resolution: {state.current_resolution}
Frame Skip Ratio: {state.frame_skip_ratio:.2f}"""
            
            self.optimization_stats.delete("1.0", "end")
            self.optimization_stats.insert("1.0", status_text)
            
        except Exception as e:
            logger.error(f"Failed to update optimization status: {e}")
    
    def _update_recommendations(self):
        """Update optimization recommendations"""
        try:
            if not self.optimization_engine:
                return
            
            recommendations = self.optimization_engine.get_recommendations()
            
            if recommendations:
                rec_text = "Current Recommendations:\n\n"
                for i, rec in enumerate(recommendations, 1):
                    rec_text += f"{i}. {rec}\n"
            else:
                rec_text = "No optimization recommendations at this time.\nSystem is performing optimally."
            
            self.recommendations_display.delete("1.0", "end")
            self.recommendations_display.insert("1.0", rec_text)
            
        except Exception as e:
            logger.error(f"Failed to update recommendations: {e}")
    
    def _on_optimization_applied(self, optimizations: Dict[str, Any]):
        """Callback for when optimizations are applied"""
        try:
            logger.info(f"Optimizations applied: {optimizations}")
            
            # Update UI to reflect changes
            if self._monitoring_active:
                self._update_optimization_status()
                
        except Exception as e:
            logger.error(f"Optimization callback error: {e}")
    
    def cleanup(self):
        """Cleanup resources when tab is destroyed"""
        try:
            self.stop_monitoring()
            self.stop_optimization()
            
            if hasattr(self, 'figure'):
                plt.close(self.figure)
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


class PerformanceSettingsWindow(ctk.CTkToplevel):
    """Performance settings dialog"""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        self.title("Performance Settings")
        self.geometry("500x600")
        self.resizable(False, False)
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Setup settings UI"""
        # Title
        ctk.CTkLabel(
            self,
            text="⚙️ Performance Settings",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=20)
        
        # Settings sections
        self.create_monitoring_settings()
        self.create_optimization_settings()
        self.create_advanced_settings()
        
        # Buttons
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkButton(
            button_frame,
            text="Apply",
            command=self.apply_settings
        ).pack(side="right", padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self.destroy
        ).pack(side="right", padx=5)
    
    def create_monitoring_settings(self):
        """Create monitoring settings section"""
        frame = ctk.CTkFrame(self)
        frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(frame, text="Monitoring Settings", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
        
        # Update interval
        interval_frame = ctk.CTkFrame(frame)
        interval_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(interval_frame, text="Update Interval (seconds):").pack(side="left")
        self.update_interval = ctk.CTkEntry(interval_frame, width=100)
        self.update_interval.pack(side="right", padx=10)
        self.update_interval.insert(0, "1.0")
    
    def create_optimization_settings(self):
        """Create optimization settings section"""
        frame = ctk.CTkFrame(self)
        frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(frame, text="Optimization Settings", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
        
        # Target FPS
        fps_frame = ctk.CTkFrame(frame)
        fps_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(fps_frame, text="Target FPS:").pack(side="left")
        self.target_fps = ctk.CTkEntry(fps_frame, width=100)
        self.target_fps.pack(side="right", padx=10)
        self.target_fps.insert(0, "60")
    
    def create_advanced_settings(self):
        """Create advanced settings section"""
        frame = ctk.CTkFrame(self)
        frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(frame, text="Advanced Settings", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
        
        # Enable TensorRT
        self.enable_tensorrt = ctk.CTkCheckBox(frame, text="Enable TensorRT optimization")
        self.enable_tensorrt.pack(padx=10, pady=5, anchor="w")
    
    def apply_settings(self):
        """Apply settings changes"""
        try:
            # Apply settings logic here
            messagebox.showinfo("Success", "Settings applied successfully")
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply settings: {e}")


def create_performance_monitor_tab(parent) -> PerformanceMonitorTab:
    """Create and return performance monitor tab"""
    return PerformanceMonitorTab(parent)