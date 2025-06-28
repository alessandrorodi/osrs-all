#!/usr/bin/env python3
"""
OSRS Bot Framework - Advanced Safety GUI Tab

Real-time behavior analysis, risk assessment dashboard, and safety recommendation system.
Integrates with the AI behavior modeling and advanced anti-detection systems.
"""

import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

try:
    import customtkinter as ctk
    import tkinter as tk
    from tkinter import ttk, messagebox
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.animation import FuncAnimation
    import numpy as np
except ImportError as e:
    print(f"Missing GUI dependencies: {e}")
    sys.exit(1)

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.logging import get_logger
from safety.behavior_modeling import get_behavior_model
from safety.advanced_anti_detection import get_anti_detection_system

logger = get_logger(__name__)


class AdvancedSafetyTab:
    """Advanced Safety tab with behavior analysis and risk assessment"""
    
    def __init__(self, parent):
        self.parent = parent
        self.behavior_model = get_behavior_model()
        self.anti_detection = get_anti_detection_system()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.update_interval = 1000  # 1 second
        
        # Data for charts
        self.risk_data = []
        self.behavior_data = []
        self.time_data = []
        self.max_data_points = 100
        
        # GUI components
        self.setup_ui()
        self.setup_charts()
        
        # Start monitoring
        self.start_monitoring()
        
        logger.info("Advanced Safety tab initialized")
    
    def setup_ui(self):
        """Setup the Advanced Safety tab UI"""
        # Main container
        self.main_frame = ctk.CTkFrame(self.parent)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create header
        self.create_header()
        
        # Create main content area with tabs
        self.create_content_area()
        
        # Create control panel
        self.create_control_panel()
        
        # Create status bar
        self.create_status_bar()
    
    def create_header(self):
        """Create header with title and quick stats"""
        header_frame = ctk.CTkFrame(self.main_frame)
        header_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="🛡️ Advanced Safety & Anti-Detection",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(side="left", padx=20, pady=10)
        
        # Quick stats frame
        stats_frame = ctk.CTkFrame(header_frame)
        stats_frame.pack(side="right", padx=20, pady=5)
        
        # Risk level indicator
        self.risk_level_label = ctk.CTkLabel(
            stats_frame,
            text="Risk: LOW",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="green"
        )
        self.risk_level_label.pack(side="left", padx=10, pady=5)
        
        # Behavior score
        self.behavior_score_label = ctk.CTkLabel(
            stats_frame,
            text="Behavior: HUMAN-LIKE",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="blue"
        )
        self.behavior_score_label.pack(side="left", padx=10, pady=5)
        
        # Safety status
        self.safety_status_label = ctk.CTkLabel(
            stats_frame,
            text="Status: PROTECTED",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="green"
        )
        self.safety_status_label.pack(side="left", padx=10, pady=5)
    
    def create_content_area(self):
        """Create main content area with tabbed interface"""
        # Create notebook for sub-tabs
        self.content_notebook = ctk.CTkTabview(self.main_frame)
        self.content_notebook.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create sub-tabs
        self.create_risk_dashboard_tab()
        self.create_behavior_analysis_tab()
        self.create_safety_recommendations_tab()
        self.create_configuration_tab()
        self.create_statistics_tab()
    
    def create_risk_dashboard_tab(self):
        """Create real-time risk assessment dashboard"""
        risk_tab = self.content_notebook.add("🎯 Risk Dashboard")
        
        # Top section - Current risk overview
        overview_frame = ctk.CTkFrame(risk_tab)
        overview_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            overview_frame,
            text="Current Risk Assessment",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        # Risk meters grid
        meters_frame = ctk.CTkFrame(overview_frame)
        meters_frame.pack(fill="x", padx=10, pady=10)
        
        # Overall risk meter
        overall_frame = ctk.CTkFrame(meters_frame)
        overall_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(overall_frame, text="Overall Risk", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.overall_risk_progress = ctk.CTkProgressBar(overall_frame, width=200, height=20)
        self.overall_risk_progress.pack(pady=5)
        self.overall_risk_value = ctk.CTkLabel(overall_frame, text="0.0%")
        self.overall_risk_value.pack()
        
        # Detection risk meter
        detection_frame = ctk.CTkFrame(meters_frame)
        detection_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(detection_frame, text="Detection Risk", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.detection_risk_progress = ctk.CTkProgressBar(detection_frame, width=200, height=20)
        self.detection_risk_progress.pack(pady=5)
        self.detection_risk_value = ctk.CTkLabel(detection_frame, text="0.0%")
        self.detection_risk_value.pack()
        
        # Pattern risk meter
        pattern_frame = ctk.CTkFrame(meters_frame)
        pattern_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(pattern_frame, text="Pattern Risk", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.pattern_risk_progress = ctk.CTkProgressBar(pattern_frame, width=200, height=20)
        self.pattern_risk_progress.pack(pady=5)
        self.pattern_risk_value = ctk.CTkLabel(pattern_frame, text="0.0%")
        self.pattern_risk_value.pack()
        
        # Configure grid weights
        meters_frame.grid_columnconfigure(0, weight=1)
        meters_frame.grid_columnconfigure(1, weight=1)
        meters_frame.grid_columnconfigure(2, weight=1)
        
        # Real-time risk chart
        chart_frame = ctk.CTkFrame(risk_tab)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            chart_frame,
            text="Risk Timeline (Last 100 Updates)",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        # Create matplotlib figure for risk chart
        self.risk_figure, self.risk_ax = plt.subplots(figsize=(10, 4))
        self.risk_figure.patch.set_facecolor('#2b2b2b')
        self.risk_ax.set_facecolor('#2b2b2b')
        self.risk_ax.tick_params(colors='white')
        self.risk_ax.set_xlabel('Time', color='white')
        self.risk_ax.set_ylabel('Risk Level', color='white')
        self.risk_ax.set_ylim(0, 1)
        
        self.risk_canvas = FigureCanvasTkAgg(self.risk_figure, chart_frame)
        self.risk_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_behavior_analysis_tab(self):
        """Create behavioral analysis tab"""
        behavior_tab = self.content_notebook.add("🧠 Behavior Analysis")
        
        # Behavior profile section
        profile_frame = ctk.CTkFrame(behavior_tab)
        profile_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            profile_frame,
            text="Current Behavior Profile",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        # Behavior metrics
        metrics_frame = ctk.CTkFrame(profile_frame)
        metrics_frame.pack(fill="x", padx=10, pady=10)
        
        # Left column - Mouse behavior
        mouse_frame = ctk.CTkFrame(metrics_frame)
        mouse_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(mouse_frame, text="Mouse Behavior", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.mouse_behavior_text = ctk.CTkTextbox(mouse_frame, height=150)
        self.mouse_behavior_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Right column - Timing behavior
        timing_frame = ctk.CTkFrame(metrics_frame)
        timing_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(timing_frame, text="Timing Behavior", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.timing_behavior_text = ctk.CTkTextbox(timing_frame, height=150)
        self.timing_behavior_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Behavior visualization
        viz_frame = ctk.CTkFrame(behavior_tab)
        viz_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            viz_frame,
            text="Behavior Pattern Analysis",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        # Create matplotlib figure for behavior visualization
        self.behavior_figure, (self.behavior_ax1, self.behavior_ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.behavior_figure.patch.set_facecolor('#2b2b2b')
        
        for ax in [self.behavior_ax1, self.behavior_ax2]:
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
        
        self.behavior_ax1.set_title('Mouse Speed Distribution', color='white')
        self.behavior_ax2.set_title('Reaction Time Distribution', color='white')
        
        self.behavior_canvas = FigureCanvasTkAgg(self.behavior_figure, viz_frame)
        self.behavior_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_safety_recommendations_tab(self):
        """Create safety recommendations tab"""
        recommendations_tab = self.content_notebook.add("💡 Recommendations")
        
        # Current recommendations
        current_frame = ctk.CTkFrame(recommendations_tab)
        current_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            current_frame,
            text="Current Safety Recommendations",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        # Recommendations list
        self.recommendations_frame = ctk.CTkScrollableFrame(current_frame, height=200)
        self.recommendations_frame.pack(fill="x", padx=10, pady=10)
        
        # Action buttons
        actions_frame = ctk.CTkFrame(recommendations_tab)
        actions_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            actions_frame,
            text="Quick Actions",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        buttons_frame = ctk.CTkFrame(actions_frame)
        buttons_frame.pack(fill="x", padx=10, pady=10)
        
        # Action buttons
        ctk.CTkButton(
            buttons_frame,
            text="🛑 Emergency Break",
            command=self.trigger_emergency_break,
            fg_color="red",
            hover_color="darkred",
            width=150
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            buttons_frame,
            text="⏸️ Safety Break",
            command=self.trigger_safety_break,
            fg_color="orange",
            hover_color="darkorange",
            width=150
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            buttons_frame,
            text="🔄 Break Patterns",
            command=self.break_patterns,
            width=150
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            buttons_frame,
            text="🎭 Rotate Fingerprint",
            command=self.rotate_fingerprint,
            width=150
        ).pack(side="left", padx=5)
        
        # Prediction section
        prediction_frame = ctk.CTkFrame(recommendations_tab)
        prediction_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            prediction_frame,
            text="Risk Predictions",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        self.predictions_text = ctk.CTkTextbox(prediction_frame)
        self.predictions_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_configuration_tab(self):
        """Create safety configuration tab"""
        config_tab = self.content_notebook.add("⚙️ Configuration")
        
        # Anti-detection settings
        anti_detection_frame = ctk.CTkFrame(config_tab)
        anti_detection_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            anti_detection_frame,
            text="Anti-Detection Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        settings_grid = ctk.CTkFrame(anti_detection_frame)
        settings_grid.pack(fill="x", padx=10, pady=10)
        
        # Obfuscation level
        ctk.CTkLabel(settings_grid, text="Obfuscation Level:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.obfuscation_slider = ctk.CTkSlider(settings_grid, from_=0.1, to=1.0, number_of_steps=18)
        self.obfuscation_slider.set(0.5)
        self.obfuscation_slider.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.obfuscation_value = ctk.CTkLabel(settings_grid, text="0.5")
        self.obfuscation_value.grid(row=0, column=2, padx=10, pady=5)
        
        # Risk thresholds
        ctk.CTkLabel(settings_grid, text="Risk Threshold:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.risk_threshold_slider = ctk.CTkSlider(settings_grid, from_=0.1, to=1.0, number_of_steps=18)
        self.risk_threshold_slider.set(0.7)
        self.risk_threshold_slider.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.risk_threshold_value = ctk.CTkLabel(settings_grid, text="0.7")
        self.risk_threshold_value.grid(row=1, column=2, padx=10, pady=5)
        
        # Break frequency
        ctk.CTkLabel(settings_grid, text="Break Frequency:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.break_frequency_slider = ctk.CTkSlider(settings_grid, from_=5, to=60, number_of_steps=11)
        self.break_frequency_slider.set(15)
        self.break_frequency_slider.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        self.break_frequency_value = ctk.CTkLabel(settings_grid, text="15 min")
        self.break_frequency_value.grid(row=2, column=2, padx=10, pady=5)
        
        # Configure grid
        settings_grid.grid_columnconfigure(1, weight=1)
        
        # Behavioral settings
        behavior_frame = ctk.CTkFrame(config_tab)
        behavior_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            behavior_frame,
            text="Behavioral Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        behavior_grid = ctk.CTkFrame(behavior_frame)
        behavior_grid.pack(fill="x", padx=10, pady=10)
        
        # Behavior checkboxes
        self.humanize_mouse = ctk.CTkCheckBox(behavior_grid, text="Humanize Mouse Movement")
        self.humanize_mouse.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.humanize_mouse.select()
        
        self.randomize_timing = ctk.CTkCheckBox(behavior_grid, text="Randomize Action Timing")
        self.randomize_timing.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.randomize_timing.select()
        
        self.break_patterns = ctk.CTkCheckBox(behavior_grid, text="Automatic Pattern Breaking")
        self.break_patterns.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.break_patterns.select()
        
        self.rotate_fingerprints = ctk.CTkCheckBox(behavior_grid, text="Automatic Fingerprint Rotation")
        self.rotate_fingerprints.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.rotate_fingerprints.select()
        
        # Apply button
        ctk.CTkButton(
            config_tab,
            text="Apply Settings",
            command=self.apply_settings,
            width=200,
            height=40
        ).pack(pady=20)
    
    def create_statistics_tab(self):
        """Create statistics tab"""
        stats_tab = self.content_notebook.add("📊 Statistics")
        
        # Statistics overview
        overview_frame = ctk.CTkFrame(stats_tab)
        overview_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            overview_frame,
            text="Safety System Statistics",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        # Stats grid
        stats_grid = ctk.CTkFrame(overview_frame)
        stats_grid.pack(fill="x", padx=10, pady=10)
        
        # Create stat labels
        self.stats_labels = {}
        stat_names = [
            "Actions Taken", "Risks Mitigated", "Patterns Broken",
            "Fingerprints Rotated", "Safety Breaks", "Detection Events"
        ]
        
        for i, stat_name in enumerate(stat_names):
            row = i // 2
            col = i % 2
            
            frame = ctk.CTkFrame(stats_grid)
            frame.grid(row=row, column=col, padx=10, pady=5, sticky="ew")
            
            ctk.CTkLabel(frame, text=stat_name, font=ctk.CTkFont(weight="bold")).pack(pady=2)
            label = ctk.CTkLabel(frame, text="0", font=ctk.CTkFont(size=18))
            label.pack(pady=2)
            
            self.stats_labels[stat_name.lower().replace(" ", "_")] = label
        
        # Configure grid
        stats_grid.grid_columnconfigure(0, weight=1)
        stats_grid.grid_columnconfigure(1, weight=1)
        
        # Performance metrics
        performance_frame = ctk.CTkFrame(stats_tab)
        performance_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            performance_frame,
            text="Performance Metrics",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        self.performance_text = ctk.CTkTextbox(performance_frame)
        self.performance_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_control_panel(self):
        """Create control panel"""
        control_frame = ctk.CTkFrame(self.main_frame)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        # Monitoring controls
        ctk.CTkLabel(control_frame, text="Monitoring:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)
        
        self.monitoring_button = ctk.CTkButton(
            control_frame,
            text="🟢 Monitoring Active",
            command=self.toggle_monitoring,
            width=150
        )
        self.monitoring_button.pack(side="left", padx=5)
        
        # Update interval
        ctk.CTkLabel(control_frame, text="Update Interval:").pack(side="left", padx=10)
        self.interval_slider = ctk.CTkSlider(control_frame, from_=0.5, to=5.0, number_of_steps=9)
        self.interval_slider.set(1.0)
        self.interval_slider.pack(side="left", padx=5)
        
        self.interval_label = ctk.CTkLabel(control_frame, text="1.0s")
        self.interval_label.pack(side="left", padx=5)
        
        # Auto actions
        self.auto_actions_switch = ctk.CTkSwitch(control_frame, text="Auto Actions")
        self.auto_actions_switch.pack(side="right", padx=10)
        self.auto_actions_switch.select()
    
    def create_status_bar(self):
        """Create status bar"""
        status_frame = ctk.CTkFrame(self.main_frame)
        status_frame.pack(fill="x", padx=10, pady=5)
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Advanced Safety System Ready",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="left", padx=10, pady=5)
        
        self.last_update_label = ctk.CTkLabel(
            status_frame,
            text="Last Update: Never",
            font=ctk.CTkFont(size=12)
        )
        self.last_update_label.pack(side="right", padx=10, pady=5)
    
    def setup_charts(self):
        """Setup chart styling and initial data"""
        # Initialize data arrays
        self.risk_data = [0] * self.max_data_points
        self.behavior_data = [0] * self.max_data_points
        self.time_data = list(range(-self.max_data_points, 0))
        
        # Setup risk chart
        self.risk_line, = self.risk_ax.plot(self.time_data, self.risk_data, 'r-', linewidth=2, label='Overall Risk')
        self.risk_ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
        self.risk_ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
        self.risk_ax.legend()
        
        # Setup behavior charts
        self.behavior_ax1.hist([0.5], bins=20, alpha=0.7, color='blue')
        self.behavior_ax2.hist([0.3], bins=20, alpha=0.7, color='green')
    
    def start_monitoring(self):
        """Start the monitoring system"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.anti_detection.start_monitoring()
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.update_display()
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_active = False
        self.anti_detection.stop_monitoring()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
    
    def toggle_monitoring(self):
        """Toggle monitoring on/off"""
        if self.monitoring_active:
            self.stop_monitoring()
            self.monitoring_button.configure(text="🔴 Monitoring Stopped")
            self.status_label.configure(text="Monitoring Stopped")
        else:
            self.start_monitoring()
            self.monitoring_button.configure(text="🟢 Monitoring Active")
            self.status_label.configure(text="Monitoring Active")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Update data
                self._update_monitoring_data()
                time.sleep(self.interval_slider.get())
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def _update_monitoring_data(self):
        """Update monitoring data"""
        try:
            # Get current risk assessment
            risk_assessment = self.anti_detection.assess_current_risk()
            
            # Update data arrays
            self.risk_data.append(risk_assessment['overall_risk'])
            self.risk_data.pop(0)
            
            # Update behavior data
            behavior_profile = self.behavior_model.update_behavior_profile()
            if behavior_profile:
                self.behavior_data.append(behavior_profile.consistency_score)
                self.behavior_data.pop(0)
            
        except Exception as e:
            logger.error(f"Error updating monitoring data: {e}")
    
    def update_display(self):
        """Update the GUI display"""
        try:
            # Get current data
            risk_assessment = self.anti_detection.assess_current_risk()
            behavior_profile = self.behavior_model.update_behavior_profile()
            
            # Update header stats
            self._update_header_stats(risk_assessment)
            
            # Update risk dashboard
            self._update_risk_dashboard(risk_assessment)
            
            # Update behavior analysis
            self._update_behavior_analysis(behavior_profile)
            
            # Update recommendations
            self._update_recommendations()
            
            # Update statistics
            self._update_statistics()
            
            # Update status
            self.last_update_label.configure(text=f"Last Update: {time.strftime('%H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"Error updating display: {e}")
        
        # Schedule next update
        if self.monitoring_active:
            self.parent.after(int(self.update_interval), self.update_display)
    
    def _update_header_stats(self, risk_assessment: Dict[str, float]):
        """Update header statistics"""
        overall_risk = risk_assessment['overall_risk']
        
        # Risk level
        if overall_risk < 0.3:
            risk_text = "Risk: LOW"
            risk_color = "green"
        elif overall_risk < 0.6:
            risk_text = "Risk: MEDIUM"
            risk_color = "orange"
        elif overall_risk < 0.8:
            risk_text = "Risk: HIGH"
            risk_color = "red"
        else:
            risk_text = "Risk: CRITICAL"
            risk_color = "darkred"
        
        self.risk_level_label.configure(text=risk_text, text_color=risk_color)
        
        # Behavior score
        behavior_profile = self.behavior_model.current_profile
        if behavior_profile:
            consistency = behavior_profile.consistency_score
            if consistency > 0.7:
                behavior_text = "Behavior: HUMAN-LIKE"
                behavior_color = "blue"
            elif consistency > 0.4:
                behavior_text = "Behavior: MODERATE"
                behavior_color = "orange"
            else:
                behavior_text = "Behavior: ROBOTIC"
                behavior_color = "red"
        else:
            behavior_text = "Behavior: UNKNOWN"
            behavior_color = "gray"
        
        self.behavior_score_label.configure(text=behavior_text, text_color=behavior_color)
        
        # Safety status
        if overall_risk < 0.5:
            status_text = "Status: PROTECTED"
            status_color = "green"
        elif overall_risk < 0.8:
            status_text = "Status: CAUTION"
            status_color = "orange"
        else:
            status_text = "Status: DANGER"
            status_color = "red"
        
        self.safety_status_label.configure(text=status_text, text_color=status_color)
    
    def _update_risk_dashboard(self, risk_assessment: Dict[str, float]):
        """Update risk dashboard"""
        # Update progress bars
        self.overall_risk_progress.set(risk_assessment['overall_risk'])
        self.overall_risk_value.configure(text=f"{risk_assessment['overall_risk']:.1%}")
        
        self.detection_risk_progress.set(risk_assessment['detection_risk'])
        self.detection_risk_value.configure(text=f"{risk_assessment['detection_risk']:.1%}")
        
        self.pattern_risk_progress.set(risk_assessment['pattern_risk'])
        self.pattern_risk_value.configure(text=f"{risk_assessment['pattern_risk']:.1%}")
        
        # Update risk chart
        self.risk_line.set_ydata(self.risk_data)
        self.risk_canvas.draw_idle()
    
    def _update_behavior_analysis(self, behavior_profile):
        """Update behavior analysis"""
        if not behavior_profile:
            return
        
        # Update text displays
        mouse_text = f"""Mouse Speed: {behavior_profile.mouse_speed_avg:.1f} ± {behavior_profile.mouse_speed_std:.1f}
Consistency: {behavior_profile.consistency_score:.2f}
Break Frequency: {behavior_profile.break_frequency:.1f} breaks/hour
Attention Span: {behavior_profile.attention_span:.1f} minutes"""
        
        timing_text = f"""Reaction Time: {behavior_profile.reaction_time_avg:.3f}s ± {behavior_profile.reaction_time_std:.3f}s
Multitasking: {behavior_profile.multitasking_tendency:.2f}
Risk Tolerance: {behavior_profile.risk_tolerance:.2f}
Profile Age: {(time.time() - behavior_profile.created_at.timestamp())/60:.1f} minutes"""
        
        self.mouse_behavior_text.delete("1.0", "end")
        self.mouse_behavior_text.insert("1.0", mouse_text)
        
        self.timing_behavior_text.delete("1.0", "end")
        self.timing_behavior_text.insert("1.0", timing_text)
    
    def _update_recommendations(self):
        """Update safety recommendations"""
        recommendations = self.anti_detection.get_safety_recommendations()
        
        # Clear existing recommendations
        for widget in self.recommendations_frame.winfo_children():
            widget.destroy()
        
        # Add current recommendations
        if not recommendations:
            ctk.CTkLabel(
                self.recommendations_frame,
                text="✅ No immediate safety concerns detected",
                text_color="green"
            ).pack(pady=5)
        else:
            for rec in recommendations:
                self._create_recommendation_widget(rec)
    
    def _create_recommendation_widget(self, recommendation):
        """Create a widget for a safety recommendation"""
        rec_frame = ctk.CTkFrame(self.recommendations_frame)
        rec_frame.pack(fill="x", padx=5, pady=5)
        
        # Urgency indicator
        urgency_color = "green" if recommendation.urgency < 0.3 else "orange" if recommendation.urgency < 0.7 else "red"
        urgency_text = "🟢" if recommendation.urgency < 0.3 else "🟡" if recommendation.urgency < 0.7 else "🔴"
        
        ctk.CTkLabel(rec_frame, text=urgency_text).pack(side="left", padx=5)
        
        # Description
        ctk.CTkLabel(
            rec_frame,
            text=recommendation.description,
            wraplength=400
        ).pack(side="left", padx=5, fill="x", expand=True)
        
        # Action button if required
        if recommendation.action_required:
            ctk.CTkButton(
                rec_frame,
                text="Take Action",
                command=lambda: self._handle_recommendation_action(recommendation),
                width=100
            ).pack(side="right", padx=5)
    
    def _handle_recommendation_action(self, recommendation):
        """Handle recommendation action"""
        if recommendation.recommendation_type == "immediate_break":
            self.trigger_emergency_break()
        elif recommendation.recommendation_type == "break_patterns":
            self.break_patterns()
        elif recommendation.recommendation_type == "vary_behavior":
            # Increase behavioral variation
            pass
        elif recommendation.recommendation_type == "humanize_timing":
            # Adjust timing settings
            pass
    
    def _update_statistics(self):
        """Update statistics display"""
        stats = self.anti_detection.get_statistics()
        
        # Update stat labels
        for key, value in stats.items():
            if key in self.stats_labels:
                self.stats_labels[key].configure(text=str(value))
        
        # Update performance text
        performance_text = f"""System Performance:
- Monitoring Active: {stats.get('monitoring_active', False)}
- Current Risk: {stats.get('current_risk', {}).get('overall_risk', 0):.1%}
- Detection Events: {stats.get('detection_events_count', 0)}
- Recent Activity: {len(stats.get('recent_events', []))} events

GPU Status: {'Available' if hasattr(self.behavior_model, 'device') and 'cuda' in str(self.behavior_model.device) else 'CPU Only'}
Model Status: {'AI Models Active' if hasattr(self.behavior_model, 'movement_model') and self.behavior_model.movement_model else 'Fallback Mode'}"""
        
        self.performance_text.delete("1.0", "end")
        self.performance_text.insert("1.0", performance_text)
    
    def trigger_emergency_break(self):
        """Trigger emergency break"""
        try:
            self.anti_detection._trigger_emergency_break()
            messagebox.showinfo("Emergency Break", "Emergency break triggered successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to trigger emergency break: {e}")
    
    def trigger_safety_break(self):
        """Trigger safety break"""
        try:
            self.anti_detection._trigger_safety_break(120)  # 2 minutes
            messagebox.showinfo("Safety Break", "Safety break triggered for 2 minutes!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to trigger safety break: {e}")
    
    def break_patterns(self):
        """Break behavioral patterns"""
        try:
            self.anti_detection._break_current_patterns()
            messagebox.showinfo("Pattern Break", "Behavioral patterns have been broken!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to break patterns: {e}")
    
    def rotate_fingerprint(self):
        """Rotate behavioral fingerprint"""
        try:
            self.anti_detection.fingerprint_manager.rotate_fingerprint()
            messagebox.showinfo("Fingerprint Rotation", "Behavioral fingerprint rotated successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to rotate fingerprint: {e}")
    
    def apply_settings(self):
        """Apply configuration settings"""
        try:
            # Update obfuscation level
            obfuscation_level = self.obfuscation_slider.get()
            self.anti_detection.behavioral_obfuscator.obfuscation_level = obfuscation_level
            self.obfuscation_value.configure(text=f"{obfuscation_level:.1f}")
            
            # Update risk threshold
            risk_threshold = self.risk_threshold_slider.get()
            self.anti_detection.risk_thresholds['medium'] = risk_threshold
            self.risk_threshold_value.configure(text=f"{risk_threshold:.1f}")
            
            # Update break frequency
            break_frequency = self.break_frequency_slider.get()
            self.anti_detection.pattern_breaker.break_interval = break_frequency * 60  # Convert to seconds
            self.break_frequency_value.configure(text=f"{break_frequency:.0f} min")
            
            messagebox.showinfo("Settings Applied", "Configuration settings have been applied!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply settings: {e}")
    
    def cleanup(self):
        """Cleanup when tab is closed"""
        self.stop_monitoring()


def create_advanced_safety_tab(parent):
    """Create and return the Advanced Safety tab"""
    return AdvancedSafetyTab(parent)