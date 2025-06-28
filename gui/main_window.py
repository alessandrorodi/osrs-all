#!/usr/bin/env python3
"""
OSRS Bot Framework - Main GUI Application

Modern, comprehensive GUI for managing and monitoring bots.
"""

import sys
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import customtkinter as ctk
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import numpy as np
except ImportError as e:
    print(f"Missing GUI dependencies: {e}")
    print("Please install: pip install --user customtkinter pillow matplotlib")
    sys.exit(1)

from core.screen_capture import screen_capture
from core.computer_vision import cv_system
from core.bot_base import BotBase, BotState
from utils.logging import setup_logging, BotLogger
from config.settings import *

# Import Phase 2 AI Vision System
try:
    from vision.intelligent_vision import intelligent_vision, GameState, SceneType
    AI_VISION_AVAILABLE = True
except ImportError as e:
    print(f"AI Vision system not available: {e}")
    AI_VISION_AVAILABLE = False

# Set theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

logger = setup_logging("gui")


class OSRSBotGUI:
    """Main GUI Application for OSRS Bot Framework"""
    
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("OSRS Bot Framework - Ultimate AI Agent")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Application state
        self.active_bots: Dict[str, BotBase] = {}
        self.bot_threads: Dict[str, threading.Thread] = {}
        self.performance_data: Dict[str, List] = {}
        self.is_monitoring = False
        
        # GUI components
        self.setup_gui()
        self.setup_monitoring()
        
        logger.info("OSRS Bot Framework GUI initialized")
    
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Main container
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create header
        self.create_header()
        
        # Create main content area
        self.create_main_content()
        
        # Create status bar
        self.create_status_bar()
    
    def create_header(self):
        """Create the header with title and quick actions"""
        header_frame = ctk.CTkFrame(self.main_frame)
        header_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="🤖 OSRS Ultimate AI Agent",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(side="left", padx=20, pady=15)
        
        # Quick action buttons
        actions_frame = ctk.CTkFrame(header_frame)
        actions_frame.pack(side="right", padx=20, pady=10)
        
        ctk.CTkButton(
            actions_frame,
            text="🎯 Calibrate Client",
            command=self.calibrate_client,
            width=120
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            actions_frame,
            text="📸 Create Templates",
            command=self.create_templates,
            width=120
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            actions_frame,
            text="⚙️ Settings",
            command=self.open_settings,
            width=100
        ).pack(side="left", padx=5)
    
    def create_main_content(self):
        """Create the main content area with tabs"""
        # Create notebook for tabs
        self.notebook = ctk.CTkTabview(self.main_frame)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_bots_tab()
        self.create_vision_tab()
        self.create_navigation_tab()
        self.create_performance_tab()
        self.create_logs_tab()
        self.create_templates_tab()
    
    def create_dashboard_tab(self):
        """Create the main dashboard tab"""
        dashboard = self.notebook.add("🏠 Dashboard")
        
        # Top section - System Status
        status_frame = ctk.CTkFrame(dashboard)
        status_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            status_frame,
            text="System Status",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=10)
        
        # Status grid
        status_grid = ctk.CTkFrame(status_frame)
        status_grid.pack(fill="x", padx=10, pady=10)
        
        # System status indicators
        self.client_status = ctk.CTkLabel(status_grid, text="🔴 Client: Not Connected")
        self.client_status.grid(row=0, column=0, padx=20, pady=5, sticky="w")
        
        self.capture_status = ctk.CTkLabel(status_grid, text="🔴 Screen Capture: Not Active")
        self.capture_status.grid(row=0, column=1, padx=20, pady=5, sticky="w")
        
        self.vision_status = ctk.CTkLabel(status_grid, text="🟡 Computer Vision: Ready")
        self.vision_status.grid(row=1, column=0, padx=20, pady=5, sticky="w")
        
        self.bot_count = ctk.CTkLabel(status_grid, text="🤖 Active Bots: 0")
        self.bot_count.grid(row=1, column=1, padx=20, pady=5, sticky="w")
        
        # Quick stats
        stats_frame = ctk.CTkFrame(dashboard)
        stats_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            stats_frame,
            text="Quick Statistics",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=10)
        
        # Stats display
        self.stats_display = ctk.CTkTextbox(stats_frame, height=200)
        self.stats_display.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Emergency stop button
        emergency_frame = ctk.CTkFrame(dashboard)
        emergency_frame.pack(fill="x", padx=10, pady=10)
        
        self.emergency_button = ctk.CTkButton(
            emergency_frame,
            text="🛑 EMERGENCY STOP ALL BOTS",
            command=self.emergency_stop_all,
            fg_color="red",
            hover_color="darkred",
            height=50,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.emergency_button.pack(pady=10)
    
    def create_bots_tab(self):
        """Create the bot management tab"""
        bots_tab = self.notebook.add("🤖 Bots")
        
        # Bot list and controls
        controls_frame = ctk.CTkFrame(bots_tab)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(
            controls_frame,
            text="➕ New Bot",
            command=self.create_new_bot
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            controls_frame,
            text="📂 Load Bot",
            command=self.load_bot
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            controls_frame,
            text="🔄 Refresh",
            command=self.refresh_bots
        ).pack(side="left", padx=5, pady=10)
        
        # Bot list
        list_frame = ctk.CTkFrame(bots_tab)
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Treeview for bot list
        columns = ("Name", "Status", "Runtime", "Actions", "Errors", "Performance")
        self.bot_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            self.bot_tree.heading(col, text=col)
            self.bot_tree.column(col, width=120)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.bot_tree.yview)
        self.bot_tree.configure(yscrollcommand=scrollbar.set)
        
        self.bot_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bot control buttons
        bot_controls = ctk.CTkFrame(bots_tab)
        bot_controls.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(
            bot_controls,
            text="▶️ Start",
            command=self.start_selected_bot
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            bot_controls,
            text="⏸️ Pause",
            command=self.pause_selected_bot
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            bot_controls,
            text="⏹️ Stop",
            command=self.stop_selected_bot
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            bot_controls,
            text="📊 Details",
            command=self.show_bot_details
        ).pack(side="left", padx=5)
    
    def create_vision_tab(self):
        """Create the computer vision tab with AI Vision capabilities"""
        vision_tab = self.notebook.add("👁️ AI Vision")
        
        # Vision mode selector
        mode_frame = ctk.CTkFrame(vision_tab)
        mode_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(mode_frame, text="Vision Mode:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)
        
        self.vision_mode = ctk.CTkOptionMenu(
            mode_frame,
            values=["Classic CV", "AI Vision (Phase 2)", "Combined"],
            command=self.on_vision_mode_change
        )
        self.vision_mode.pack(side="left", padx=5)
        
        if AI_VISION_AVAILABLE:
            self.vision_mode.set("AI Vision (Phase 2)")
        else:
            self.vision_mode.set("Classic CV")
        
        # Vision controls
        controls_frame = ctk.CTkFrame(vision_tab)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(
            controls_frame,
            text="📷 Capture & Analyze",
            command=self.capture_and_analyze
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            controls_frame,
            text="🔍 Test Detection",
            command=self.test_detection
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            controls_frame,
            text="🎯 Live Analysis",
            command=self.toggle_live_analysis
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            controls_frame,
            text="🧠 Game State",
            command=self.show_game_state
        ).pack(side="left", padx=5, pady=10)
        
        # Main content area with paned window
        content_paned = ctk.CTkFrame(vision_tab)
        content_paned.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left side - Vision display
        display_frame = ctk.CTkFrame(content_paned)
        display_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        ctk.CTkLabel(display_frame, text="Vision Analysis", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        # Canvas for image display
        self.vision_canvas = tk.Canvas(display_frame, bg="black")
        self.vision_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Right side - Analysis results
        results_frame = ctk.CTkFrame(content_paned)
        results_frame.pack(side="right", fill="y", padx=(5, 0), sticky="ns")
        results_frame.configure(width=400)
        
        ctk.CTkLabel(results_frame, text="Analysis Results", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        # Game state display
        self.game_state_display = ctk.CTkTextbox(results_frame, height=200, width=380)
        self.game_state_display.pack(padx=10, pady=5)
        
        # Detection statistics
        stats_label = ctk.CTkLabel(results_frame, text="Detection Statistics", font=ctk.CTkFont(size=14, weight="bold"))
        stats_label.pack(pady=(10, 5))
        
        self.detection_stats = ctk.CTkTextbox(results_frame, height=150, width=380)
        self.detection_stats.pack(padx=10, pady=5)
        
        # AI Vision settings (if available)
        if AI_VISION_AVAILABLE:
            settings_label = ctk.CTkLabel(results_frame, text="AI Vision Settings", font=ctk.CTkFont(size=14, weight="bold"))
            settings_label.pack(pady=(10, 5))
            
            settings_frame = ctk.CTkFrame(results_frame)
            settings_frame.pack(fill="x", padx=10, pady=5)
            
            # Confidence threshold
            ctk.CTkLabel(settings_frame, text="Confidence:").pack(anchor="w", padx=5)
            self.confidence_slider = ctk.CTkSlider(settings_frame, from_=0.1, to=1.0, number_of_steps=18)
            self.confidence_slider.set(0.5)
            self.confidence_slider.pack(fill="x", padx=5, pady=2)
            
            # Detection types
            ctk.CTkLabel(settings_frame, text="Detection Types:").pack(anchor="w", padx=5, pady=(10, 5))
            
            self.detect_npcs = ctk.CTkCheckBox(settings_frame, text="NPCs")
            self.detect_npcs.pack(anchor="w", padx=10)
            self.detect_npcs.select()
            
            self.detect_items = ctk.CTkCheckBox(settings_frame, text="Items")
            self.detect_items.pack(anchor="w", padx=10)
            self.detect_items.select()
            
            self.detect_players = ctk.CTkCheckBox(settings_frame, text="Players")
            self.detect_players.pack(anchor="w", padx=10)
            self.detect_players.select()
            
            self.detect_ui = ctk.CTkCheckBox(settings_frame, text="UI Elements")
            self.detect_ui.pack(anchor="w", padx=10)
            self.detect_ui.select()
            
            # OCR settings
            self.enable_ocr = ctk.CTkCheckBox(settings_frame, text="Enable OCR")
            self.enable_ocr.pack(anchor="w", padx=10, pady=(10, 5))
            self.enable_ocr.select()
        
        # Performance info
        perf_frame = ctk.CTkFrame(vision_tab)
        perf_frame.pack(fill="x", padx=10, pady=10)
        
        self.vision_performance = ctk.CTkLabel(perf_frame, text="Performance: Ready")
        self.vision_performance.pack(pady=5)
    
    def create_navigation_tab(self):
        """Create the navigation tab with minimap analysis and pathfinding"""
        nav_tab = self.notebook.add("🧭 Navigation")
        
        # Try to import navigation components
        try:
            # Import required for navigation fallback methods
            from gui.widgets.navigation_panel import NavigationPanel
            
            # Create navigation panel
            self.navigation_panel = NavigationPanel(nav_tab, width=780, height=580)
            self.navigation_panel.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Store reference for cleanup
            if not hasattr(self, 'navigation_components'):
                self.navigation_components = []
            self.navigation_components.append(self.navigation_panel)
            
        except ImportError as e:
            # Fallback UI if navigation components not available
            error_frame = ctk.CTkFrame(nav_tab)
            error_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            ctk.CTkLabel(
                error_frame,
                text="🧭 Navigation System",
                font=ctk.CTkFont(size=24, weight="bold")
            ).pack(pady=20)
            
            ctk.CTkLabel(
                error_frame,
                text="Navigation components not available",
                font=ctk.CTkFont(size=16)
            ).pack(pady=10)
            
            ctk.CTkLabel(
                error_frame,
                text="Please install required dependencies:",
                font=ctk.CTkFont(size=14)
            ).pack(pady=5)
            
            deps_text = """
Required dependencies:
• opencv-python>=4.8.0
• numpy>=1.24.0
• customtkinter>=5.0.0
• PIL (Pillow)

Features when available:
• Live minimap analysis with YOLOv8 integration
• A* pathfinding with obstacle avoidance  
• Multi-floor navigation (stairs, ladders, teleports)
• Real-time processing at 60fps (RTX 4090 optimized)
• Danger zone detection and safe route planning
• Interactive minimap with path visualization
• Movement efficiency metrics and debug tools
            """
            
            deps_display = ctk.CTkTextbox(error_frame, height=300, width=700)
            deps_display.pack(pady=20, padx=20)
            deps_display.insert("1.0", deps_text.strip())
            deps_display.configure(state="disabled")
            
            # Manual controls as fallback
            manual_frame = ctk.CTkFrame(error_frame)
            manual_frame.pack(fill="x", padx=20, pady=10)
            
            ctk.CTkLabel(
                manual_frame,
                text="Manual Navigation Tools",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(pady=5)
            
            tools_frame = ctk.CTkFrame(manual_frame)
            tools_frame.pack(fill="x", pady=10)
            
            ctk.CTkButton(
                tools_frame,
                text="📷 Capture Minimap",
                command=self.capture_minimap_manual
            ).pack(side="left", padx=5)
            
            ctk.CTkButton(
                tools_frame,
                text="🗺️ Analyze Image",
                command=self.analyze_image_manual
            ).pack(side="left", padx=5)
            
            ctk.CTkButton(
                tools_frame,
                text="📊 Show Stats",
                command=self.show_navigation_stats
            ).pack(side="left", padx=5)
    
    def create_performance_tab(self):
        """Create the performance monitoring tab"""
        perf_tab = self.notebook.add("📊 Performance")
        
        # Performance controls
        controls_frame = ctk.CTkFrame(perf_tab)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        self.monitoring_button = ctk.CTkButton(
            controls_frame,
            text="▶️ Start Monitoring",
            command=self.toggle_monitoring
        )
        self.monitoring_button.pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            controls_frame,
            text="🔄 Refresh Charts",
            command=self.refresh_charts
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            controls_frame,
            text="💾 Export Data",
            command=self.export_performance_data
        ).pack(side="left", padx=5, pady=10)
        
        # Charts frame
        charts_frame = ctk.CTkFrame(perf_tab)
        charts_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.perf_figure, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.perf_figure.patch.set_facecolor('#2b2b2b')
        
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
        
        self.perf_canvas = FigureCanvasTkAgg(self.perf_figure, charts_frame)
        self.perf_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def create_logs_tab(self):
        """Create the logs viewing tab"""
        logs_tab = self.notebook.add("📋 Logs")
        
        # Log controls
        controls_frame = ctk.CTkFrame(logs_tab)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(
            controls_frame,
            text="🔄 Refresh",
            command=self.refresh_logs
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            controls_frame,
            text="🗑️ Clear",
            command=self.clear_logs
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            controls_frame,
            text="💾 Export",
            command=self.export_logs
        ).pack(side="left", padx=5, pady=10)
        
        # Log level filter
        ctk.CTkLabel(controls_frame, text="Level:").pack(side="left", padx=10)
        self.log_level = ctk.CTkOptionMenu(
            controls_frame,
            values=["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            command=self.filter_logs
        )
        self.log_level.pack(side="left", padx=5)
        
        # Log display
        log_frame = ctk.CTkFrame(logs_tab)
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.log_display = ctk.CTkTextbox(log_frame)
        self.log_display.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_templates_tab(self):
        """Create the template management tab"""
        templates_tab = self.notebook.add("🖼️ Templates")
        
        # Template controls
        controls_frame = ctk.CTkFrame(templates_tab)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(
            controls_frame,
            text="➕ Create Template",
            command=self.create_template
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            controls_frame,
            text="📂 Import Template",
            command=self.import_template
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            controls_frame,
            text="🔄 Refresh",
            command=self.refresh_templates
        ).pack(side="left", padx=5, pady=10)
        
        # Template list and preview
        main_template_frame = ctk.CTkFrame(templates_tab)
        main_template_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Template list
        list_frame = ctk.CTkFrame(main_template_frame)
        list_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        ctk.CTkLabel(list_frame, text="Templates", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.template_listbox = tk.Listbox(list_frame, width=30, height=20)
        self.template_listbox.pack(fill="both", expand=True, padx=10, pady=10)
        self.template_listbox.bind('<<ListboxSelect>>', self.on_template_select)
        
        # Template preview
        preview_frame = ctk.CTkFrame(main_template_frame)
        preview_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(preview_frame, text="Preview", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.template_canvas = tk.Canvas(preview_frame, bg="black", width=400, height=300)
        self.template_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Template info
        info_frame = ctk.CTkFrame(preview_frame)
        info_frame.pack(fill="x", padx=10, pady=10)
        
        self.template_info = ctk.CTkTextbox(info_frame, height=100)
        self.template_info.pack(fill="x", padx=10, pady=10)
    
    def create_status_bar(self):
        """Create the status bar"""
        self.status_frame = ctk.CTkFrame(self.main_frame)
        self.status_frame.pack(fill="x", padx=10, pady=(5, 10))
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready - OSRS Bot Framework Loaded",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="left", padx=10, pady=5)
        
        # System info
        self.system_info = ctk.CTkLabel(
            self.status_frame,
            text=f"Python {sys.version_info.major}.{sys.version_info.minor} | CV2 Available | Framework v1.0",
            font=ctk.CTkFont(size=10)
        )
        self.system_info.pack(side="right", padx=10, pady=5)
    
    def setup_monitoring(self):
        """Setup background monitoring"""
        self.update_system_status()
        self.root.after(1000, self.monitoring_loop)  # Update every second
    
    def monitoring_loop(self):
        """Main monitoring loop"""
        if self.is_monitoring:
            self.update_bot_list()
            self.update_performance_charts()
        
        self.update_system_status()
        self.root.after(1000, self.monitoring_loop)
    
    def update_system_status(self):
        """Update system status indicators"""
        try:
            # Check client status
            if screen_capture.find_client_window():
                self.client_status.configure(text="🟢 Client: Connected")
            else:
                self.client_status.configure(text="🔴 Client: Not Found")
            
            # Check screen capture
            test_image = screen_capture.capture_screen()
            if test_image is not None:
                self.capture_status.configure(text="🟢 Screen Capture: Active")
            else:
                self.capture_status.configure(text="🔴 Screen Capture: Failed")
            
            # Update bot count
            active_count = len([bot for bot in self.active_bots.values() if bot.state == BotState.RUNNING])
            total_count = len(self.active_bots)
            self.bot_count.configure(text=f"🤖 Active Bots: {active_count}/{total_count}")
            
        except Exception as e:
            logger.error(f"Error updating system status: {e}")
    
    def update_bot_list(self):
        """Update the bot list display"""
        # Clear existing items
        for item in self.bot_tree.get_children():
            self.bot_tree.delete(item)
        
        # Add current bots
        for name, bot in self.active_bots.items():
            try:
                status = bot.get_status()
                self.bot_tree.insert("", "end", values=(
                    name,
                    status["state"],
                    f"{status['runtime']:.1f}s",
                    status["actions"],
                    status["errors"],
                    f"{status['performance']['actions_per_minute']:.1f} APM"
                ))
            except Exception as e:
                logger.error(f"Error updating bot {name}: {e}")
    
    def update_performance_charts(self):
        """Update performance monitoring charts"""
        try:
            if not self.active_bots:
                return
            
            # Clear axes
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.clear()
                ax.set_facecolor('#2b2b2b')
            
            # Chart 1: Actions per minute
            bot_names = list(self.active_bots.keys())
            apms = [bot.get_status()["performance"]["actions_per_minute"] 
                   for bot in self.active_bots.values()]
            
            self.ax1.bar(bot_names, apms, color='skyblue')
            self.ax1.set_title('Actions Per Minute', color='white')
            self.ax1.set_ylabel('APM', color='white')
            
            # Chart 2: Success rate
            success_rates = [bot.get_status()["performance"]["success_rate"] 
                           for bot in self.active_bots.values()]
            
            self.ax2.bar(bot_names, success_rates, color='lightgreen')
            self.ax2.set_title('Success Rate %', color='white')
            self.ax2.set_ylabel('Success %', color='white')
            
            # Chart 3: Runtime
            runtimes = [bot.get_status()["runtime"] for bot in self.active_bots.values()]
            
            self.ax3.bar(bot_names, runtimes, color='orange')
            self.ax3.set_title('Runtime (seconds)', color='white')
            self.ax3.set_ylabel('Seconds', color='white')
            
            # Chart 4: Error rates
            error_rates = [bot.get_status()["performance"]["errors_per_hour"] 
                          for bot in self.active_bots.values()]
            
            self.ax4.bar(bot_names, error_rates, color='salmon')
            self.ax4.set_title('Errors Per Hour', color='white')
            self.ax4.set_ylabel('Errors/Hour', color='white')
            
            self.perf_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating performance charts: {e}")
    
    # Event handlers and utility methods
    def calibrate_client(self):
        """Launch client calibration"""
        threading.Thread(target=self._run_calibration, daemon=True).start()
    
    def _run_calibration(self):
        """Run calibration in background thread"""
        try:
            import subprocess
            result = subprocess.run([sys.executable, "tools/calibrate_client.py"], 
                                  cwd=project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.update_status("✅ Client calibration successful!")
            else:
                self.update_status("❌ Client calibration failed!")
        except Exception as e:
            self.update_status(f"❌ Calibration error: {e}")
    
    def create_templates(self):
        """Launch template creator"""
        threading.Thread(target=self._run_template_creator, daemon=True).start()
    
    def _run_template_creator(self):
        """Run template creator in background thread"""
        try:
            import subprocess
            subprocess.Popen([sys.executable, "tools/template_creator.py"], cwd=project_root)
            self.update_status("🎯 Template creator launched!")
        except Exception as e:
            self.update_status(f"❌ Template creator error: {e}")
    
    def emergency_stop_all(self):
        """Emergency stop all bots"""
        try:
            for bot in self.active_bots.values():
                if bot.state == BotState.RUNNING:
                    bot.emergency_stop()
            self.update_status("🛑 EMERGENCY STOP - All bots stopped!")
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
    
    def update_status(self, message: str):
        """Update status bar message"""
        self.status_label.configure(text=message)
        logger.info(f"Status: {message}")
    
    def run(self):
        """Start the GUI application"""
        try:
            logger.info("Starting OSRS Bot Framework GUI...")
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("GUI shutdown requested")
        except Exception as e:
            logger.error(f"GUI error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up GUI resources...")
        # Stop all active bots
        for bot in self.active_bots.values():
            try:
                bot.stop()
            except:
                pass
    
    # AI Vision Methods (Phase 2)
    def on_vision_mode_change(self, value):
        """Handle vision mode change"""
        self.update_status(f"Vision mode changed to: {value}")
        if value == "AI Vision (Phase 2)" and not AI_VISION_AVAILABLE:
            messagebox.showwarning("AI Vision Unavailable", 
                                 "AI Vision system is not available. Please install required dependencies.")
            self.vision_mode.set("Classic CV")
    
    def capture_and_analyze(self):
        """Capture screen and run AI analysis"""
        try:
            # Capture screenshot
            image = screen_capture.capture_client()
            if image is None:
                self.update_status("❌ Failed to capture screen")
                return
            
            self.update_status("📷 Analyzing screenshot...")
            
            # Run analysis based on mode
            mode = self.vision_mode.get()
            if mode == "AI Vision (Phase 2)" and AI_VISION_AVAILABLE:
                # Use AI Vision system
                game_state = intelligent_vision.analyze_game_state(image)
                self._display_ai_results(image, game_state)
            else:
                # Use classic computer vision
                results = cv_system.process_image(image)
                self._display_classic_results(image, results)
            
        except Exception as e:
            self.update_status(f"❌ Analysis failed: {e}")
            logger.error(f"Analysis error: {e}")
    
    def toggle_live_analysis(self):
        """Toggle live analysis mode"""
        # This would implement real-time analysis
        self.update_status("🎯 Live analysis not yet implemented")
    
    def show_game_state(self):
        """Show detailed game state analysis"""
        if not AI_VISION_AVAILABLE:
            messagebox.showinfo("AI Vision Required", "This feature requires the AI Vision system.")
            return
        
        try:
            image = screen_capture.capture_client()
            if image is None:
                self.update_status("❌ Failed to capture screen")
                return
            
            # Analyze current game state
            game_state = intelligent_vision.analyze_game_state(image)
            
            # Create detailed game state window
            self._show_game_state_window(game_state)
            
        except Exception as e:
            self.update_status(f"❌ Game state analysis failed: {e}")
            logger.error(f"Game state error: {e}")
    
    def _display_ai_results(self, image, game_state: 'GameState'):
        """Display AI vision analysis results"""
        try:
            # Display image with annotations
            annotated_image = intelligent_vision.visualize_analysis(image, game_state)
            self._display_image_on_canvas(annotated_image, self.vision_canvas)
            
            # Update game state display
            state_text = f"""🧠 AI Vision Analysis Results
📅 Timestamp: {time.strftime('%H:%M:%S')}
🎯 Scene: {game_state.scene_type.value} ({game_state.confidence:.2f})
⚡ Processing Time: {game_state.processing_time:.3f}s

👤 Player Status:
  Health: {game_state.player_status.health_percent:.1f}%
  Prayer: {game_state.player_status.prayer_percent:.1f}%
  Energy: {game_state.player_status.energy_percent:.1f}%
  In Combat: {game_state.player_status.is_in_combat}

🗺️ Minimap:
  NPCs Visible: {len(game_state.minimap.visible_npcs or [])}
  Players Visible: {len(game_state.minimap.visible_players or [])}
  Region: {game_state.minimap.region_type}

🎒 Inventory:
  Free Slots: {game_state.inventory.free_slots}/28
  Items: {len(game_state.inventory.items or [])}
  Valuable Items: {len(game_state.inventory.valuable_items or [])}

💬 Interface:
  Open Interfaces: {len(game_state.interface_state.open_interfaces or [])}
  Chat Messages: {len(game_state.interface_state.active_chat or [])}
  Clickable Elements: {len(game_state.interface_state.clickable_elements or [])}
"""
            self.game_state_display.delete("0.0", "end")
            self.game_state_display.insert("0.0", state_text)
            
            # Update detection statistics
            stats_text = f"""📊 Detection Statistics
🎯 Objects Detected:
  NPCs: {len(game_state.npcs or [])}
  Items: {len(game_state.items or [])}
  Players: {len(game_state.players or [])}
  UI Elements: {len(game_state.ui_elements or [])}
  Environment: {len(game_state.environment or [])}

🏆 Top Priority Objects:
"""
            top_objects = game_state.get_highest_priority_objects(3)
            for i, obj in enumerate(top_objects, 1):
                stats_text += f"  {i}. {obj.label} ({obj.action_priority:.2f})\n"
            
            # Performance stats
            if hasattr(intelligent_vision, 'get_performance_stats'):
                perf_stats = intelligent_vision.get_performance_stats()
                stats_text += f"""
⚡ Performance:
  Frames Processed: {perf_stats.get('frames_processed', 0)}
  Avg Processing Time: {perf_stats.get('avg_processing_time', 0):.3f}s
  Last Analysis: {perf_stats.get('last_analysis_time', 0):.3f}s
"""
            
            self.detection_stats.delete("0.0", "end")
            self.detection_stats.insert("0.0", stats_text)
            
            # Update performance label
            self.vision_performance.configure(
                text=f"Performance: {game_state.processing_time:.3f}s | Objects: {len(game_state.npcs + game_state.items + game_state.players)}"
            )
            
            self.update_status(f"✅ AI analysis complete - {game_state.scene_type.value} scene detected")
            
        except Exception as e:
            logger.error(f"Error displaying AI results: {e}")
            self.update_status(f"❌ Display error: {e}")
    
    def _display_classic_results(self, image, results):
        """Display classic computer vision results"""
        try:
            # Display original image
            self._display_image_on_canvas(image, self.vision_canvas)
            
            # Update displays with classic results
            results_text = f"""🔍 Classic Computer Vision Results
📅 Timestamp: {time.strftime('%H:%M:%S')}
🖼️ Image Shape: {results.get('image_shape', 'Unknown')}

📊 Detections:
  Health Bars: {len(results.get('health_bars', []))}
  Features: {results.get('features', {}).get('keypoint_count', 0)} keypoints
"""
            
            self.game_state_display.delete("0.0", "end")
            self.game_state_display.insert("0.0", results_text)
            
            self.detection_stats.delete("0.0", "end")
            self.detection_stats.insert("0.0", "Classic CV mode - basic feature detection")
            
            self.update_status("✅ Classic CV analysis complete")
            
        except Exception as e:
            logger.error(f"Error displaying classic results: {e}")
            self.update_status(f"❌ Display error: {e}")
    
    def _display_image_on_canvas(self, cv_image, canvas):
        """Display OpenCV image on tkinter canvas"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # Resize to fit canvas
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                pil_image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Clear canvas and display image
            canvas.delete("all")
            canvas.create_image(canvas_width//2, canvas_height//2, image=photo, anchor="center")
            
            # Keep a reference to prevent garbage collection
            canvas.image = photo
            
        except Exception as e:
            logger.error(f"Error displaying image: {e}")
    
    def _show_game_state_window(self, game_state: 'GameState'):
        """Show detailed game state in a separate window"""
        try:
            # Create new window
            state_window = ctk.CTkToplevel(self.root)
            state_window.title("Detailed Game State Analysis")
            state_window.geometry("800x600")
            
            # Create scrollable text area
            text_area = ctk.CTkTextbox(state_window)
            text_area.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Format detailed game state
            detailed_text = f"""🧠 COMPREHENSIVE GAME STATE ANALYSIS
═══════════════════════════════════════════════════════════════

📅 Analysis Metadata:
  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(game_state.timestamp))}
  Processing Time: {game_state.processing_time:.4f} seconds
  Analysis Version: {game_state.analysis_version}
  Frame Quality: {game_state.frame_quality:.2f}

🎯 Scene Classification:
  Scene Type: {game_state.scene_type.value.upper()}
  Confidence: {game_state.confidence:.4f}

👤 Player Status:
  Health: {game_state.player_status.health_percent:.2f}%
  Prayer: {game_state.player_status.prayer_percent:.2f}%
  Energy: {game_state.player_status.energy_percent:.2f}%
  Combat Level: {game_state.player_status.combat_level or 'Unknown'}
  In Combat: {game_state.player_status.is_in_combat}
  Moving: {game_state.player_status.is_moving}
  Animation: {game_state.player_status.animation_state}

🗺️ Minimap Analysis:
  Player Position: {game_state.minimap.player_position}
  North Direction: {game_state.minimap.north_direction:.1f}°
  Region Type: {game_state.minimap.region_type}
  NPCs Visible: {len(game_state.minimap.visible_npcs)}
  Players Visible: {len(game_state.minimap.visible_players)}
  Points of Interest: {len(game_state.minimap.points_of_interest)}

🎒 Inventory Analysis:
  Total Items: {len(game_state.inventory.items)}
  Free Slots: {game_state.inventory.free_slots}/28
  Valuable Items: {len(game_state.inventory.valuable_items)}
  Consumables: {len(game_state.inventory.consumables)}
  Equipment: {len(game_state.inventory.equipment)}

💬 Interface State:
  Open Interfaces: {len(game_state.interface_state.open_interfaces)}
  Clickable Elements: {len(game_state.interface_state.clickable_elements)}
  Dialog Messages: {len(game_state.interface_state.dialog_text)}
  Chat Messages: {len(game_state.interface_state.active_chat)}

🎯 Detected Objects Summary:
  NPCs: {len(game_state.npcs)}
  Items: {len(game_state.items)}
  Players: {len(game_state.players)}
  UI Elements: {len(game_state.ui_elements)}
  Environment: {len(game_state.environment)}

🏆 High Priority Objects:
"""
            
            # Add high priority objects
            priority_objects = game_state.get_highest_priority_objects(10)
            for i, obj in enumerate(priority_objects, 1):
                detailed_text += f"  {i:2d}. {obj.label:<15} | Type: {obj.object_type:<12} | Priority: {obj.action_priority:.3f} | Confidence: {obj.confidence:.3f}\n"
            
            # Add detailed object listings
            if game_state.npcs:
                detailed_text += f"\n🔥 NPCs Detected ({len(game_state.npcs)}):\n"
                for i, npc in enumerate(game_state.npcs[:10], 1):  # Show top 10
                    detailed_text += f"  {i:2d}. {npc.label} at ({npc.x}, {npc.y}) - Priority: {npc.action_priority:.3f}\n"
            
            if game_state.items:
                detailed_text += f"\n💎 Items Detected ({len(game_state.items)}):\n"
                for i, item in enumerate(game_state.items[:10], 1):  # Show top 10
                    detailed_text += f"  {i:2d}. {item.label} at ({item.x}, {item.y}) - Priority: {item.action_priority:.3f}\n"
            
            if game_state.interface_state.active_chat:
                detailed_text += f"\n💬 Recent Chat Messages:\n"
                for i, msg in enumerate(game_state.interface_state.active_chat[-5:], 1):  # Last 5 messages
                    detailed_text += f"  {i}. {msg}\n"
            
            # Insert text
            text_area.insert("0.0", detailed_text)
            
        except Exception as e:
            logger.error(f"Error showing game state window: {e}")
            messagebox.showerror("Error", f"Failed to show game state: {e}")
    
    # Placeholder methods for remaining functionality
    def open_settings(self): pass
    def create_new_bot(self): pass
    def load_bot(self): pass
    def refresh_bots(self): pass
    def start_selected_bot(self): pass
    def pause_selected_bot(self): pass
    def stop_selected_bot(self): pass
    def show_bot_details(self): pass
    def capture_screen(self): pass
    def test_detection(self): pass
    def toggle_live_view(self): pass
    def toggle_monitoring(self): pass
    def refresh_charts(self): pass
    def export_performance_data(self): pass
    def refresh_logs(self): pass
    def clear_logs(self): pass
    def export_logs(self): pass
    def filter_logs(self, level): pass
    def create_template(self): pass
    def import_template(self): pass
    def refresh_templates(self): pass
    def on_template_select(self, event): pass
    # Navigation fallback methods
    def capture_minimap_manual(self):
        """Manual minimap capture for fallback"""
        logger.info("Manual minimap capture requested")
        messagebox.showinfo("Navigation", "Minimap capture would be performed here when dependencies are available")
    
    def analyze_image_manual(self):
        """Manual image analysis for fallback"""
        logger.info("Manual image analysis requested")
        messagebox.showinfo("Navigation", "Image analysis would be performed here when dependencies are available")
    
    def show_navigation_stats(self):
        """Show navigation statistics for fallback"""
        logger.info("Navigation statistics requested")
        messagebox.showinfo("Navigation", "Navigation statistics would be displayed here when dependencies are available")


def main():
    """Main entry point"""
    try:
        app = OSRSBotGUI()
        app.run()
    except Exception as e:
        print(f"Failed to start GUI: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install --user customtkinter pillow matplotlib")


if __name__ == "__main__":
    main() 