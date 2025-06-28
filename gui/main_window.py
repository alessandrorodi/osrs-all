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
        """Create the computer vision tab"""
        vision_tab = self.notebook.add("👁️ Vision")
        
        # Vision controls
        controls_frame = ctk.CTkFrame(vision_tab)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(
            controls_frame,
            text="📷 Capture Screen",
            command=self.capture_screen
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            controls_frame,
            text="🔍 Test Detection",
            command=self.test_detection
        ).pack(side="left", padx=5, pady=10)
        
        ctk.CTkButton(
            controls_frame,
            text="🎯 Live View",
            command=self.toggle_live_view
        ).pack(side="left", padx=5, pady=10)
        
        # Vision display
        display_frame = ctk.CTkFrame(vision_tab)
        display_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Canvas for image display
        self.vision_canvas = tk.Canvas(display_frame, bg="black")
        self.vision_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Detection results
        results_frame = ctk.CTkFrame(vision_tab)
        results_frame.pack(fill="x", padx=10, pady=10)
        
        self.detection_results = ctk.CTkTextbox(results_frame, height=100)
        self.detection_results.pack(fill="x", padx=10, pady=10)
    
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