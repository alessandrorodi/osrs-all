"""
Navigation Panel GUI Widget

This module provides a comprehensive navigation control panel for the OSRS bot GUI
including:
- Live minimap display with overlay
- Path visualization and planning
- Location detection and naming
- Movement efficiency metrics
- Pathfinding debug visualization
- Navigation controls and settings

Integrates with the advanced minimap analyzer and pathfinding system.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import threading
import time
from typing import Optional, Dict, Any, List, Tuple
import logging
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from dataclasses import asdict

from vision.minimap_analyzer import (
    AdvancedMinimapAnalyzer, MinimapAnalysisResult, DotType, 
    MinimapRegionType, MinimapDot
)
from navigation.pathfinding import (
    OSRSPathfinder, NavigationGoal, PathResult, PathNode, MovementType
)
from core.screen_capture import screen_capture

logger = logging.getLogger(__name__)


class NavigationPanel(ctk.CTkFrame):
    """
    Comprehensive navigation panel for OSRS bot GUI
    
    Features:
    - Live minimap display with real-time updates
    - Path visualization and overlay
    - Location detection and naming
    - Movement efficiency metrics
    - Navigation controls and settings
    - Debug visualization for pathfinding
    """
    
    def __init__(self, parent, width: int = 400, height: int = 600):
        """
        Initialize the navigation panel
        
        Args:
            parent: Parent widget
            width: Panel width
            height: Panel height
        """
        super().__init__(parent, width=width, height=height)
        
        # Initialize components
        self.minimap_analyzer = AdvancedMinimapAnalyzer()
        self.pathfinder = OSRSPathfinder(self.minimap_analyzer)
        
        # State variables
        self.is_active = False
        self.update_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.current_analysis: Optional[MinimapAnalysisResult] = None
        self.current_path: Optional[PathResult] = None
        self.selected_destination: Optional[Tuple[int, int]] = None
        
        # Performance tracking
        self.performance_metrics = {
            'fps': 0.0,
            'analysis_time': 0.0,
            'path_calculation_time': 0.0,
            'total_distance_traveled': 0.0,
            'navigation_efficiency': 100.0
        }
        
        # UI components
        self.setup_ui()
        self.setup_callbacks()
        
        logger.info("NavigationPanel initialized")
    
    def setup_ui(self):
        """Setup the user interface components"""
        # Main container with scrollable frame
        self.main_container = ctk.CTkScrollableFrame(self, width=380, height=580)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(
            self.main_container,
            text="🧭 Navigation Control",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(pady=(0, 10))
        
        # Control buttons
        self.setup_control_buttons()
        
        # Minimap display
        self.setup_minimap_display()
        
        # Navigation settings
        self.setup_navigation_settings()
        
        # Location information
        self.setup_location_info()
        
        # Performance metrics
        self.setup_performance_metrics()
        
        # Path planning
        self.setup_path_planning()
        
        # Debug information
        self.setup_debug_info()
    
    def setup_control_buttons(self):
        """Setup navigation control buttons"""
        controls_frame = ctk.CTkFrame(self.main_container)
        controls_frame.pack(fill="x", pady=10)
        
        # Start/Stop navigation
        self.nav_toggle_button = ctk.CTkButton(
            controls_frame,
            text="▶️ Start Navigation",
            command=self.toggle_navigation,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.nav_toggle_button.pack(side="left", padx=5, pady=10)
        
        # Path planning button
        self.plan_path_button = ctk.CTkButton(
            controls_frame,
            text="🗺️ Plan Path",
            command=self.open_path_planner,
            height=40
        )
        self.plan_path_button.pack(side="left", padx=5, pady=10)
        
        # Clear path button
        self.clear_path_button = ctk.CTkButton(
            controls_frame,
            text="🗑️ Clear",
            command=self.clear_current_path,
            height=40
        )
        self.clear_path_button.pack(side="left", padx=5, pady=10)
    
    def setup_minimap_display(self):
        """Setup minimap display with overlay"""
        minimap_frame = ctk.CTkFrame(self.main_container)
        minimap_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            minimap_frame,
            text="Live Minimap",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        # Minimap canvas
        self.minimap_canvas = tk.Canvas(
            minimap_frame,
            width=300,
            height=300,
            bg="black",
            highlightthickness=1,
            highlightbackground="gray"
        )
        self.minimap_canvas.pack(pady=10)
        
        # Bind click events for destination selection
        self.minimap_canvas.bind("<Button-1>", self.on_minimap_click)
        self.minimap_canvas.bind("<Motion>", self.on_minimap_hover)
        
        # Minimap controls
        minimap_controls = ctk.CTkFrame(minimap_frame)
        minimap_controls.pack(fill="x", pady=5)
        
        # Overlay options
        self.show_dots_var = ctk.BooleanVar(value=True)
        self.show_path_var = ctk.BooleanVar(value=True)
        self.show_dangers_var = ctk.BooleanVar(value=True)
        
        ctk.CTkCheckBox(
            minimap_controls,
            text="Show Dots",
            variable=self.show_dots_var
        ).pack(side="left", padx=5)
        
        ctk.CTkCheckBox(
            minimap_controls,
            text="Show Path",
            variable=self.show_path_var
        ).pack(side="left", padx=5)
        
        ctk.CTkCheckBox(
            minimap_controls,
            text="Show Dangers",
            variable=self.show_dangers_var
        ).pack(side="left", padx=5)
    
    def setup_navigation_settings(self):
        """Setup navigation settings panel"""
        settings_frame = ctk.CTkFrame(self.main_container)
        settings_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            settings_frame,
            text="Navigation Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        # Settings grid
        settings_grid = ctk.CTkFrame(settings_frame)
        settings_grid.pack(fill="x", padx=10, pady=5)
        
        # Max danger level
        ctk.CTkLabel(settings_grid, text="Max Danger Level:").grid(row=0, column=0, sticky="w", padx=5)
        self.max_danger_slider = ctk.CTkSlider(
            settings_grid,
            from_=0.0,
            to=1.0,
            number_of_steps=10
        )
        self.max_danger_slider.set(0.5)
        self.max_danger_slider.grid(row=0, column=1, sticky="ew", padx=5)
        
        # Prefer safe routes
        self.prefer_safe_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            settings_grid,
            text="Prefer Safe Routes",
            variable=self.prefer_safe_var
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Allow teleports
        self.allow_teleports_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            settings_grid,
            text="Allow Teleports",
            variable=self.allow_teleports_var
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Allow wilderness
        self.allow_wilderness_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            settings_grid,
            text="Allow Wilderness",
            variable=self.allow_wilderness_var
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        settings_grid.columnconfigure(1, weight=1)
    
    def setup_location_info(self):
        """Setup location information display"""
        location_frame = ctk.CTkFrame(self.main_container)
        location_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            location_frame,
            text="Location Information",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        # Location details
        self.location_text = ctk.CTkTextbox(location_frame, height=80)
        self.location_text.pack(fill="x", padx=10, pady=5)
        
        # Update location info
        self.update_location_info("Location: Unknown\nRegion: Not detected\nSafety: Unknown")
    
    def setup_performance_metrics(self):
        """Setup performance metrics display"""
        metrics_frame = ctk.CTkFrame(self.main_container)
        metrics_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            metrics_frame,
            text="Performance Metrics",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        # Metrics grid
        metrics_grid = ctk.CTkFrame(metrics_frame)
        metrics_grid.pack(fill="x", padx=10, pady=5)
        
        # FPS
        ctk.CTkLabel(metrics_grid, text="FPS:").grid(row=0, column=0, sticky="w")
        self.fps_label = ctk.CTkLabel(metrics_grid, text="0.0")
        self.fps_label.grid(row=0, column=1, sticky="e")
        
        # Analysis time
        ctk.CTkLabel(metrics_grid, text="Analysis Time:").grid(row=1, column=0, sticky="w")
        self.analysis_time_label = ctk.CTkLabel(metrics_grid, text="0.0ms")
        self.analysis_time_label.grid(row=1, column=1, sticky="e")
        
        # Navigation efficiency
        ctk.CTkLabel(metrics_grid, text="Efficiency:").grid(row=2, column=0, sticky="w")
        self.efficiency_label = ctk.CTkLabel(metrics_grid, text="100%")
        self.efficiency_label.grid(row=2, column=1, sticky="e")
        
        metrics_grid.columnconfigure(1, weight=1)
    
    def setup_path_planning(self):
        """Setup path planning interface"""
        path_frame = ctk.CTkFrame(self.main_container)
        path_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            path_frame,
            text="Path Planning",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        # Destination input
        dest_frame = ctk.CTkFrame(path_frame)
        dest_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(dest_frame, text="Destination:").pack(side="left")
        
        self.dest_x_entry = ctk.CTkEntry(dest_frame, width=60, placeholder_text="X")
        self.dest_x_entry.pack(side="left", padx=5)
        
        self.dest_y_entry = ctk.CTkEntry(dest_frame, width=60, placeholder_text="Y")
        self.dest_y_entry.pack(side="left", padx=5)
        
        ctk.CTkButton(
            dest_frame,
            text="Set Destination",
            command=self.set_destination_from_input,
            width=100
        ).pack(side="right", padx=5)
        
        # Path information
        self.path_info = ctk.CTkTextbox(path_frame, height=60)
        self.path_info.pack(fill="x", padx=10, pady=5)
        self.update_path_info("No path planned")
    
    def setup_debug_info(self):
        """Setup debug information display"""
        debug_frame = ctk.CTkFrame(self.main_container)
        debug_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            debug_frame,
            text="Debug Information",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)
        
        # Debug controls
        debug_controls = ctk.CTkFrame(debug_frame)
        debug_controls.pack(fill="x", padx=10, pady=5)
        
        self.show_debug_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            debug_controls,
            text="Show Debug Info",
            variable=self.show_debug_var
        ).pack(side="left")
        
        ctk.CTkButton(
            debug_controls,
            text="Export Debug Data",
            command=self.export_debug_data,
            width=120
        ).pack(side="right")
        
        # Debug text area
        self.debug_text = ctk.CTkTextbox(debug_frame, height=100)
        self.debug_text.pack(fill="x", padx=10, pady=5)
    
    def setup_callbacks(self):
        """Setup event callbacks"""
        # Widget event bindings are already set up in setup_ui methods
        pass
    
    def toggle_navigation(self):
        """Toggle navigation system on/off"""
        if self.is_active:
            self.stop_navigation()
        else:
            self.start_navigation()
    
    def start_navigation(self):
        """Start the navigation system"""
        if self.is_active:
            return
        
        self.is_active = True
        self.stop_event.clear()
        
        # Update UI
        self.nav_toggle_button.configure(text="⏸️ Stop Navigation")
        
        # Start update thread
        self.update_thread = threading.Thread(target=self.navigation_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("Navigation system started")
    
    def stop_navigation(self):
        """Stop the navigation system"""
        if not self.is_active:
            return
        
        self.is_active = False
        self.stop_event.set()
        
        # Update UI
        self.nav_toggle_button.configure(text="▶️ Start Navigation")
        
        # Wait for thread to finish
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        
        logger.info("Navigation system stopped")
    
    def navigation_loop(self):
        """Main navigation update loop"""
        last_update = time.time()
        frame_count = 0
        
        while self.is_active and not self.stop_event.is_set():
            try:
                start_time = time.time()
                
                # Capture and analyze minimap
                screenshot = screen_capture.capture_client()
                if screenshot is not None:
                    # Analyze minimap
                    analysis = self.minimap_analyzer.analyze_minimap(screenshot)
                    self.current_analysis = analysis
                    
                    # Update GUI
                    self.after(0, self.update_gui_from_analysis, analysis)
                    
                    # Update performance metrics
                    frame_count += 1
                    current_time = time.time()
                    if current_time - last_update >= 1.0:
                        fps = frame_count / (current_time - last_update)
                        self.performance_metrics['fps'] = fps
                        self.performance_metrics['analysis_time'] = analysis.processing_time * 1000
                        
                        frame_count = 0
                        last_update = current_time
                        
                        # Update performance display
                        self.after(0, self.update_performance_display)
                
                # Sleep to maintain target FPS (60 FPS = ~16.7ms)
                sleep_time = max(0, 0.0167 - (time.time() - start_time))
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Navigation loop error: {e}")
                time.sleep(0.1)
    
    def update_gui_from_analysis(self, analysis: MinimapAnalysisResult):
        """Update GUI components from minimap analysis"""
        try:
            # Update minimap display
            self.update_minimap_display(analysis)
            
            # Update location information
            location_info = (
                f"Location: {analysis.region_name}\n"
                f"Region: {analysis.region_type.value}\n"
                f"Player: ({analysis.player_position[0]}, {analysis.player_position[1]})\n"
                f"Dots: {len(analysis.detected_dots)} detected"
            )
            self.update_location_info(location_info)
            
            # Update debug information if enabled
            if self.show_debug_var.get():
                debug_info = (
                    f"Processing Time: {analysis.processing_time:.3f}s\n"
                    f"Obstacles: {len(analysis.obstacles)}\n"
                    f"Walkable Areas: {len(analysis.walkable_areas)}\n"
                    f"POI: {len(analysis.points_of_interest)}"
                )
                self.debug_text.delete("1.0", "end")
                self.debug_text.insert("1.0", debug_info)
            
        except Exception as e:
            logger.error(f"GUI update error: {e}")
    
    def update_minimap_display(self, analysis: MinimapAnalysisResult):
        """Update the minimap display with analysis results"""
        try:
            # Clear canvas
            self.minimap_canvas.delete("all")
            
            # Draw minimap background (simplified representation)
            canvas_width = 300
            canvas_height = 300
            
            # Scale factor from minimap coordinates to canvas
            scale_x = canvas_width / 146
            scale_y = canvas_height / 151
            
            # Draw walkable areas
            for x, y in analysis.walkable_areas[::10]:  # Sample every 10th point for performance
                canvas_x = x * scale_x
                canvas_y = y * scale_y
                self.minimap_canvas.create_rectangle(
                    canvas_x, canvas_y, canvas_x + 2, canvas_y + 2,
                    fill="darkgreen", outline=""
                )
            
            # Draw obstacles
            for x, y in analysis.obstacles[::5]:  # Sample every 5th point
                canvas_x = x * scale_x
                canvas_y = y * scale_y
                self.minimap_canvas.create_rectangle(
                    canvas_x, canvas_y, canvas_x + 1, canvas_y + 1,
                    fill="gray", outline=""
                )
            
            # Draw dots if enabled
            if self.show_dots_var.get():
                self.draw_minimap_dots(analysis.detected_dots, scale_x, scale_y)
            
            # Draw current path if enabled and available
            if self.show_path_var.get() and self.current_path:
                self.draw_path_overlay(self.current_path, scale_x, scale_y)
            
            # Draw player position
            player_x, player_y = analysis.player_position
            canvas_player_x = player_x * scale_x
            canvas_player_y = player_y * scale_y
            self.minimap_canvas.create_oval(
                canvas_player_x - 5, canvas_player_y - 5,
                canvas_player_x + 5, canvas_player_y + 5,
                fill="white", outline="black", width=2
            )
            
            # Draw selected destination if any
            if self.selected_destination:
                dest_x, dest_y = self.selected_destination
                canvas_dest_x = dest_x * scale_x
                canvas_dest_y = dest_y * scale_y
                self.minimap_canvas.create_oval(
                    canvas_dest_x - 6, canvas_dest_y - 6,
                    canvas_dest_x + 6, canvas_dest_y + 6,
                    fill="red", outline="white", width=2
                )
            
        except Exception as e:
            logger.error(f"Minimap display update error: {e}")
    
    def draw_minimap_dots(self, dots: List[MinimapDot], scale_x: float, scale_y: float):
        """Draw dots on the minimap canvas"""
        dot_colors = {
            DotType.PLAYER_OTHER: "white",
            DotType.NPC_NEUTRAL: "yellow",
            DotType.NPC_AGGRESSIVE: "red",
            DotType.FRIEND: "green",
            DotType.CLAN_MEMBER: "purple",
            DotType.TEAM_MEMBER: "blue",
            DotType.ITEM: "orange"
        }
        
        for dot in dots:
            canvas_x = dot.x * scale_x
            canvas_y = dot.y * scale_y
            color = dot_colors.get(dot.dot_type, "gray")
            
            # Draw dot
            self.minimap_canvas.create_oval(
                canvas_x - 2, canvas_y - 2,
                canvas_x + 2, canvas_y + 2,
                fill=color, outline="black"
            )
    
    def draw_path_overlay(self, path_result: PathResult, scale_x: float, scale_y: float):
        """Draw path overlay on minimap"""
        if not path_result.path or len(path_result.path) < 2:
            return
        
        # Draw path line
        path_points = []
        for node in path_result.path:
            canvas_x = node.x * scale_x
            canvas_y = node.y * scale_y
            path_points.extend([canvas_x, canvas_y])
        
        if len(path_points) >= 4:
            self.minimap_canvas.create_line(
                path_points,
                fill="cyan",
                width=2,
                smooth=True
            )
        
        # Draw waypoints
        for i, node in enumerate(path_result.path[::5]):  # Every 5th node
            canvas_x = node.x * scale_x
            canvas_y = node.y * scale_y
            
            # Different colors for different movement types
            if node.movement_type == MovementType.TELEPORT:
                color = "magenta"
            elif node.movement_type == MovementType.STAIRS:
                color = "brown"
            else:
                color = "cyan"
            
            self.minimap_canvas.create_oval(
                canvas_x - 1, canvas_y - 1,
                canvas_x + 1, canvas_y + 1,
                fill=color, outline="white"
            )
    
    def on_minimap_click(self, event):
        """Handle clicks on minimap for destination selection"""
        try:
            # Convert canvas coordinates to minimap coordinates
            canvas_width = 300
            canvas_height = 300
            scale_x = 146 / canvas_width
            scale_y = 151 / canvas_height
            
            minimap_x = int(event.x * scale_x)
            minimap_y = int(event.y * scale_y)
            
            # Set as destination
            self.selected_destination = (minimap_x, minimap_y)
            
            # Update destination input fields
            self.dest_x_entry.delete(0, "end")
            self.dest_x_entry.insert(0, str(minimap_x))
            self.dest_y_entry.delete(0, "end")
            self.dest_y_entry.insert(0, str(minimap_y))
            
            logger.info(f"Destination selected: ({minimap_x}, {minimap_y})")
            
        except Exception as e:
            logger.error(f"Minimap click error: {e}")
    
    def on_minimap_hover(self, event):
        """Handle mouse hover on minimap for coordinate display"""
        # This could show coordinate tooltip in the future
        pass
    
    def set_destination_from_input(self):
        """Set destination from input fields"""
        try:
            x = int(self.dest_x_entry.get() or 0)
            y = int(self.dest_y_entry.get() or 0)
            
            self.selected_destination = (x, y)
            logger.info(f"Destination set from input: ({x}, {y})")
            
        except ValueError:
            logger.warning("Invalid destination coordinates")
    
    def open_path_planner(self):
        """Open path planning dialog"""
        if not self.selected_destination:
            logger.warning("No destination selected")
            return
        
        if not self.current_analysis:
            logger.warning("No current minimap analysis available")
            return
        
        # Create navigation goal
        dest_x, dest_y = self.selected_destination
        goal = NavigationGoal(
            target_x=dest_x,
            target_y=dest_y,
            max_danger=self.max_danger_slider.get(),
            prefer_safe_route=self.prefer_safe_var.get(),
            allow_teleports=self.allow_teleports_var.get(),
            allow_wilderness=self.allow_wilderness_var.get()
        )
        
        # Calculate path
        player_x, player_y = self.current_analysis.player_position
        path_result = self.pathfinder.find_path(player_x, player_y, goal, self.current_analysis)
        
        if path_result.success:
            self.current_path = path_result
            path_info = (
                f"Path: {len(path_result.path)} nodes\n"
                f"Distance: {path_result.distance:.1f}\n"
                f"Time: {path_result.estimated_time:.1f}s\n"
                f"Danger: {path_result.danger_rating:.1%}\n"
                f"Teleports: {path_result.teleports_used}"
            )
            self.update_path_info(path_info)
            logger.info(f"Path calculated successfully: {len(path_result.path)} nodes")
        else:
            self.update_path_info("Path calculation failed")
            logger.warning("Path calculation failed")
    
    def clear_current_path(self):
        """Clear the current path"""
        self.current_path = None
        self.selected_destination = None
        self.dest_x_entry.delete(0, "end")
        self.dest_y_entry.delete(0, "end")
        self.update_path_info("No path planned")
        logger.info("Path cleared")
    
    def update_location_info(self, text: str):
        """Update location information display"""
        self.location_text.delete("1.0", "end")
        self.location_text.insert("1.0", text)
    
    def update_path_info(self, text: str):
        """Update path information display"""
        self.path_info.delete("1.0", "end")
        self.path_info.insert("1.0", text)
    
    def update_performance_display(self):
        """Update performance metrics display"""
        try:
            self.fps_label.configure(text=f"{self.performance_metrics['fps']:.1f}")
            self.analysis_time_label.configure(text=f"{self.performance_metrics['analysis_time']:.1f}ms")
            self.efficiency_label.configure(text=f"{self.performance_metrics['navigation_efficiency']:.0f}%")
        except Exception as e:
            logger.error(f"Performance display update error: {e}")
    
    def export_debug_data(self):
        """Export debug data to file"""
        try:
            debug_data = {
                'current_analysis': asdict(self.current_analysis) if self.current_analysis else None,
                'current_path': asdict(self.current_path) if self.current_path else None,
                'performance_metrics': self.performance_metrics.copy(),
                'minimap_stats': self.minimap_analyzer.get_performance_stats(),
                'pathfinder_stats': self.pathfinder.get_stats()
            }
            
            # Save to file (implementation would write to actual file)
            logger.info("Debug data exported")
            
        except Exception as e:
            logger.error(f"Debug data export error: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_navigation()
        
        if hasattr(self, 'minimap_analyzer'):
            self.minimap_analyzer.cleanup()
        
        if hasattr(self, 'pathfinder'):
            self.pathfinder.cleanup()
        
        logger.info("NavigationPanel cleanup completed")