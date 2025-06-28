"""
OSRS Text Intelligence GUI Widgets

GUI components for displaying and interacting with OSRS text intelligence data.
Provides live text overlays, chat message filtering, XP tracking, and
intelligent text visualization integrated with the main GUI framework.

Components:
- TextIntelligencePanel: Main control panel for text intelligence
- LiveTextOverlay: Real-time text overlay on game feed
- ChatFilterWidget: Advanced chat message filtering
- XPTrackingWidget: XP rate and progression tracking
- ItemValueWidget: Real-time item value display
- AlertWidget: Important alerts and notifications
"""

import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from typing import Dict, List, Optional, Any, Callable
import time
import threading
from datetime import datetime
import logging

from vision.osrs_ocr import ChatMessage, ItemInfo, PlayerStats, osrs_text_intelligence
from core.text_intelligence import XPEvent, TextPriority, text_intelligence

logger = logging.getLogger(__name__)


class TextIntelligencePanel(ctk.CTkFrame):
    """
    Main text intelligence control panel
    
    Provides configuration and monitoring for all text intelligence features
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.parent = parent
        self.is_active = False
        self.update_interval = 1000  # milliseconds
        self.last_update = 0
        
        # Text intelligence data
        self.current_data = {}
        self.intelligence_results = {}
        
        # Setup UI
        self.setup_ui()
        self.setup_callbacks()
        
        logger.info("Text Intelligence Panel initialized")
    
    def setup_ui(self):
        """Setup the text intelligence panel UI"""
        # Header
        header_frame = ctk.CTkFrame(self)
        header_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="🧠 Text Intelligence",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(side="left", padx=10, pady=10)
        
        # Status indicator
        self.status_label = ctk.CTkLabel(
            header_frame,
            text="● Inactive",
            text_color="red"
        )
        self.status_label.pack(side="right", padx=10, pady=10)
        
        # Control buttons
        controls_frame = ctk.CTkFrame(self)
        controls_frame.pack(fill="x", padx=10, pady=5)
        
        self.start_button = ctk.CTkButton(
            controls_frame,
            text="▶️ Start Intelligence",
            command=self.toggle_intelligence,
            width=140
        )
        self.start_button.pack(side="left", padx=5, pady=10)
        
        self.refresh_button = ctk.CTkButton(
            controls_frame,
            text="🔄 Refresh",
            command=self.refresh_display,
            width=100
        )
        self.refresh_button.pack(side="left", padx=5, pady=10)
        
        self.settings_button = ctk.CTkButton(
            controls_frame,
            text="⚙️ Settings",
            command=self.open_settings,
            width=100
        )
        self.settings_button.pack(side="left", padx=5, pady=10)
        
        # Main content area with tabs
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_overview_tab()
        self.create_chat_tab()
        self.create_xp_tab()
        self.create_items_tab()
        self.create_alerts_tab()
    
    def create_overview_tab(self):
        """Create the overview tab"""
        overview_tab = self.tab_view.add("📊 Overview")
        
        # Quick stats
        stats_frame = ctk.CTkFrame(overview_tab)
        stats_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            stats_frame,
            text="Session Statistics",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10)
        
        # Stats grid
        self.stats_grid = ctk.CTkFrame(stats_frame)
        self.stats_grid.pack(fill="x", padx=10, pady=10)
        
        # Initialize stat labels
        self.stat_labels = {}
        stat_names = [
            "Session Duration", "Total XP Gained", "Messages Processed",
            "Items Detected", "Alerts Generated", "Processing Rate"
        ]
        
        for i, stat_name in enumerate(stat_names):
            row = i // 2
            col = i % 2
            
            label = ctk.CTkLabel(self.stats_grid, text=f"{stat_name}: --")
            label.grid(row=row, column=col, padx=20, pady=5, sticky="w")
            self.stat_labels[stat_name] = label
        
        # Performance metrics
        perf_frame = ctk.CTkFrame(overview_tab)
        perf_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            perf_frame,
            text="Performance Metrics",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10)
        
        self.performance_display = ctk.CTkTextbox(perf_frame, height=150)
        self.performance_display.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_chat_tab(self):
        """Create the chat analysis tab"""
        chat_tab = self.tab_view.add("💬 Chat")
        
        # Chat filter controls
        filter_frame = ctk.CTkFrame(chat_tab)
        filter_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(filter_frame, text="Chat Filters:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=5)
        
        # Filter checkboxes
        filter_options_frame = ctk.CTkFrame(filter_frame)
        filter_options_frame.pack(fill="x", padx=10, pady=5)
        
        self.filter_options = {}
        filter_types = ["Public", "Private", "Game Messages", "System", "Important Only"]
        
        for i, filter_type in enumerate(filter_types):
            var = tk.BooleanVar(value=True)
            checkbox = ctk.CTkCheckBox(
                filter_options_frame,
                text=filter_type,
                variable=var,
                command=self.update_chat_filter
            )
            checkbox.grid(row=0, column=i, padx=10, pady=5)
            self.filter_options[filter_type] = var
        
        # Chat display
        chat_display_frame = ctk.CTkFrame(chat_tab)
        chat_display_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            chat_display_frame,
            text="Recent Chat Messages",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10)
        
        self.chat_display = ctk.CTkTextbox(chat_display_frame)
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_xp_tab(self):
        """Create the XP tracking tab"""
        xp_tab = self.tab_view.add("⚡ XP Tracking")
        
        # XP summary
        summary_frame = ctk.CTkFrame(xp_tab)
        summary_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            summary_frame,
            text="XP Summary",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10)
        
        self.xp_summary_display = ctk.CTkTextbox(summary_frame, height=100)
        self.xp_summary_display.pack(fill="x", padx=10, pady=10)
        
        # XP rates table
        rates_frame = ctk.CTkFrame(xp_tab)
        rates_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            rates_frame,
            text="Current XP Rates",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10)
        
        # Create treeview for XP rates
        columns = ("Skill", "Session XP", "XP/Hour", "Level Est.", "Time to Level")
        self.xp_tree = ttk.Treeview(rates_frame, columns=columns, show="headings", height=8)
        
        for col in columns:
            self.xp_tree.heading(col, text=col)
            self.xp_tree.column(col, width=100)
        
        # Scrollbar for treeview
        xp_scrollbar = ttk.Scrollbar(rates_frame, orient="vertical", command=self.xp_tree.yview)
        self.xp_tree.configure(yscrollcommand=xp_scrollbar.set)
        
        self.xp_tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        xp_scrollbar.pack(side="right", fill="y", padx=(0, 10), pady=10)
    
    def create_items_tab(self):
        """Create the items analysis tab"""
        items_tab = self.tab_view.add("💎 Items")
        
        # Item value summary
        value_frame = ctk.CTkFrame(items_tab)
        value_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            value_frame,
            text="Inventory Value",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10)
        
        self.value_labels = {}
        value_stats = ["Total Value", "Valuable Items", "Average Item Value", "Most Valuable"]
        
        for i, stat in enumerate(value_stats):
            label = ctk.CTkLabel(value_frame, text=f"{stat}: --")
            label.pack(anchor="w", padx=20, pady=2)
            self.value_labels[stat] = label
        
        # Item list
        items_list_frame = ctk.CTkFrame(items_tab)
        items_list_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            items_list_frame,
            text="Detected Items",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10)
        
        # Items treeview
        item_columns = ("Item", "Quantity", "Unit Price", "Total Value", "Category")
        self.items_tree = ttk.Treeview(items_list_frame, columns=item_columns, show="headings", height=10)
        
        for col in item_columns:
            self.items_tree.heading(col, text=col)
            self.items_tree.column(col, width=100)
        
        items_scrollbar = ttk.Scrollbar(items_list_frame, orient="vertical", command=self.items_tree.yview)
        self.items_tree.configure(yscrollcommand=items_scrollbar.set)
        
        self.items_tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        items_scrollbar.pack(side="right", fill="y", padx=(0, 10), pady=10)
    
    def create_alerts_tab(self):
        """Create the alerts tab"""
        alerts_tab = self.tab_view.add("🚨 Alerts")
        
        # Alert controls
        alert_controls = ctk.CTkFrame(alerts_tab)
        alert_controls.pack(fill="x", padx=10, pady=10)
        
        self.alert_sound = ctk.CTkCheckBox(alert_controls, text="Sound Alerts")
        self.alert_sound.pack(side="left", padx=10, pady=10)
        
        self.alert_popup = ctk.CTkCheckBox(alert_controls, text="Popup Alerts")
        self.alert_popup.pack(side="left", padx=10, pady=10)
        
        clear_button = ctk.CTkButton(
            alert_controls,
            text="Clear All",
            command=self.clear_alerts,
            width=100
        )
        clear_button.pack(side="right", padx=10, pady=10)
        
        # Alerts display
        alerts_display_frame = ctk.CTkFrame(alerts_tab)
        alerts_display_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            alerts_display_frame,
            text="Recent Alerts",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10)
        
        self.alerts_display = ctk.CTkTextbox(alerts_display_frame)
        self.alerts_display.pack(fill="both", expand=True, padx=10, pady=10)
    
    def setup_callbacks(self):
        """Setup UI callbacks and bindings"""
        # Auto-update timer
        self.after(self.update_interval, self.update_display)
    
    def toggle_intelligence(self):
        """Toggle text intelligence processing"""
        self.is_active = not self.is_active
        
        if self.is_active:
            self.start_button.configure(text="⏸️ Stop Intelligence")
            self.status_label.configure(text="● Active", text_color="green")
            logger.info("Text intelligence activated")
        else:
            self.start_button.configure(text="▶️ Start Intelligence")
            self.status_label.configure(text="● Inactive", text_color="red")
            logger.info("Text intelligence deactivated")
    
    def refresh_display(self):
        """Force refresh of all displays"""
        if self.is_active:
            self.update_display()
    
    def open_settings(self):
        """Open text intelligence settings"""
        # Create settings dialog
        settings_window = ctk.CTkToplevel(self)
        settings_window.title("Text Intelligence Settings")
        settings_window.geometry("400x300")
        
        # Settings content
        ctk.CTkLabel(
            settings_window,
            text="Text Intelligence Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=20)
        
        # Update interval setting
        interval_frame = ctk.CTkFrame(settings_window)
        interval_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(interval_frame, text="Update Interval (ms):").pack(side="left", padx=10)
        
        interval_var = tk.StringVar(value=str(self.update_interval))
        interval_entry = ctk.CTkEntry(interval_frame, textvariable=interval_var, width=80)
        interval_entry.pack(side="right", padx=10)
        
        # Apply button
        def apply_settings():
            try:
                self.update_interval = int(interval_var.get())
                settings_window.destroy()
            except ValueError:
                pass
        
        ctk.CTkButton(
            settings_window,
            text="Apply",
            command=apply_settings
        ).pack(pady=20)
    
    def update_display(self):
        """Update all display elements"""
        if not self.is_active:
            self.after(self.update_interval, self.update_display)
            return
        
        try:
            # Get latest intelligence data
            if hasattr(self.parent, 'get_latest_screenshot'):
                screenshot = self.parent.get_latest_screenshot()
                if screenshot is not None:
                    # Analyze text
                    self.current_data = osrs_text_intelligence.analyze_game_text(screenshot)
                    self.intelligence_results = text_intelligence.analyze_text_intelligence(self.current_data)
                    
                    # Update displays
                    self.update_overview_display()
                    self.update_chat_display()
                    self.update_xp_display()
                    self.update_items_display()
                    self.update_alerts_display()
        
        except Exception as e:
            logger.error(f"Failed to update text intelligence display: {e}")
        
        # Schedule next update
        self.after(self.update_interval, self.update_display)
    
    def update_overview_display(self):
        """Update the overview tab"""
        if not self.intelligence_results:
            return
        
        # Update session stats
        session_data = text_intelligence.get_session_summary()
        
        # Format duration
        duration = session_data.get('session_duration', 0)
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        duration_str = f"{hours:02d}:{minutes:02d}"
        
        # Update stat labels
        stats_updates = {
            "Session Duration": duration_str,
            "Total XP Gained": f"{session_data.get('total_session_xp', 0):,}",
            "Messages Processed": f"{len(self.current_data.get('chat_messages', []))}",
            "Items Detected": f"{len(self.current_data.get('items', []))}",
            "Alerts Generated": f"{len(self.intelligence_results.get('alerts', []))}",
            "Processing Rate": f"{1000/self.intelligence_results.get('processing_time', 1):.1f} Hz"
        }
        
        for stat_name, value in stats_updates.items():
            if stat_name in self.stat_labels:
                self.stat_labels[stat_name].configure(text=f"{stat_name}: {value}")
        
        # Update performance display
        perf_stats = osrs_text_intelligence.get_performance_stats()
        perf_text = f"""Processing Performance:
Average Latency: {perf_stats.get('avg_latency', 0):.3f}s
Cache Efficiency: {perf_stats.get('cache_efficiency', 0):.1%}
Total OCR Calls: {perf_stats.get('total_ocr_calls', 0):,}
GPU Enabled: {perf_stats.get('gpu_enabled', False)}

Intelligence Analysis:
XP Events: {len(self.intelligence_results.get('xp_analysis', {}).get('events', []))}
Combat Events: {len(self.intelligence_results.get('combat_analysis', {}).get('events', []))}
Trade Events: {len(self.intelligence_results.get('market_analysis', {}).get('trade_events', []))}
"""
        
        self.performance_display.delete(1.0, tk.END)
        self.performance_display.insert(1.0, perf_text)
    
    def update_chat_display(self):
        """Update the chat tab"""
        if not self.current_data.get('chat_messages'):
            return
        
        # Filter messages based on selected filters
        filtered_messages = self.filter_chat_messages(self.current_data['chat_messages'])
        
        # Display recent messages
        chat_text = ""
        for msg in filtered_messages[-20:]:  # Show last 20 messages
            timestamp = datetime.fromtimestamp(msg.timestamp).strftime("%H:%M:%S")
            if msg.is_system:
                chat_text += f"[{timestamp}] {msg.message}\n"
            else:
                chat_text += f"[{timestamp}] {msg.player_name}: {msg.message}\n"
        
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.insert(1.0, chat_text)
        self.chat_display.see(tk.END)
    
    def update_xp_display(self):
        """Update the XP tracking tab"""
        if not self.intelligence_results.get('xp_analysis'):
            return
        
        xp_analysis = self.intelligence_results['xp_analysis']
        session_data = text_intelligence.get_session_summary()
        
        # Update XP summary
        summary_text = f"""Session XP Summary:
Total XP Gained: {session_data.get('total_session_xp', 0):,}
Average XP/Hour: {session_data.get('avg_xp_per_hour', 0):,.0f}
Skills Trained: {session_data.get('skills_trained', 0)}
Most Trained: {session_data.get('most_trained_skill', 'None')}
"""
        
        self.xp_summary_display.delete(1.0, tk.END)
        self.xp_summary_display.insert(1.0, summary_text)
        
        # Update XP rates table
        for item in self.xp_tree.get_children():
            self.xp_tree.delete(item)
        
        session_xp = session_data.get('session_xp', {})
        xp_rates = session_data.get('xp_rates', {})
        
        for skill in session_xp.keys():
            session_gained = session_xp.get(skill, 0)
            hourly_rate = xp_rates.get(skill, 0)
            
            self.xp_tree.insert("", "end", values=(
                skill.title(),
                f"{session_gained:,}",
                f"{hourly_rate:,.0f}",
                "Est. Level",  # Would need level calculation
                "Est. Time"    # Would need time calculation
            ))
    
    def update_items_display(self):
        """Update the items tab"""
        if not self.current_data.get('items'):
            return
        
        items = self.current_data['items']
        total_value = sum(item.ge_price * item.quantity for item in items if item.ge_price)
        valuable_items = [item for item in items if item.is_valuable]
        
        # Update value summary
        value_updates = {
            "Total Value": f"{total_value:,} gp",
            "Valuable Items": f"{len(valuable_items)}",
            "Average Item Value": f"{total_value // len(items) if items else 0:,} gp",
            "Most Valuable": max(items, key=lambda x: x.ge_price or 0).name if items else "None"
        }
        
        for stat_name, value in value_updates.items():
            if stat_name in self.value_labels:
                self.value_labels[stat_name].configure(text=f"{stat_name}: {value}")
        
        # Update items table
        for item in self.items_tree.get_children():
            self.items_tree.delete(item)
        
        for item in items:
            unit_price = item.ge_price or 0
            total_item_value = unit_price * item.quantity
            
            self.items_tree.insert("", "end", values=(
                item.name,
                f"{item.quantity:,}",
                f"{unit_price:,} gp",
                f"{total_item_value:,} gp",
                item.category.title()
            ))
    
    def update_alerts_display(self):
        """Update the alerts tab"""
        if not self.intelligence_results.get('alerts'):
            return
        
        alerts = self.intelligence_results['alerts']
        
        # Display recent alerts
        alerts_text = ""
        for alert in alerts[-10:]:  # Show last 10 alerts
            timestamp = datetime.fromtimestamp(alert['timestamp']).strftime("%H:%M:%S")
            priority = alert.get('priority', TextPriority.LOW)
            priority_symbol = {
                TextPriority.CRITICAL: "🔴",
                TextPriority.HIGH: "🟠",
                TextPriority.MEDIUM: "🟡",
                TextPriority.LOW: "🟢"
            }.get(priority, "🔵")
            
            alerts_text += f"[{timestamp}] {priority_symbol} {alert['message']}\n"
        
        self.alerts_display.delete(1.0, tk.END)
        self.alerts_display.insert(1.0, alerts_text)
        self.alerts_display.see(tk.END)
    
    def filter_chat_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Filter chat messages based on selected filters"""
        filtered = []
        
        for msg in messages:
            # Check filter options
            if msg.chat_type == 'public' and self.filter_options['Public'].get():
                filtered.append(msg)
            elif msg.chat_type.startswith('private') and self.filter_options['Private'].get():
                filtered.append(msg)
            elif msg.chat_type == 'game' and self.filter_options['Game Messages'].get():
                filtered.append(msg)
            elif msg.is_system and self.filter_options['System'].get():
                filtered.append(msg)
        
        # Important only filter
        if self.filter_options['Important Only'].get():
            important_keywords = ['level', 'died', 'rare', 'trade', 'attack']
            filtered = [msg for msg in filtered 
                       if any(keyword in msg.message.lower() for keyword in important_keywords)]
        
        return filtered
    
    def update_chat_filter(self):
        """Update chat display when filters change"""
        if self.is_active:
            self.update_chat_display()
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts_display.delete(1.0, tk.END)


class LiveTextOverlay(ctk.CTkFrame):
    """
    Live text overlay widget for displaying text on game feed
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.overlay_items = []
        self.is_visible = True
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup overlay UI"""
        # Overlay controls
        controls_frame = ctk.CTkFrame(self)
        controls_frame.pack(fill="x", padx=5, pady=5)
        
        self.visibility_toggle = ctk.CTkCheckBox(
            controls_frame,
            text="Show Overlay",
            command=self.toggle_visibility
        )
        self.visibility_toggle.pack(side="left", padx=5)
        self.visibility_toggle.select()
        
        # Overlay canvas
        self.overlay_canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.overlay_canvas.pack(fill="both", expand=True, padx=5, pady=5)
    
    def toggle_visibility(self):
        """Toggle overlay visibility"""
        self.is_visible = self.visibility_toggle.get()
        if not self.is_visible:
            self.overlay_canvas.delete("all")
    
    def update_overlay(self, text_data: Dict[str, Any]):
        """Update overlay with new text data"""
        if not self.is_visible:
            return
        
        # Clear previous overlay
        self.overlay_canvas.delete("all")
        
        # Add text overlays for important items
        y_offset = 20
        
        # Health warning
        if 'player_stats' in text_data:
            stats = text_data['player_stats']
            if stats.health_current and stats.health_max:
                health_percent = (stats.health_current / stats.health_max) * 100
                if health_percent < 30:
                    color = "red" if health_percent < 20 else "orange"
                    self.overlay_canvas.create_text(
                        10, y_offset, text=f"Health: {health_percent:.0f}%",
                        fill=color, font=("Arial", 12, "bold"), anchor="w"
                    )
                    y_offset += 25
        
        # Valuable items
        if 'items' in text_data:
            valuable_items = [item for item in text_data['items'] if item.is_valuable]
            if valuable_items:
                total_value = sum(item.ge_price * item.quantity for item in valuable_items if item.ge_price)
                self.overlay_canvas.create_text(
                    10, y_offset, text=f"Valuable Items: {total_value:,} gp",
                    fill="gold", font=("Arial", 10), anchor="w"
                )
                y_offset += 20
        
        # XP rates
        session_data = text_intelligence.get_session_summary()
        if session_data.get('avg_xp_per_hour', 0) > 0:
            self.overlay_canvas.create_text(
                10, y_offset, text=f"XP/Hour: {session_data['avg_xp_per_hour']:,.0f}",
                fill="cyan", font=("Arial", 10), anchor="w"
            )


# Export main widget class
__all__ = ['TextIntelligencePanel', 'LiveTextOverlay']