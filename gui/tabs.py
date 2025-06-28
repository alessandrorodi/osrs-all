"""
GUI Tab Creation Methods
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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