#!/usr/bin/env python3
"""
OSRS Bot Framework - GUI Application Launcher

Complete GUI application combining all components.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the main window and add methods
from gui.main_window import OSRSBotGUI
from gui import tabs, handlers

# Add tab creation methods to OSRSBotGUI class
OSRSBotGUI.create_dashboard_tab = tabs.create_dashboard_tab
OSRSBotGUI.create_bots_tab = tabs.create_bots_tab
OSRSBotGUI.create_vision_tab = tabs.create_vision_tab
OSRSBotGUI.create_performance_tab = tabs.create_performance_tab
OSRSBotGUI.create_logs_tab = tabs.create_logs_tab
OSRSBotGUI.create_templates_tab = tabs.create_templates_tab

# Add handler methods to OSRSBotGUI class
OSRSBotGUI.setup_monitoring = handlers.setup_monitoring
OSRSBotGUI.monitoring_loop = handlers.monitoring_loop
OSRSBotGUI.update_system_status = handlers.update_system_status
OSRSBotGUI.update_bot_list = handlers.update_bot_list
OSRSBotGUI.update_performance_charts = handlers.update_performance_charts
OSRSBotGUI.calibrate_client = handlers.calibrate_client
OSRSBotGUI._run_calibration = handlers._run_calibration
OSRSBotGUI.create_templates = handlers.create_templates
OSRSBotGUI._run_template_creator = handlers._run_template_creator
OSRSBotGUI.emergency_stop_all = handlers.emergency_stop_all
OSRSBotGUI.update_status = handlers.update_status
OSRSBotGUI.run = handlers.run
OSRSBotGUI.cleanup = handlers.cleanup

# Add placeholder methods
OSRSBotGUI.open_settings = handlers.open_settings
OSRSBotGUI.create_new_bot = handlers.create_new_bot
OSRSBotGUI.load_bot = handlers.load_bot
OSRSBotGUI.refresh_bots = handlers.refresh_bots
OSRSBotGUI.start_selected_bot = handlers.start_selected_bot
OSRSBotGUI.pause_selected_bot = handlers.pause_selected_bot
OSRSBotGUI.stop_selected_bot = handlers.stop_selected_bot
OSRSBotGUI.show_bot_details = handlers.show_bot_details
OSRSBotGUI.capture_screen = handlers.capture_screen
OSRSBotGUI.test_detection = handlers.test_detection
OSRSBotGUI.toggle_live_view = handlers.toggle_live_view
OSRSBotGUI.toggle_monitoring = handlers.toggle_monitoring
OSRSBotGUI.refresh_charts = handlers.refresh_charts
OSRSBotGUI.export_performance_data = handlers.export_performance_data
OSRSBotGUI.refresh_logs = handlers.refresh_logs
OSRSBotGUI.clear_logs = handlers.clear_logs
OSRSBotGUI.export_logs = handlers.export_logs
OSRSBotGUI.filter_logs = handlers.filter_logs
OSRSBotGUI.create_template = handlers.create_template
OSRSBotGUI.import_template = handlers.import_template
OSRSBotGUI.refresh_templates = handlers.refresh_templates
OSRSBotGUI.on_template_select = handlers.on_template_select


def main():
    """Main entry point for GUI application"""
    print("🤖 OSRS Bot Framework - Ultimate AI Agent")
    print("=" * 50)
    print("Starting GUI application...")
    
    try:
        # Create and run the GUI application
        app = OSRSBotGUI()
        app.run()
    except Exception as e:
        print(f"❌ Failed to start GUI: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed:")
        print("   pip install --user customtkinter pillow matplotlib")
        print("2. Check that Python 3.8+ is being used")
        print("3. Ensure the framework setup was completed successfully")
        sys.exit(1)


if __name__ == "__main__":
    main() 