"""
GUI Event Handlers and Utility Methods
"""

import threading
import subprocess
import sys
from pathlib import Path
from core.screen_capture import screen_capture
from core.bot_base import BotState
from utils.logging import setup_logging

logger = setup_logging("gui_handlers")
project_root = Path(__file__).parent.parent


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


def calibrate_client(self):
    """Launch client calibration"""
    threading.Thread(target=self._run_calibration, daemon=True).start()


def _run_calibration(self):
    """Run calibration in background thread"""
    try:
        # Use built-in screen capture calibration instead of external tool
        success = screen_capture.calibrate_client()
        
        if success:
            self.update_status("✅ Client calibration successful!")
            # Test screenshot to verify
            screenshot = screen_capture.capture_client()
            if screenshot is not None:
                self.update_status(f"✅ Screenshot test passed: {screenshot.shape}")
            else:
                self.update_status("⚠️ Calibration successful but screenshot test failed")
        else:
            self.update_status("❌ Client calibration failed - check if RuneLite is open")
    except Exception as e:
        logger.error(f"Calibration error: {e}")
        self.update_status(f"❌ Calibration error: {e}")


def create_templates(self):
    """Launch template creator"""
    threading.Thread(target=self._run_template_creator, daemon=True).start()


def _run_template_creator(self):
    """Run template creator in background thread"""
    try:
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
def open_settings(self): 
    self.update_status("⚙️ Settings panel coming soon...")

def create_new_bot(self): 
    self.update_status("➕ Bot creator coming soon...")

def load_bot(self): 
    self.update_status("📂 Bot loader coming soon...")

def refresh_bots(self): 
    self.update_status("🔄 Refreshing bot list...")

def start_selected_bot(self): 
    self.update_status("▶️ Starting selected bot...")

def pause_selected_bot(self): 
    self.update_status("⏸️ Pausing selected bot...")

def stop_selected_bot(self): 
    self.update_status("⏹️ Stopping selected bot...")

def show_bot_details(self): 
    self.update_status("📊 Bot details coming soon...")

def capture_screen(self): 
    self.update_status("📷 Screen capture functionality coming soon...")

def test_detection(self): 
    self.update_status("🔍 Detection test coming soon...")

def toggle_live_view(self): 
    self.update_status("🎯 Live view coming soon...")

def toggle_monitoring(self):
    self.is_monitoring = not self.is_monitoring
    status = "▶️ Start Monitoring" if not self.is_monitoring else "⏸️ Stop Monitoring"
    self.monitoring_button.configure(text=status)
    self.update_status(f"📊 Monitoring {'started' if self.is_monitoring else 'stopped'}")

def refresh_charts(self): 
    self.update_status("🔄 Refreshing performance charts...")

def export_performance_data(self): 
    self.update_status("💾 Export functionality coming soon...")

def refresh_logs(self): 
    self.update_status("🔄 Refreshing logs...")

def clear_logs(self): 
    self.update_status("🗑️ Clearing logs...")

def export_logs(self): 
    self.update_status("💾 Export logs coming soon...")

def filter_logs(self, level): 
    self.update_status(f"🔍 Filtering logs by {level}...")

def create_template(self): 
    self.create_templates()

def import_template(self): 
    self.update_status("📂 Template import coming soon...")

def refresh_templates(self): 
    self.update_status("🔄 Refreshing templates...")

def on_template_select(self, event): 
    self.update_status("🖼️ Template selected...") 