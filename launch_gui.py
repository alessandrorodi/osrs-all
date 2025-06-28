#!/usr/bin/env python3
"""
OSRS Bot Framework - GUI Launcher

Simple launcher for the GUI application.
"""

if __name__ == "__main__":
    try:
        from gui.gui_app import main
        main()
    except ImportError:
        print("❌ GUI dependencies not found!")
        print("Please install required packages:")
        print("pip install --user customtkinter pillow matplotlib")
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        input("Press Enter to exit...") 