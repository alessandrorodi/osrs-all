# OSRS Bot Framework

A modular, extensible framework for creating Old School RuneScape automation bots using computer vision and machine learning techniques.

## Features

- **Modular Architecture**: Easy to create new bots by extending base classes
- **Computer Vision**: Advanced image recognition using OpenCV and template matching
- **Safety Features**: Anti-detection mechanisms and failsafes
- **Visual Debugging**: Real-time overlay and debugging tools
- **Configurable**: YAML-based configuration system
- **Extensible**: Plugin-based architecture for custom detectors

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd osrs-bot-framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run initial setup:
```bash
python setup.py
```

## Quick Start

### Option 1: GUI Interface (Recommended)
```bash
python launch_gui.py
```
Use the modern graphical interface to:
- Monitor system status in real-time
- Manage bots visually
- Test computer vision
- View performance charts
- Browse logs and templates

### Option 2: Command Line
1. **Calibrate your client**: Run the calibration tool to detect your OSRS client
2. **Create templates**: Use the template creator to capture UI elements
3. **Configure your bot**: Edit configuration files for your specific needs
4. **Run your bot**: Execute with built-in safety features

## Project Structure

```
osrs-bot-framework/
├── core/              # Core framework components
├── vision/            # Computer vision modules
├── bots/              # Bot implementations
├── gui/               # Modern GUI interface
├── config/            # Configuration files
├── utils/             # Utility functions
├── tools/             # Development and debugging tools
├── data/              # Templates, models, and data files
└── launch_gui.py      # GUI launcher script
```

## Safety and Ethics

This framework is designed for educational purposes and personal use. Please:
- Respect Jagex's Terms of Service
- Use responsibly and at your own risk
- Consider the impact on other players
- Always include proper failsafes and breaks

## License

This project is for educational and personal use only. 