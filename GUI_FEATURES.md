# OSRS Bot Framework - GUI Documentation

## 🖥️ **Modern GUI Interface**

The OSRS Bot Framework includes a comprehensive graphical user interface built with CustomTkinter for a modern, dark-themed experience.

## 🚀 **Launching the GUI**

### Quick Launch
```bash
python launch_gui.py
```

### Direct Launch
```bash
python gui/gui_app.py
```

## 📱 **Interface Overview**

### **Main Window Features:**
- **Dark Theme**: Modern, eye-friendly interface
- **Tabbed Layout**: Organized sections for different functions
- **Real-time Monitoring**: Live updates of system status
- **Responsive Design**: Scales to different screen sizes

---

## 🏠 **Dashboard Tab**

### **System Status Panel**
- **🔴/🟢 Client Status**: OSRS client connection status
- **🔴/🟢 Screen Capture**: Screen capture system status
- **🟡 Computer Vision**: CV system readiness
- **🤖 Active Bots**: Count of running/total bots

### **Quick Statistics**
- Real-time performance metrics
- System health indicators
- Recent activity summary

### **Emergency Controls**
- **🛑 Emergency Stop**: Immediately stops all running bots
- Safety override for critical situations

---

## 🤖 **Bots Tab**

### **Bot Management**
- **➕ New Bot**: Create a new bot instance
- **📂 Load Bot**: Load saved bot configurations
- **🔄 Refresh**: Update bot list display

### **Bot List View**
- **Name**: Bot identifier
- **Status**: Current state (idle, running, paused, etc.)
- **Runtime**: How long the bot has been running
- **Actions**: Total actions performed
- **Errors**: Error count
- **Performance**: Actions per minute (APM)

### **Bot Controls**
- **▶️ Start**: Begin bot execution
- **⏸️ Pause**: Temporarily pause bot
- **⏹️ Stop**: Stop bot completely
- **📊 Details**: View detailed bot information

---

## 👁️ **Vision Tab**

### **Computer Vision Testing**
- **📷 Capture Screen**: Take screenshot of current game state
- **🔍 Test Detection**: Test template matching and object detection
- **🎯 Live View**: Real-time computer vision overlay

### **Visual Display**
- Live image feed from OSRS client
- Detection overlay showing found objects
- Confidence scores and bounding boxes

### **Detection Results**
- Text output of detection results
- Performance metrics
- Debug information

---

## 📊 **Performance Tab**

### **Real-time Monitoring**
- **▶️ Start/Stop Monitoring**: Toggle performance tracking
- **🔄 Refresh Charts**: Update chart displays
- **💾 Export Data**: Save performance data to file

### **Performance Charts**
1. **Actions Per Minute**: Bot efficiency comparison
2. **Success Rate**: Percentage of successful actions
3. **Runtime**: How long each bot has been active
4. **Error Rate**: Errors per hour for each bot

### **Chart Features**
- Dark theme matplotlib integration
- Real-time updates
- Multi-bot comparison
- Export capabilities

---

## 📋 **Logs Tab**

### **Log Management**
- **🔄 Refresh**: Update log display
- **🗑️ Clear**: Clear current log view
- **💾 Export**: Save logs to file

### **Log Filtering**
- **Level Filter**: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Real-time Updates**: Logs update automatically
- **Search Functionality**: Find specific log entries

### **Log Display**
- Syntax highlighting for different log levels
- Timestamp information
- Source component identification
- Scrollable view with search

---

## 🖼️ **Templates Tab**

### **Template Management**
- **➕ Create Template**: Launch template creator tool
- **📂 Import Template**: Import existing template files
- **🔄 Refresh**: Update template list

### **Template Browser**
- **Template List**: All available templates
- **Preview Panel**: Visual preview of selected template
- **Template Info**: Size, creation date, usage statistics

### **Template Actions**
- Click template to preview
- Double-click to edit
- Right-click for context menu
- Drag and drop support

---

## 🎛️ **Header Controls**

### **Quick Actions**
- **🎯 Calibrate Client**: Launch client calibration tool
- **📸 Create Templates**: Open template creation tool
- **⚙️ Settings**: Access framework settings

### **System Information**
- Python version indicator
- OpenCV availability status
- Framework version display

---

## 📊 **Status Bar**

### **Real-time Status**
- Current operation status
- System messages and notifications
- Progress indicators for long operations

### **System Info**
- Python version
- Computer vision status
- Framework version

---

## 🔧 **Settings & Configuration**

### **Planned Features** (Coming Soon)
- **Bot Configuration Editor**: Visual bot configuration
- **Framework Settings**: Adjust system parameters
- **Theme Selection**: Light/dark theme options
- **Hotkey Configuration**: Customize keyboard shortcuts
- **Performance Tuning**: Optimize system performance

---

## 🎮 **Usage Workflow**

### **First Time Setup**
1. **Launch GUI**: `python launch_gui.py`
2. **Check Dashboard**: Verify system status is green
3. **Calibrate Client**: Click "🎯 Calibrate Client"
4. **Create Templates**: Click "📸 Create Templates"
5. **Test System**: Use Vision tab to test detection

### **Daily Usage**
1. **Open GUI**: Launch the interface
2. **Check Status**: Verify OSRS client connection
3. **Load/Create Bot**: Use Bots tab to manage bots
4. **Monitor Performance**: Watch real-time charts
5. **Review Logs**: Check for any issues

### **Bot Development**
1. **Use Vision Tab**: Test computer vision components
2. **Create Templates**: Capture game objects
3. **Test Detection**: Verify accuracy
4. **Monitor Performance**: Track bot efficiency
5. **Debug Issues**: Use logs and monitoring

---

## 🛡️ **Safety Features**

### **Emergency Controls**
- **Emergency Stop Button**: Stops all bots immediately
- **Fail-safe Mechanisms**: Automatic safety triggers
- **Status Monitoring**: Continuous system health checks

### **Monitoring & Alerts**
- **Performance Tracking**: Detect unusual behavior
- **Error Detection**: Alert on system issues
- **Resource Monitoring**: Track CPU/memory usage

---

## 🔮 **Future Enhancements**

### **Planned Features**
- **Visual Bot Builder**: Drag-and-drop bot creation
- **Advanced Analytics**: ML-powered performance analysis
- **Remote Monitoring**: Web-based monitoring dashboard
- **Plugin System**: Community-developed extensions
- **AI Assistant**: Intelligent bot configuration helper

### **Advanced Features**
- **Multi-Client Support**: Manage multiple OSRS accounts
- **Cloud Integration**: Sync configurations across devices
- **Advanced Debugging**: Visual debugging tools
- **Performance Optimization**: Automatic tuning recommendations

---

## 🎯 **Keyboard Shortcuts** (Planned)

| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | New Bot |
| `Ctrl+O` | Load Bot |
| `Ctrl+S` | Save Configuration |
| `F5` | Refresh Current Tab |
| `Ctrl+Shift+E` | Emergency Stop All |
| `Ctrl+,` | Open Settings |
| `F11` | Toggle Fullscreen |
| `Ctrl+D` | Open Dashboard |

---

**The GUI makes the OSRS Bot Framework accessible to both beginners and advanced users, providing a comprehensive interface for bot management and monitoring!** 🚀 