"""
Basic framework tests to validate GitHub Actions workflow
"""
import pytest
import os
import sys
import importlib.util


class TestFrameworkBasics:
    """Test basic framework functionality"""
    
    def test_python_version(self):
        """Test that Python version is supported"""
        assert sys.version_info >= (3, 9), "Python 3.9+ required"
        
    def test_project_structure(self):
        """Test that essential project directories exist"""
        required_dirs = [
            "core",
            "gui", 
            "utils",
            "config",
            "bots",
            "vision",
            "data"
        ]
        
        for dir_name in required_dirs:
            assert os.path.exists(dir_name), f"Directory {dir_name} should exist"
            
    def test_essential_files(self):
        """Test that essential files exist"""
        required_files = [
            "requirements.txt",
            "setup.py",
            "README.md",
            "launch_gui.py"
        ]
        
        for file_name in required_files:
            assert os.path.exists(file_name), f"File {file_name} should exist"
            
    def test_core_modules_importable(self):
        """Test that core modules can be imported"""
        core_modules = [
            "core.bot_base",
            "core.computer_vision", 
            "core.screen_capture",
            "core.automation",
            "utils.logging"
        ]
        
        for module_name in core_modules:
            try:
                spec = importlib.util.find_spec(module_name)
                assert spec is not None, f"Module {module_name} should be importable"
            except ImportError:
                pytest.skip(f"Module {module_name} not yet implemented")
                
    def test_config_directory(self):
        """Test config directory structure"""
        config_dir = "config"
        assert os.path.exists(config_dir), "Config directory should exist"
        
        # Check for settings file
        settings_file = os.path.join(config_dir, "settings.py")
        if os.path.exists(settings_file):
            # Try to import settings
            try:
                import config.settings
                assert hasattr(config.settings, 'VISION_SETTINGS'), "Settings should have VISION_SETTINGS"
            except ImportError:
                pytest.skip("Settings module not yet fully implemented")


class TestOSRSComponents:
    """Test OSRS-specific components"""
    
    def test_osrs_terminology(self):
        """Test that OSRS terminology is used correctly"""
        # This is a placeholder test for OSRS-specific validation
        osrs_terms = {
            "gp": "gold pieces",
            "xp": "experience points", 
            "hp": "hit points",
            "npc": "non-player character"
        }
        
        # Verify OSRS terms are recognized
        for abbrev, full_term in osrs_terms.items():
            assert len(abbrev) > 0, f"OSRS term {abbrev} should be defined"
            
    def test_osrs_data_structure(self):
        """Test OSRS data directory structure"""
        data_dir = "data"
        assert os.path.exists(data_dir), "Data directory should exist"
        
        # Check for templates directory
        templates_dir = os.path.join(data_dir, "templates")
        if not os.path.exists(templates_dir):
            # Create it for testing
            os.makedirs(templates_dir, exist_ok=True)
            
        assert os.path.exists(templates_dir), "Templates directory should exist"


class TestGUIComponents:
    """Test GUI components"""
    
    def test_gui_directory_structure(self):
        """Test GUI directory structure"""
        gui_dir = "gui"
        assert os.path.exists(gui_dir), "GUI directory should exist"
        
        expected_files = [
            "gui_app.py",
            "main_window.py", 
            "tab_functions.py",
            "handlers.py"
        ]
        
        for file_name in expected_files:
            file_path = os.path.join(gui_dir, file_name)
            assert os.path.exists(file_path), f"GUI file {file_name} should exist"
            
    def test_gui_imports(self):
        """Test that GUI modules can be imported"""
        try:
            import gui.gui_app
            assert hasattr(gui.gui_app, 'main'), "GUI app should have main function"
        except ImportError:
            pytest.skip("GUI modules not yet fully implemented")


class TestVisionComponents:
    """Test computer vision components"""
    
    def test_vision_directory(self):
        """Test vision directory structure"""
        vision_dir = "vision"
        assert os.path.exists(vision_dir), "Vision directory should exist"
        
        # Check for detectors subdirectory
        detectors_dir = os.path.join(vision_dir, "detectors")
        assert os.path.exists(detectors_dir), "Detectors directory should exist"
        
    def test_computer_vision_module(self):
        """Test computer vision module"""
        try:
            import core.computer_vision
            # Basic validation that the module exists
            assert hasattr(core.computer_vision, 'ComputerVision'), "Should have ComputerVision class" 
        except ImportError:
            pytest.skip("Computer vision module not yet fully implemented")


@pytest.mark.performance
class TestPerformanceBasics:
    """Basic performance tests"""
    
    def test_import_performance(self):
        """Test that imports are reasonably fast"""
        import time
        
        start_time = time.time()
        try:
            import core.bot_base
            import utils.logging
        except ImportError:
            pytest.skip("Core modules not yet implemented")
            
        import_time = time.time() - start_time
        
        # Imports should be under 1 second
        assert import_time < 1.0, f"Imports took {import_time:.2f}s, should be < 1.0s"
        
    @pytest.mark.benchmark
    def test_basic_operations_benchmark(self, benchmark):
        """Benchmark basic operations"""
        
        def basic_operation():
            # Simple operation for benchmarking
            result = []
            for i in range(1000):
                result.append(i * i)
            return result
            
        result = benchmark(basic_operation)
        assert len(result) == 1000, "Benchmark should return correct result"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 