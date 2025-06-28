#!/usr/bin/env python3
"""
Test Runner for OSRS Bot Framework

Simple wrapper around pytest for common test scenarios.
"""

import sys
import subprocess


def run_command(cmd):
    """Run a command and return success status"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    """Main test runner function"""
    if len(sys.argv) < 2:
        print("OSRS Bot Framework - Test Runner")
        print("=" * 40)
        print("Usage: python run_tests.py <command>")
        print()
        print("Available commands:")
        print("  all        - Run all tests")
        print("  unit       - Run unit tests only")
        print("  coverage   - Run all tests with coverage report")
        print("  fast       - Run tests excluding slow ones")
        print("  verbose    - Run all tests with verbose output")
        print()
        print("Examples:")
        print("  python run_tests.py all")
        print("  python run_tests.py coverage")
        print("  python run_tests.py unit")
        return 1

    command = sys.argv[1].lower()
    
    base_cmd = [sys.executable, "-m", "pytest"]
    
    if command == "all":
        cmd = base_cmd + ["tests/"]
    elif command == "unit":
        cmd = base_cmd + ["-m", "unit", "tests/"]
    elif command == "coverage":
        cmd = base_cmd + ["tests/", "--cov=core", "--cov=gui", "--cov=utils", "--cov=config", "--cov=bots", "--cov-report=html", "--cov-report=term"]
    elif command == "fast":
        cmd = base_cmd + ["-m", "not slow", "tests/"]
    elif command == "verbose":
        cmd = base_cmd + ["tests/", "-v"]
    else:
        print(f"Unknown command: {command}")
        return 1
    
    success = run_command(cmd)
    
    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 