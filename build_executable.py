#!/usr/bin/env python3
"""
Script to build executable from the GUI application using PyInstaller.
"""

import os
import sys
import subprocess

def build_executable():
    """Build executable using PyInstaller."""
    
    print("Building executable for Colorization GUI...")
    print("=" * 50)
    
    # Check if PyInstaller is installed by checking if the command exists
    try:
        # Try to run pyinstaller --version to check if it's installed
        subprocess.run(
            ["pyinstaller", "--version"],
            capture_output=True,
            check=True,
            timeout=5
        )
        print("✓ PyInstaller is installed")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("PyInstaller is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✓ PyInstaller installed successfully!")
    
    # PyInstaller command
    # Note: colorize_gui.py is standalone and doesn't need colorize.py
    cmd = [
        "pyinstaller",
        "--onefile",                    # Create a single executable file
        "--windowed",                   # No console window (GUI only)
        "--name=ImageColorizer",        # Name of the executable
        "--icon=NONE",                  # No icon (you can add one later)
        "colorize_gui.py"               # Main GUI script (standalone)
    ]
    
    # Alternative: if you want to include model files (makes executable larger)
    # Uncomment these lines if you want models bundled:
    # if os.path.exists("models"):
    #     cmd.insert(-1, "--add-data=models;models")
    
    print("Running PyInstaller...")
    print("Command:", " ".join(cmd))
    print()
    
    try:
        subprocess.check_call(cmd)
        print()
        print("=" * 50)
        print("✓ Build successful!")
        print()
        print("Executable location: dist/ImageColorizer.exe")
        print()
        print("Note: The executable will download model files on first run.")
        print("To include models in the executable, uncomment the model bundling")
        print("lines in this script and rebuild.")
    except subprocess.CalledProcessError as e:
        print(f"Error building executable: {e}")
        sys.exit(1)


if __name__ == '__main__':
    build_executable()

