# GUI Version - Quick Start Guide

## Running the GUI

### Option 1: Run Directly (Python Required)

1. **Make sure dependencies are installed:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the GUI:**
   ```bash
   python colorize_gui.py
   ```

3. **Use the GUI:**
   - Click "Browse..." to select an input image
   - Choose where to save the output (or use auto-generated name)
   - Click "🎨 Colorize Image" button
   - Wait for processing to complete
   - Done! Your colorized image is saved

### Option 2: Create an Executable (No Python Required)

1. **Install PyInstaller:**
   ```bash
   pip install pyinstaller
   ```

2. **Build the executable:**
   ```bash
   python build_executable.py
   ```

3. **Find your executable:**
   - Look in the `dist/` folder
   - File: `ImageColorizer.exe`
   - Double-click to run (no Python needed!)

## GUI Features

- ✅ **Simple Interface** - Easy to use, no command line needed
- ✅ **File Browser** - Click to select input/output files
- ✅ **Progress Indicator** - See what's happening
- ✅ **Status Updates** - Know when processing is done
- ✅ **Auto Output Naming** - Automatically suggests output filename
- ✅ **Error Handling** - Clear error messages if something goes wrong

## First Run

On the first run, the GUI will:
1. Check for model files
2. Download models if needed (~170MB)
3. Show progress in the status area
4. Ready to use once models are downloaded

## Usage Tips

- **Input Image:** Click "Browse..." next to "Select a black and white image"
- **Output Image:** Click "Browse..." to choose where to save, or let it auto-generate
- **Processing:** Click "🎨 Colorize Image" and wait
- **Status:** Watch the status area at the bottom for updates

## Troubleshooting

**GUI won't open:**
- Make sure tkinter is installed (usually comes with Python)
- On Linux, you may need: `sudo apt-get install python3-tk`

**Models won't download:**
- Check your internet connection
- Make sure you have ~170MB free space
- Check firewall settings

**Processing is slow:**
- Large images take longer
- First run is slower (downloading models)
- Normal processing: 1-5 seconds per image

**Error messages:**
- Read the error message in the status area
- Check that the input image exists
- Make sure output path is writable

## Building Executable

See `BUILD_EXECUTABLE.md` for detailed instructions on creating a standalone executable.

## Comparison: GUI vs Command Line

| Feature | GUI Version | Command Line Version |
|---------|-------------|---------------------|
| Ease of use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Batch processing | ❌ | ✅ |
| Automation | ❌ | ✅ |
| User-friendly | ✅ | ❌ |
| File browsing | ✅ | ❌ |
| Progress display | ✅ | ✅ |

**Use GUI if:** You want an easy-to-use interface for occasional use
**Use Command Line if:** You need batch processing or automation

