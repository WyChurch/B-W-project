# Building an Executable from the GUI

This guide explains how to create a standalone executable (.exe) file from the GUI application.

## Option 1: Quick Build (Recommended)

### Step 1: Install PyInstaller
```bash
pip install pyinstaller
```

### Step 2: Run the build script
```bash
python build_executable.py
```

This will create `dist/ImageColorizer.exe` - a standalone executable that you can run without Python installed!

## Option 2: Manual Build

### Step 1: Install PyInstaller
```bash
pip install pyinstaller
```

### Step 2: Build the executable
```bash
pyinstaller --onefile --windowed --name=ImageColorizer colorize_gui.py
```

### Step 3: Find your executable
The executable will be in the `dist/` folder: `dist/ImageColorizer.exe`

## Option 3: Include Model Files (Larger but Complete)

If you want to bundle the model files with the executable (so users don't need to download them):

### Step 1: Download models first
Run the GUI once to download models:
```bash
python colorize_gui.py
```

### Step 2: Build with models included
```bash
pyinstaller --onefile --windowed --name=ImageColorizer --add-data="models;models" colorize_gui.py
```

**Note:** This will make the executable much larger (~200MB+) but users won't need internet to download models.

## Advanced Options

### Create a custom icon
1. Create or download an `.ico` file
2. Use the `--icon` option:
```bash
pyinstaller --onefile --windowed --icon=icon.ico --name=ImageColorizer colorize_gui.py
```

### Reduce executable size
The executable will be large (~100-200MB) because it includes:
- Python interpreter
- OpenCV
- NumPy
- All dependencies

To reduce size, you can:
- Use `--exclude-module` to exclude unused modules
- Use UPX compression (if available)

## Distribution

### Single File Distribution
The `--onefile` option creates a single executable that:
- ✅ Can be run without Python installed
- ✅ Can be distributed as a single file
- ⚠️ Takes longer to start (extracts to temp folder)
- ⚠️ Larger file size

### Folder Distribution (Alternative)
If you want faster startup, use folder mode:
```bash
pyinstaller --windowed --name=ImageColorizer colorize_gui.py
```

This creates a folder with the executable and dependencies. You'll need to distribute the entire folder.

## Troubleshooting

### "Failed to execute script"
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Try building without `--windowed` to see error messages
- Check that `colorize.py` is in the same directory

### Executable is very large
- This is normal! OpenCV and NumPy are large libraries
- The executable includes everything needed to run
- Consider the folder distribution option if size is a concern

### Models not found
- Models are downloaded on first run (requires internet)
- To include models: download them first, then use `--add-data` option
- Models will be stored in a `models/` folder next to the executable

### Antivirus warnings
- Some antivirus software may flag PyInstaller executables as suspicious
- This is a false positive - the executable is safe
- You may need to add an exception or sign the executable

## Testing the Executable

1. Navigate to the `dist/` folder
2. Double-click `ImageColorizer.exe`
3. Test with a black and white image
4. The first run will download model files (if not bundled)

## File Structure After Build

```
B&W project/
├── colorize_gui.py          # GUI source code
├── colorize.py              # Core colorization code
├── build_executable.py      # Build script
├── dist/
│   └── ImageColorizer.exe   # Your executable (distribute this!)
├── build/                   # Build files (can be deleted)
└── ImageColorizer.spec      # PyInstaller spec file
```

## Distribution Checklist

- [ ] Test the executable on a clean machine (without Python)
- [ ] Verify model downloads work (or bundle models)
- [ ] Test with different image formats
- [ ] Check file size (consider compression if needed)
- [ ] Create a README for end users
- [ ] Consider code signing for distribution

## Notes

- The executable will be platform-specific (Windows .exe, Mac .app, Linux binary)
- Users don't need Python installed to run the executable
- First run may be slower as models download
- The executable is self-contained and includes all dependencies

