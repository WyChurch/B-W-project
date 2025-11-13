# Example Usage

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Colorize a single image:**
   ```bash
   python colorize.py input.jpg output.jpg
   ```

3. **The first time you run it, the program will download the model files (~170MB).**

## Examples

### Single Image

```bash
# Basic usage
python colorize.py photo.jpg colorized_photo.jpg

# Output will be automatically named if not specified
python colorize.py photo.jpg
# Creates: colorized_photo.jpg in the same directory
```

### Batch Processing

```bash
# Process all images in a folder
python colorize.py --batch ./old_photos/ ./colorized_output/

# Auto output directory (saves to input_dir/colorized_output/)
python colorize.py --batch ./old_photos/
```

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WEBP (.webp)

## Tips

- **First run:** The model files will be downloaded automatically (~170MB). This only happens once.
- **Processing time:** Typically 1-5 seconds per image depending on size and hardware.
- **Best results:** Works best with natural photographs (portraits, landscapes, etc.).
- **Image quality:** Higher resolution images may take longer but often produce better results.

## Troubleshooting

If you encounter any issues:

1. **Check dependencies:**
   ```bash
   pip install opencv-python numpy
   ```

2. **Verify image file:**
   - Make sure the image file exists
   - Check that it's a valid image format
   - Ensure the file is not corrupted

3. **Check internet connection:**
   - Required for initial model download
   - After first download, works offline

4. **Memory issues:**
   - Process smaller images
   - Process one image at a time instead of batch mode

