# How to Run the Colorization Program

## Quick Start

### 1. Make sure dependencies are installed
```bash
pip install -r requirements.txt
```
*(You've already done this!)*

### 2. Colorize a single image

**Basic command:**
```bash
python colorize.py input.jpg output.jpg
```

**Example:**
```bash
# If your image is called "old_photo.jpg"
python colorize.py old_photo.jpg colorized_photo.jpg
```

**Auto-named output (saves as "colorized_input.jpg"):**
```bash
python colorize.py old_photo.jpg
# Creates: colorized_old_photo.jpg in the same folder
```

### 3. Process multiple images (batch mode)

**Process all images in a folder:**
```bash
python colorize.py --batch ./input_folder/ ./output_folder/
```

**Auto output folder (saves to input_folder/colorized_output/):**
```bash
python colorize.py --batch ./input_folder/
```

## Step-by-Step Example

1. **Put your black and white image in the project folder**
   - For example: `photo.jpg`

2. **Open terminal in the project folder**
   - You're already there: `c:\Users\josea\Desktop\Cursor\B&W project`

3. **Run the command:**
   ```bash
   python colorize.py photo.jpg colorized_photo.jpg
   ```

4. **First time only:** Wait for models to download (~170MB)
   - You'll see: "Downloading Prototxt file..."
   - Then: "Downloading Caffe model file..." (this is the big one)
   - Finally: "Downloading Cluster centers file..."

5. **Wait for processing:**
   - You'll see: "Processing: photo.jpg"
   - Then: "✓ Colorized image saved to: colorized_photo.jpg"

6. **Done!** Check the `colorized_photo.jpg` file

## What to Expect

**First run:**
- Downloads model files (~170MB) - takes a few minutes
- Processes your image - takes 1-5 seconds
- Saves the colorized result

**Subsequent runs:**
- No downloads needed
- Just processes images - very fast!

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WEBP (.webp)

## Tips

- **Best results:** Works best with natural photographs (people, landscapes, objects)
- **Image quality:** Higher resolution images may take longer but often look better
- **Batch processing:** Great for processing multiple old photos at once
- **Output location:** You can specify any output path, not just the current folder

## Troubleshooting

**"File not found" error:**
- Make sure the image file exists
- Check the file path is correct
- Use quotes if the path has spaces: `python colorize.py "my photo.jpg" "output.jpg"`

**"Could not read image" error:**
- Make sure it's a valid image file
- Check the file isn't corrupted
- Try a different image format

**Slow processing:**
- Large images take longer
- First run is slower (downloading models)
- Normal processing: 1-5 seconds per image

## Example Commands

```bash
# Single image with full path
python colorize.py "C:\Users\josea\Pictures\old_photo.jpg" "C:\Users\josea\Pictures\colorized.jpg"

# Multiple images in a folder
python colorize.py --batch "C:\Users\josea\Pictures\old_photos\" "C:\Users\josea\Pictures\colorized\"

# Simple - just the filename (if image is in current folder)
python colorize.py photo.jpg result.jpg
```

