# Black and White Image Colorization

A Python program that converts black and white (grayscale) images to color using deep learning models.

## Features

- 🎨 Colorize single images or batch process directories
- 🚀 Automatic model download on first run
- 💻 Works on CPU (no GPU required)
- 📁 Batch processing for multiple images
- 🖼️ Supports common image formats (JPG, PNG, BMP, TIFF, WEBP)
- ⚡ Fast processing with OpenCV DNN

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Colorize an image:**
   ```bash
   python colorize.py input.jpg output.jpg
   ```

3. **The program will automatically download the model files (~170MB) on first run.**

That's it! Your colorized image will be saved to `output.jpg`.

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **The program will automatically download pre-trained models (~170MB) on first run.**

## Usage

### Colorize a Single Image

```bash
python colorize.py input.jpg output.jpg
```

### Batch Process Directory

```bash
python colorize.py --batch input_directory/ output_directory/
```

### Batch Process with Auto Output Directory

```bash
python colorize.py --batch input_directory/
```

## Examples

```bash
# Basic usage - colorize a single image
python colorize.py old_photo.jpg colorized_photo.jpg

# Process all images in a folder
python colorize.py --batch ./old_photos/ ./colorized_photos/

# Process with auto output directory (saves to input_dir/colorized_output/)
python colorize.py --batch ./old_photos/
```

## Requirements

- Python 3.7+
- OpenCV (opencv-python)
- NumPy
- Internet connection (for initial model download)
- 2GB+ RAM recommended

## How It Works

This program uses a pre-trained Caffe deep learning model based on the research paper "Colorful Image Colorization" by Richard Zhang et al. The model:

1. Converts the grayscale image to LAB color space
2. Uses the L (lightness) channel as input
3. Predicts the A and B (color) channels using a convolutional neural network
4. Combines the predicted colors with the original lightness
5. Converts back to RGB for the final colorized image

## Model Files

The program automatically downloads these files on first run (stored in `models/` directory):
- `colorization_deploy_v2.prototxt` - Model architecture
- `colorization_release_v2.caffemodel` - Trained model weights (~167MB)
- `pts_in_hull.npy` - Cluster centers for color prediction

## Notes

- The first run will download model files (~170MB total)
- Processing time depends on image size (typically 1-5 seconds per image)
- Results are artistic interpretations and may not match original colors exactly
- Works best with natural photographs
- The model was trained on natural images, so results may vary for art, cartoons, etc.

## Troubleshooting

**Import errors:**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- If you get "No module named cv2", install opencv-python: `pip install opencv-python`

**Model download issues:**
- Check internet connection
- Models are downloaded automatically on first use
- If download fails, you can manually download from the URLs in the code and place them in the `models/` directory

**Out of memory:**
- Process smaller images
- Process images one at a time instead of batch mode

**Poor colorization results:**
- Results depend on image content
- Works best with natural photographs
- May not work well with art, cartoons, or heavily processed images

## License

This program uses a pre-trained model based on research by Richard Zhang et al.:
- Paper: "Colorful Image Colorization" (ECCV 2016)
- GitHub: https://github.com/richzhang/colorization

