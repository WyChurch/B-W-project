#!/usr/bin/env python3
"""
Black and White Image Colorization Program
Converts grayscale/black and white images to color using deep learning.
"""

import argparse
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Error: opencv-python is not installed.")
    print("Please install it using: pip install opencv-python")
    print("Or install all requirements: pip install -r requirements.txt")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: numpy is not installed.")
    print("Please install it using: pip install numpy")
    print("Or install all requirements: pip install -r requirements.txt")
    sys.exit(1)

# Model files URLs
MODEL_URLS = {
    'prototxt': 'https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt',
    'caffemodel': 'https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1',
    'pts_in_hull': 'https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy'
}

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
PROTOTXT_PATH = os.path.join(MODEL_DIR, 'colorization_deploy_v2.prototxt')
CAFFEMODEL_PATH = os.path.join(MODEL_DIR, 'colorization_release_v2.caffemodel')
PTS_IN_HULL_PATH = os.path.join(MODEL_DIR, 'pts_in_hull.npy')


def download_file(url, filepath, description):
    """Download a file from URL if it doesn't exist."""
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        print(f"✓ {description} already exists ({file_size / (1024*1024):.1f} MB)")
        return
    
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Saving to: {filepath}")
    print("  This may take a few minutes for large files...")
    
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create a request with a User-Agent header to avoid 403 errors
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100.0 / total_size, 100.0)
            if total_size > 0:
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
        
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            block_size = 8192
            
            with open(filepath, 'wb') as out_file:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = downloaded * 100.0 / total_size
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
        
        print()  # New line after progress
        file_size = os.path.getsize(filepath)
        print(f"✓ {description} downloaded successfully ({file_size / (1024*1024):.1f} MB)")
    except urllib.error.HTTPError as e:
        print(f"\n✗ HTTP Error {e.code}: Failed to download {description}")
        print(f"  URL: {url}")
        print(f"  Please check your internet connection or download manually.")
        if os.path.exists(filepath):
            os.remove(filepath)
        raise
    except urllib.error.URLError as e:
        print(f"\n✗ Network Error: Failed to download {description}")
        print(f"  Reason: {e.reason}")
        print(f"  Please check your internet connection.")
        if os.path.exists(filepath):
            os.remove(filepath)
        raise
    except Exception as e:
        print(f"\n✗ Error downloading {description}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        raise


def download_models():
    """Download required model files if they don't exist."""
    print("Checking for model files...")
    download_file(MODEL_URLS['prototxt'], PROTOTXT_PATH, 'Prototxt file')
    download_file(MODEL_URLS['caffemodel'], CAFFEMODEL_PATH, 'Caffe model file')
    download_file(MODEL_URLS['pts_in_hull'], PTS_IN_HULL_PATH, 'Cluster centers file')
    print("All model files ready!\n")


def load_colorizer():
    """Load and configure the colorization model."""
    # Download models if needed
    download_models()
    
    # Check if files exist
    if not all(os.path.exists(p) for p in [PROTOTXT_PATH, CAFFEMODEL_PATH, PTS_IN_HULL_PATH]):
        raise FileNotFoundError("Model files not found. Please check your internet connection and try again.")
    
    # Load the model
    print("Loading colorization model...")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
    
    # Load cluster centers
    pts = np.load(PTS_IN_HULL_PATH)
    
    # Populate cluster centers as 1x1 convolution kernel
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype(np.float32)]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]
    
    print("✓ Model loaded successfully")
    return net


def colorize_image(input_path, output_path, net=None):
    """
    Colorize a single image.
    
    Args:
        input_path: Path to input black and white image
        output_path: Path to save colorized image
        net: Pre-loaded colorization network (optional)
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load model if not provided
    if net is None:
        net = load_colorizer()
    
    # Read image
    print(f"Processing: {input_path}")
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError(f"Could not read image: {input_path}. Please check if the file exists and is a valid image.")
    
    # If image is already grayscale (2D), convert to 3 channel
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Convert to float and normalize
    img_rgb = (img.astype(np.float32)) / 255.0
    
    # Convert to LAB color space
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)
    
    # Extract L channel (lightness)
    l_channel = img_lab[:, :, 0]
    
    # Resize L channel to network input size (224x224)
    l_resized = cv2.resize(l_channel, (224, 224))
    l_resized -= 50  # Mean centering
    
    # Run the network
    net.setInput(cv2.dnn.blobFromImage(l_resized))
    ab_pred = net.forward()[0, :, :, :].transpose((1, 2, 0))
    
    # Resize predicted ab channels to original image size
    ab_pred_resized = cv2.resize(ab_pred, (img.shape[1], img.shape[0]))
    
    # Combine L channel with predicted ab channels
    lab_output = np.concatenate((l_channel[:, :, np.newaxis], ab_pred_resized), axis=2)
    
    # Convert LAB to BGR
    img_bgr = cv2.cvtColor(lab_output, cv2.COLOR_LAB2BGR)
    img_bgr = np.clip(img_bgr, 0, 1)
    img_bgr = (img_bgr * 255).astype(np.uint8)
    
    # Save result
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Validate and fix output path extension
    output_path = os.path.normpath(output_path)
    file_ext = os.path.splitext(output_path)[1].lower()
    
    # Supported extensions by OpenCV
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    # If no extension or unsupported extension, default to .jpg
    if not file_ext or file_ext not in supported_extensions:
        if file_ext:
            # Unsupported extension, change to .jpg
            output_path = os.path.splitext(output_path)[0] + '.jpg'
            print(f"Warning: Unsupported extension '{file_ext}', changed to .jpg")
        else:
            # No extension, add .jpg
            output_path = output_path + '.jpg'
            print(f"Warning: No extension found, added .jpg")
        file_ext = '.jpg'
    
    # Ensure we're using a format that OpenCV supports
    # For JPEG, use .jpg extension explicitly
    if file_ext in ['.jpg', '.jpeg']:
        # Normalize to .jpg for consistency
        output_path = os.path.splitext(output_path)[0] + '.jpg'
    
    # Write the image with proper parameters for JPEG quality
    if file_ext == '.jpg' or file_ext == '.jpeg':
        # Use JPEG encoding with quality parameter
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        success = cv2.imwrite(output_path, img_bgr, encode_params)
    elif file_ext == '.png':
        # Use PNG encoding with compression
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        success = cv2.imwrite(output_path, img_bgr, encode_params)
    else:
        # For other formats, use default
        success = cv2.imwrite(output_path, img_bgr)
    
    if not success:
        # Try to get more information about the failure
        abs_path = os.path.abspath(output_path)
        raise IOError(
            f"Failed to save image to: {abs_path}\n"
            f"Extension: {file_ext}\n"
            f"Please ensure:\n"
            f"1. The file path is valid and doesn't contain invalid characters\n"
            f"2. You have write permissions to the directory\n"
            f"3. The directory exists: {os.path.dirname(abs_path)}\n"
            f"4. There's enough disk space\n"
            f"5. Try using .jpg or .png extension"
        )
    
    print(f"✓ Colorized image saved to: {output_path}")
    return output_path


def batch_colorize(input_dir, output_dir, extensions=None):
    """
    Colorize multiple images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save colorized images
        extensions: List of image extensions to process
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model once for all images
    net = load_colorizer()
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image(s) to process\n")
    
    # Process each image
    successful = 0
    failed = 0
    
    for i, img_path in enumerate(image_files, 1):
        try:
            output_path = os.path.join(output_dir, f"colorized_{img_path.name}")
            print(f"[{i}/{len(image_files)}] Processing: {img_path.name}")
            colorize_image(str(img_path), output_path, net)
            successful += 1
        except Exception as e:
            print(f"✗ Error processing {img_path.name}: {e}")
            failed += 1
        print()
    
    print(f"Batch processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Colorize black and white images using deep learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Colorize a single image
  python colorize.py input.jpg output.jpg
  
  # Batch process directory
  python colorize.py --batch input_dir/ output_dir/
  
  # Batch process with auto output directory
  python colorize.py --batch input_dir/
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input image path or directory (for batch mode)')
    parser.add_argument('output', nargs='?', help='Output image path or directory (for batch mode)')
    parser.add_argument('--batch', action='store_true', help='Process all images in input directory')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.batch:
            if not args.output:
                args.output = os.path.join(args.input, 'colorized_output')
            batch_colorize(args.input, args.output)
        else:
            if not args.output:
                # Generate output filename
                input_path = Path(args.input)
                args.output = str(input_path.parent / f"colorized_{input_path.name}")
            colorize_image(args.input, args.output)
        
        print("\n✓ Colorization complete!")
        
    except KeyboardInterrupt:
        print("\n\n✗ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
