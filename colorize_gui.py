#!/usr/bin/env python3
"""
Black and White Image Colorization Program - GUI Version
Converts grayscale/black and white images to color using deep learning.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading

try:
    import cv2
except ImportError:
    print("Error: opencv-python is not installed.")
    print("Please install it using: pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: numpy is not installed.")
    print("Please install it using: pip install numpy")
    sys.exit(1)

# Import the colorization functions from the main script
# We need to import these after tkinter is available for error messages
import urllib.request
import urllib.error

# Model files URLs (same as colorize.py)
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
PROTOTXT_PATH = os.path.join(MODEL_DIR, 'colorization_deploy_v2.prototxt')
CAFFEMODEL_PATH = os.path.join(MODEL_DIR, 'colorization_release_v2.caffemodel')
PTS_IN_HULL_PATH = os.path.join(MODEL_DIR, 'pts_in_hull.npy')

MODEL_URLS = {
    'prototxt': 'https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt',
    'caffemodel': 'https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1',
    'pts_in_hull': 'https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy'
}


def download_file(url, filepath, description, progress_callback=None):
    """Download a file from URL if it doesn't exist."""
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        if progress_callback:
            progress_callback(f"✓ {description} already exists ({file_size / (1024*1024):.1f} MB)")
        return
    
    if progress_callback:
        progress_callback(f"Downloading {description}...")
    
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create a request with a User-Agent header to avoid 403 errors
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        with urllib.request.urlopen(req) as response:
            with open(filepath, 'wb') as out_file:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                block_size = 8192
                
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0 and progress_callback:
                        percent = downloaded * 100.0 / total_size
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        progress_callback(f"Downloading {description}... {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
        file_size = os.path.getsize(filepath)
        if progress_callback:
            progress_callback(f"✓ {description} downloaded ({file_size / (1024*1024):.1f} MB)")
    except urllib.error.HTTPError as e:
        error_msg = f"HTTP Error {e.code}: Failed to download {description}\nURL: {url}\n\nPlease check your internet connection or download manually."
        if progress_callback:
            progress_callback(f"✗ {error_msg}")
        raise Exception(error_msg) from e
    except urllib.error.URLError as e:
        error_msg = f"Network Error: Failed to download {description}\nReason: {e.reason}\n\nPlease check your internet connection."
        if progress_callback:
            progress_callback(f"✗ {error_msg}")
        raise Exception(error_msg) from e
    except Exception as e:
        error_msg = f"Error downloading {description}: {str(e)}"
        if progress_callback:
            progress_callback(f"✗ {error_msg}")
        raise Exception(error_msg) from e


def download_models(progress_callback=None):
    """Download required model files if they don't exist."""
    if progress_callback:
        progress_callback("Checking for model files...")
    download_file(MODEL_URLS['prototxt'], PROTOTXT_PATH, 'Prototxt file', progress_callback)
    download_file(MODEL_URLS['caffemodel'], CAFFEMODEL_PATH, 'Caffe model file', progress_callback)
    download_file(MODEL_URLS['pts_in_hull'], PTS_IN_HULL_PATH, 'Cluster centers file', progress_callback)


def load_colorizer():
    """Load and configure the colorization model."""
    # Check if files exist
    if not all(os.path.exists(p) for p in [PROTOTXT_PATH, CAFFEMODEL_PATH, PTS_IN_HULL_PATH]):
        raise FileNotFoundError("Model files not found. Please check your internet connection and try again.")
    
    # Load the model
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
    
    # Load cluster centers
    pts = np.load(PTS_IN_HULL_PATH)
    
    # Populate cluster centers as 1x1 convolution kernel
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype(np.float32)]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]
    
    return net


def colorize_image(input_path, output_path, net=None):
    """Colorize a single image.
    
    Returns:
        str: The final output path that was used (may differ from input if extension was fixed)
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load model if not provided
    if net is None:
        net = load_colorizer()
    
    # Read image
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")
    
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
        else:
            # No extension, add .jpg
            output_path = output_path + '.jpg'
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
    
    return output_path


class ColorizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Black & White Image Colorizer")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.is_processing = False
        self.net = None
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        self.create_widgets()
        self.check_models()
        
    def create_widgets(self):
        # Title
        title_frame = ttk.Frame(self.root, padding="20")
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            title_frame,
            text="🎨 Black & White Image Colorizer",
            font=("Arial", 18, "bold"),
            fg="#2c3e50"
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Convert grayscale images to color using AI",
            font=("Arial", 10),
            fg="#7f8c8d"
        )
        subtitle_label.pack(pady=(5, 0))
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input file selection
        input_frame = ttk.LabelFrame(main_frame, text="Input Image", padding="15")
        input_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(input_frame, text="Select a black and white image:").pack(anchor=tk.W, pady=(0, 10))
        
        input_path_frame = ttk.Frame(input_frame)
        input_path_frame.pack(fill=tk.X)
        
        self.input_entry = ttk.Entry(input_path_frame, textvariable=self.input_path, width=50)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        ttk.Button(
            input_path_frame,
            text="Browse...",
            command=self.browse_input,
            width=12
        ).pack(side=tk.LEFT)
        
        # Output file selection
        output_frame = ttk.LabelFrame(main_frame, text="Output Image", padding="15")
        output_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(output_frame, text="Save colorized image as:").pack(anchor=tk.W, pady=(0, 10))
        
        output_path_frame = ttk.Frame(output_frame)
        output_path_frame.pack(fill=tk.X)
        
        self.output_entry = ttk.Entry(output_path_frame, textvariable=self.output_path, width=50)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        ttk.Button(
            output_path_frame,
            text="Browse...",
            command=self.browse_output,
            width=12
        ).pack(side=tk.LEFT)
        
        # Auto-generate output path when input changes
        self.input_path.trace('w', self.auto_generate_output)
        
        # Process button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.process_button = ttk.Button(
            button_frame,
            text="🎨 Colorize Image",
            command=self.process_image,
            style="Accent.TButton"
        )
        self.process_button.pack(fill=tk.X, pady=(0, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(
            button_frame,
            mode='indeterminate',
            length=400
        )
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        # Status label
        self.status_label = tk.Label(
            button_frame,
            text="Ready",
            font=("Arial", 9),
            fg="#7f8c8d",
            justify=tk.LEFT,
            wraplength=550
        )
        self.status_label.pack(fill=tk.X)
        
        # Info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(20, 0))
        
        info_text = tk.Text(
            info_frame,
            height=4,
            wrap=tk.WORD,
            font=("Arial", 8),
            fg="#7f8c8d",
            bg=self.root.cget('bg'),
            relief=tk.FLAT,
            padx=10,
            pady=5
        )
        info_text.pack(fill=tk.X)
        info_text.insert('1.0', "💡 Tip: Works best with natural photographs. The first run will download model files (~170MB).")
        info_text.config(state=tk.DISABLED)
        
    def check_models(self):
        """Check if model files exist, download if needed."""
        def update_status(msg):
            self.status_label.config(text=msg)
            self.root.update()
        
        try:
            download_models(progress_callback=update_status)
            self.status_label.config(text="Ready - Model files available")
        except Exception as e:
            error_msg = str(e)
            self.status_label.config(text=f"Error: {error_msg}")
            # Show detailed error dialog
            messagebox.showerror(
                "Model Download Error",
                f"Failed to download model files.\n\n{error_msg}\n\n"
                "You can:\n"
                "1. Check your internet connection\n"
                "2. Manually download the files (see README)\n"
                "3. Try again later"
            )
    
    def browse_input(self):
        """Browse for input image file."""
        filename = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.input_path.set(filename)
            if not self.output_path.get():
                self.auto_generate_output()
    
    def browse_output(self):
        """Browse for output image file location."""
        filename = filedialog.asksaveasfilename(
            title="Save Colorized Image As",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("BMP files", "*.bmp"),
                ("TIFF files", "*.tiff"),
                ("All files", "*.*")
            ]
        )
        if filename:
            # Ensure the filename has a valid extension
            ext = os.path.splitext(filename)[1].lower()
            if not ext or ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                # Add .jpg if no extension or unsupported extension
                if not ext:
                    filename = filename + '.jpg'
                else:
                    filename = os.path.splitext(filename)[0] + '.jpg'
            self.output_path.set(filename)
    
    def auto_generate_output(self, *args):
        """Auto-generate output path from input path."""
        if self.input_path.get() and not self.output_path.get():
            input_file = Path(self.input_path.get())
            # Get input extension and use .jpg for output (most compatible)
            input_ext = input_file.suffix.lower()
            # Change extension to .jpg for output
            output_name = f"colorized_{input_file.stem}.jpg"
            output_file = input_file.parent / output_name
            self.output_path.set(str(output_file))
    
    def update_status(self, message):
        """Update status label."""
        self.status_label.config(text=message)
        self.root.update()
    
    def process_image(self):
        """Process the image in a separate thread."""
        if self.is_processing:
            return
        
        # Validate input
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input image.")
            return
        
        if not os.path.exists(self.input_path.get()):
            messagebox.showerror("Error", "Input file does not exist.")
            return
        
        if not self.output_path.get():
            messagebox.showerror("Error", "Please specify an output file path.")
            return
        
        # Start processing in a separate thread
        self.is_processing = True
        self.process_button.config(state=tk.DISABLED, text="Processing...")
        self.progress.start()
        
        thread = threading.Thread(target=self._process_image_thread)
        thread.daemon = True
        thread.start()
    
    def _process_image_thread(self):
        """Process image in background thread."""
        try:
            # Load model if not already loaded
            if self.net is None:
                self.root.after(0, self.update_status, "Loading colorization model...")
                self.net = load_colorizer()
            
            # Process image
            self.root.after(0, self.update_status, "Processing image...")
            final_output_path = colorize_image(self.input_path.get(), self.output_path.get(), self.net)
            
            # Update output path in case it was modified (e.g., extension fixed)
            self.root.after(0, lambda: self.output_path.set(final_output_path))
            
            # Success
            self.root.after(0, self._processing_complete, True, "Image colorized successfully!", final_output_path)
            
        except Exception as e:
            self.root.after(0, self._processing_complete, False, f"Error: {str(e)}", None)
    
    def _processing_complete(self, success, message, output_path=None):
        """Handle processing completion."""
        self.is_processing = False
        self.progress.stop()
        self.process_button.config(state=tk.NORMAL, text="🎨 Colorize Image")
        self.status_label.config(text=message)
        
        if success:
            final_path = output_path if output_path else self.output_path.get()
            messagebox.showinfo("Success", f"Image colorized successfully!\n\nSaved to:\n{final_path}")
        else:
            messagebox.showerror("Error", message)


def main():
    root = tk.Tk()
    app = ColorizationGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

