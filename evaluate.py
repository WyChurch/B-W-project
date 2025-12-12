#!/usr/bin/env python3
"""
Evaluation Metrics for Image Colorization
Computes quantitative metrics to assess colorization quality.
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not installed. Some metrics will be unavailable.")
    print("Install with: pip install scikit-image")


def calculate_psnr(img1, img2):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1: First image (numpy array)
        img2: Second image (numpy array)
    
    Returns:
        float: PSNR value in dB
    """
    if HAS_SKIMAGE:
        # Convert to grayscale for PSNR calculation
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2
        
        return psnr(img1_gray, img2_gray, data_range=255)
    else:
        # Manual PSNR calculation
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2
        
        mse = np.mean((img1_gray.astype(np.float64) - img2_gray.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1: First image (numpy array)
        img2: Second image (numpy array)
    
    Returns:
        float: SSIM value (0-1, higher is better)
    """
    if not HAS_SKIMAGE:
        print("SSIM requires scikit-image. Install with: pip install scikit-image")
        return None
    
    # Convert to grayscale for SSIM calculation
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    return ssim(img1_gray, img2_gray, data_range=255)


def calculate_mse(img1, img2):
    """
    Calculate Mean Squared Error (MSE) between two images.
    
    Args:
        img1: First image (numpy array)
        img2: Second image (numpy array)
    
    Returns:
        float: MSE value
    """
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    return np.mean((img1_gray.astype(np.float64) - img2_gray.astype(np.float64)) ** 2)


def calculate_color_statistics(img):
    """
    Calculate color statistics for an image.
    
    Args:
        img: Input image (BGR format)
    
    Returns:
        dict: Dictionary containing color statistics
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Extract channels
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    
    # Calculate statistics
    stats = {
        'l_mean': np.mean(l_channel),
        'l_std': np.std(l_channel),
        'a_mean': np.mean(a_channel),
        'a_std': np.std(a_channel),
        'b_mean': np.mean(b_channel),
        'b_std': np.std(b_channel),
        'saturation': np.mean(np.sqrt(a_channel.astype(np.float32) ** 2 + b_channel.astype(np.float32) ** 2))
    }
    
    return stats


def compare_images(original_path, colorized_path, reference_path=None):
    """
    Compare original grayscale image with colorized version.
    Optionally compare with a reference color image if available.
    
    Args:
        original_path: Path to original grayscale image
        colorized_path: Path to colorized image
        reference_path: Optional path to reference color image (ground truth)
    
    Returns:
        dict: Dictionary containing comparison metrics
    """
    # Load images
    original = cv2.imread(original_path)
    colorized = cv2.imread(colorized_path)
    
    if original is None:
        raise ValueError(f"Could not load original image: {original_path}")
    if colorized is None:
        raise ValueError(f"Could not load colorized image: {colorized_path}")
    
    # Ensure same size
    if original.shape[:2] != colorized.shape[:2]:
        print(f"Warning: Image sizes differ. Resizing colorized to match original.")
        colorized = cv2.resize(colorized, (original.shape[1], original.shape[0]))
    
    results = {
        'original_path': original_path,
        'colorized_path': colorized_path,
        'original_size': original.shape,
        'colorized_size': colorized.shape
    }
    
    # Calculate metrics between original grayscale and colorized
    # Convert original to grayscale if needed
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
    
    colorized_gray = cv2.cvtColor(colorized, cv2.COLOR_BGR2GRAY)
    
    results['mse'] = calculate_mse(original_gray, colorized_gray)
    results['psnr'] = calculate_psnr(original_gray, colorized_gray)
    
    if HAS_SKIMAGE:
        results['ssim'] = calculate_ssim(original_gray, colorized_gray)
    
    # Color statistics
    results['colorized_stats'] = calculate_color_statistics(colorized)
    
    # If reference image is provided, compare colorized with reference
    if reference_path and os.path.exists(reference_path):
        reference = cv2.imread(reference_path)
        if reference is not None:
            if reference.shape[:2] != colorized.shape[:2]:
                reference = cv2.resize(reference, (colorized.shape[1], colorized.shape[0]))
            
            results['reference_path'] = reference_path
            results['psnr_vs_reference'] = calculate_psnr(reference, colorized)
            if HAS_SKIMAGE:
                results['ssim_vs_reference'] = calculate_ssim(reference, colorized)
            results['mse_vs_reference'] = calculate_mse(reference, colorized)
            results['reference_stats'] = calculate_color_statistics(reference)
    
    return results


def print_results(results):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nOriginal Image: {results['original_path']}")
    print(f"Colorized Image: {results['colorized_path']}")
    print(f"\nImage Dimensions: {results['original_size']}")
    
    print("\n--- Grayscale Preservation Metrics ---")
    print(f"MSE (Mean Squared Error): {results['mse']:.2f}")
    print(f"PSNR (Peak Signal-to-Noise Ratio): {results['psnr']:.2f} dB")
    if 'ssim' in results:
        print(f"SSIM (Structural Similarity Index): {results['ssim']:.4f}")
    
    print("\n--- Color Statistics (Colorized Image) ---")
    stats = results['colorized_stats']
    print(f"Lightness (L) - Mean: {stats['l_mean']:.2f}, Std: {stats['l_std']:.2f}")
    print(f"Green-Red (A) - Mean: {stats['a_mean']:.2f}, Std: {stats['a_std']:.2f}")
    print(f"Blue-Yellow (B) - Mean: {stats['b_mean']:.2f}, Std: {stats['b_std']:.2f}")
    print(f"Average Saturation: {stats['saturation']:.2f}")
    
    if 'reference_path' in results:
        print("\n--- Comparison with Reference Image ---")
        print(f"Reference Image: {results['reference_path']}")
        print(f"PSNR vs Reference: {results['psnr_vs_reference']:.2f} dB")
        if 'ssim_vs_reference' in results:
            print(f"SSIM vs Reference: {results['ssim_vs_reference']:.4f}")
        print(f"MSE vs Reference: {results['mse_vs_reference']:.2f}")
        
        print("\n--- Reference Color Statistics ---")
        ref_stats = results['reference_stats']
        print(f"Lightness (L) - Mean: {ref_stats['l_mean']:.2f}, Std: {ref_stats['l_std']:.2f}")
        print(f"Green-Red (A) - Mean: {ref_stats['a_mean']:.2f}, Std: {ref_stats['a_std']:.2f}")
        print(f"Blue-Yellow (B) - Mean: {ref_stats['b_mean']:.2f}, Std: {ref_stats['b_std']:.2f}")
        print(f"Average Saturation: {ref_stats['saturation']:.2f}")
    
    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate colorization quality using quantitative metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare original and colorized images
  python evaluate.py original.jpg colorized.jpg
  
  # Compare with reference (ground truth) image
  python evaluate.py original.jpg colorized.jpg --reference reference.jpg
  
  # Batch evaluation
  python evaluate.py --batch input_dir/ output_dir/
        """
    )
    
    parser.add_argument('original', nargs='?', help='Path to original grayscale image')
    parser.add_argument('colorized', nargs='?', help='Path to colorized image')
    parser.add_argument('--reference', help='Path to reference color image (ground truth, optional)')
    parser.add_argument('--batch', action='store_true', help='Batch process directory')
    parser.add_argument('--input-dir', help='Input directory (for batch mode)')
    parser.add_argument('--output-dir', help='Output directory (for batch mode)')
    
    args = parser.parse_args()
    
    if args.batch:
        if not args.input_dir or not args.output_dir:
            parser.error("--batch requires --input-dir and --output-dir")
        
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        
        # Find all images
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images to evaluate\n")
        
        all_results = []
        for img_path in image_files:
            original_path = str(img_path)
            colorized_path = str(output_dir / f"colorized_{img_path.name}")
            
            if not os.path.exists(colorized_path):
                print(f"Skipping {img_path.name}: colorized version not found")
                continue
            
            try:
                results = compare_images(original_path, colorized_path, args.reference)
                print_results(results)
                all_results.append(results)
            except Exception as e:
                print(f"Error evaluating {img_path.name}: {e}")
        
        # Summary statistics
        if all_results:
            print("\n" + "="*60)
            print("BATCH EVALUATION SUMMARY")
            print("="*60)
            psnr_values = [r['psnr'] for r in all_results]
            mse_values = [r['mse'] for r in all_results]
            
            print(f"\nProcessed {len(all_results)} images")
            print(f"Average PSNR: {np.mean(psnr_values):.2f} dB")
            print(f"Average MSE: {np.mean(mse_values):.2f}")
            if all('ssim' in r for r in all_results):
                ssim_values = [r['ssim'] for r in all_results]
                print(f"Average SSIM: {np.mean(ssim_values):.4f}")
            print("="*60 + "\n")
    
    else:
        if not args.original or not args.colorized:
            parser.print_help()
            sys.exit(1)
        
        try:
            results = compare_images(args.original, args.colorized, args.reference)
            print_results(results)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()



