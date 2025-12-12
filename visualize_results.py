#!/usr/bin/env python3
"""
Results Visualization Script
Creates side-by-side comparison images of original and colorized results.
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path


def create_comparison(original_path, colorized_path, output_path, labels=None):
    """
    Create a side-by-side comparison image.
    
    Args:
        original_path: Path to original grayscale image
        colorized_path: Path to colorized image
        output_path: Path to save comparison image
        labels: Tuple of (original_label, colorized_label) or None
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
        print(f"Resizing colorized image to match original size")
        colorized = cv2.resize(colorized, (original.shape[1], original.shape[0]))
    
    # Convert original to 3-channel if grayscale
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    elif original.shape[2] == 1:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    # Get dimensions
    h, w = original.shape[:2]
    
    # Create comparison image (side by side with padding)
    padding = 20
    label_height = 40 if labels else 0
    comparison = np.ones((h + label_height, w * 2 + padding * 3, 3), dtype=np.uint8) * 255
    
    # Place images
    comparison[label_height:label_height+h, padding:padding+w] = original
    comparison[label_height:label_height+h, padding*2+w:padding*2+w*2] = colorized
    
    # Add labels if provided
    if labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (0, 0, 0)
        
        # Original label
        (text_w1, text_h1), _ = cv2.getTextSize(labels[0], font, font_scale, thickness)
        cv2.putText(comparison, labels[0],
                   (padding + (w - text_w1) // 2, label_height - 10),
                   font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Colorized label
        (text_w2, text_h2), _ = cv2.getTextSize(labels[1], font, font_scale, thickness)
        cv2.putText(comparison, labels[1],
                   (padding*2 + w + (w - text_w2) // 2, label_height - 10),
                   font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Save comparison
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(output_path, comparison)
    print(f"Comparison saved to: {output_path}")


def create_grid_comparison(image_pairs, output_path, cols=2, labels=None):
    """
    Create a grid comparison of multiple image pairs.
    
    Args:
        image_pairs: List of (original_path, colorized_path) tuples
        output_path: Path to save grid image
        cols: Number of columns in grid
        labels: Tuple of (original_label, colorized_label) or None
    """
    if not image_pairs:
        raise ValueError("No image pairs provided")
    
    # Load all images
    images = []
    max_h, max_w = 0, 0
    
    for orig_path, color_path in image_pairs:
        orig = cv2.imread(orig_path)
        color = cv2.imread(color_path)
        
        if orig is None or color is None:
            print(f"Warning: Skipping pair ({orig_path}, {color_path})")
            continue
        
        # Resize to same size (use original size)
        if orig.shape[:2] != color.shape[:2]:
            color = cv2.resize(color, (orig.shape[1], orig.shape[0]))
        
        # Convert to 3-channel if needed
        if len(orig.shape) == 2:
            orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        elif orig.shape[2] == 1:
            orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        
        images.append((orig, color))
        max_h = max(max_h, orig.shape[0])
        max_w = max(max_w, orig.shape[1])
    
    if not images:
        raise ValueError("No valid image pairs found")
    
    # Resize all images to same size
    target_size = (max_w, max_h)
    resized_images = []
    for orig, color in images:
        if orig.shape[:2] != target_size[::-1]:
            orig = cv2.resize(orig, target_size)
        if color.shape[:2] != target_size[::-1]:
            color = cv2.resize(color, target_size)
        resized_images.append((orig, color))
    
    # Calculate grid dimensions
    rows = (len(resized_images) + cols - 1) // cols
    padding = 20
    label_height = 40 if labels else 0
    
    # Create grid
    grid_h = rows * (max_h + padding) + padding + label_height
    grid_w = cols * (max_w * 2 + padding * 3) + padding
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    
    # Place images in grid
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_color = (0, 0, 0)
    
    for idx, (orig, color) in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        
        y_start = label_height + row * (max_h + padding) + padding
        x_start = col * (max_w * 2 + padding * 3) + padding
        
        # Place original
        grid[y_start:y_start+max_h, x_start:x_start+max_w] = orig
        
        # Place colorized
        grid[y_start:y_start+max_h, x_start+max_w+padding:x_start+max_w*2+padding] = color
        
        # Add labels
        if labels:
            # Original label
            (text_w1, _), _ = cv2.getTextSize(labels[0], font, font_scale, thickness)
            cv2.putText(grid, labels[0],
                       (x_start + (max_w - text_w1) // 2, label_height - 10),
                       font, font_scale, text_color, thickness, cv2.LINE_AA)
            
            # Colorized label
            (text_w2, _), _ = cv2.getTextSize(labels[1], font, font_scale, thickness)
            cv2.putText(grid, labels[1],
                       (x_start+max_w+padding + (max_w - text_w2) // 2, label_height - 10),
                       font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    # Save grid
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(output_path, grid)
    print(f"Grid comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create visualization comparisons of original and colorized images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single comparison
  python visualize_results.py original.jpg colorized.jpg comparison.jpg
  
  # With labels
  python visualize_results.py original.jpg colorized.jpg comparison.jpg --labels "Original" "Colorized"
  
  # Batch grid comparison
  python visualize_results.py --batch input_dir/ output_dir/ grid.jpg
        """
    )
    
    parser.add_argument('original', nargs='?', help='Path to original grayscale image')
    parser.add_argument('colorized', nargs='?', help='Path to colorized image')
    parser.add_argument('output', nargs='?', help='Path to save comparison image')
    parser.add_argument('--labels', nargs=2, metavar=('ORIG_LABEL', 'COLOR_LABEL'),
                       help='Labels for original and colorized images')
    parser.add_argument('--batch', action='store_true', help='Create grid from directory')
    parser.add_argument('--input-dir', help='Input directory (for batch mode)')
    parser.add_argument('--output-dir', help='Output directory with colorized images (for batch mode)')
    parser.add_argument('--cols', type=int, default=2, help='Number of columns in grid (default: 2)')
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            if not args.input_dir or not args.output_dir or not args.output:
                parser.error("--batch requires --input-dir, --output-dir, and output path")
            
            input_dir = Path(args.input_dir)
            output_dir = Path(args.output_dir)
            
            # Find all images
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
            image_files = []
            for ext in extensions:
                image_files.extend(input_dir.glob(f'*{ext}'))
                image_files.extend(input_dir.glob(f'*{ext.upper()}'))
            
            # Create pairs
            pairs = []
            for img_path in sorted(image_files):
                orig_path = str(img_path)
                color_path = str(output_dir / f"colorized_{img_path.name}")
                if os.path.exists(color_path):
                    pairs.append((orig_path, color_path))
            
            if not pairs:
                print("No matching image pairs found")
                sys.exit(1)
            
            print(f"Creating grid comparison with {len(pairs)} image pairs...")
            create_grid_comparison(pairs, args.output, cols=args.cols, labels=args.labels)
        
        else:
            if not args.original or not args.colorized or not args.output:
                parser.print_help()
                sys.exit(1)
            
            labels = tuple(args.labels) if args.labels else None
            create_comparison(args.original, args.colorized, args.output, labels)
        
        print("Visualization complete!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()



