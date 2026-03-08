# apply-indices.py
#
# Code to apply various vegetative indices to a set of input images
#
# Simon Parsons
# University of Lincoln
# 26-03-06

# Written with liberal help from Claude

import cv2
import argparse
import pandas as pd
import vegetative_indices as vg
from pathlib import Path

def process_images(input_dir, output_csv, selected_functions, threshold, normalize):
    """
    Process images from a directory, apply functions, and save results to CSV.
    
    Parameters:
    -----------
    input_dir : str or Path
        Directory containing input images
    output_csv : str or Path
        Path for output CSV file
    selected_functions : list
        List of function names to apply
    threshold : float
        Threshold parameter for functions that need it
    """
    
    # Define your processing functions here
    # Each function should take an image (numpy array) and return a single value or metric

    # Functions to set up calls to the code that computes the
    # vegetative indices. Repetitve, but allows for different default
    # thresholds for each.
    def computeExG(img, threshold=None):
        if threshold is None:
            threshold = 0
        exgImg, exgCount = vg.applyThreshold(vg.computeExGImage(img), threshold)
        return exgCount

    def computeExGR(img, threshold=None):
        if threshold is None:
            threshold = 0
        exgrImg, exgrCount = vg.applyThreshold(vg.computeExGRImage(img), threshold)
        return exgrCount

    def computeGLI(img, threshold=None):
        if threshold is None:
            threshold = 0
        gliImg, gliCount = vg.applyThreshold(vg.computeGLIImage(img), threshold)
        return gliCount

    def computeVARI(img, threshold=None):
        if threshold is None:
            threshold = 0
        variImg, variCount = vg.applyThreshold(vg.computeGLIImage(img), threshold)
        return variCount

    # Templates for future functions
    def function_2(img, threshold=None):
        """Example: Get image dimensions"""
        # Replace with your actual function
        return img.shape[0] * img.shape[1]
    
    def function_3(img, threshold=None):
        """Example: Count number of channels"""
        # Replace with your actual function
        return img.shape[2] if len(img.shape) == 3 else 1
    
    def function_with_threshold(img, threshold=None):
        """Example: Count pixels above threshold"""
        # Replace with your actual threshold-based function
        if threshold is None:
            threshold = 128
        return (img > threshold).sum()
    
    # Add more functions as needed
    
    # All available functions
    all_functions = {
        'ExG':  computeExG,
        'ExGR': computeExGR,
        'GLI':  computeGLI,
        'VARI': computeVARI,
        # Add more function mappings here
    }
    
    # Filter to selected functions
    if selected_functions:
        functions = {name: all_functions[name] for name in selected_functions 
                    if name in all_functions}
        
        # Check for invalid function names
        invalid = set(selected_functions) - set(all_functions.keys())
        if invalid:
            print(f"Warning: Unknown functions will be ignored: {invalid}")
    else:
        # If no functions specified, use all
        functions = all_functions
    
    if not functions:
        raise ValueError("No valid functions selected")
    
    # Initialize list to store results
    results = []
    
    # Get all image files from directory
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        raise ValueError(f"No image files found in {input_dir}")
    
    # Process each image
    for img_path in sorted(image_files):
        print(f"Processing: {img_path.name}")
        
        # Read image
        img = cv2.imread(str(img_path))
       
        if img is None:
            print(f"Warning: Could not read {img_path.name}, skipping...")
            continue
        
        if normalize:
            # If needed, normalize bands
            print(f" Normalizing: {img_path.name}")
            b_normal, g_normal ,r_normal  = vg.normalizeBands(img)
            img = cv2.merge([b_normal, g_normal, r_normal])
            
        # Apply each function and store results
        row_data = {'filename': img_path.name}
        
        for col_name, func in functions.items():
            try:
                print(f" Applying: {col_name} to {img_path.name}")
                row_data[col_name] = func(img, threshold=threshold)
            except Exception as e:
                print(f"Error applying {col_name} to {img_path.name}: {e}")
                row_data[col_name] = None
        
        results.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Export to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\nProcessing complete. Results saved to: {output_csv}")
    print(f"Processed {len(df)} images with {len(functions)} function(s)")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Process images with OpenCV and export results to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images with all functions
  python script.py /path/to/images output.csv
  
  # Process with specific functions
  python script.py /path/to/images output.csv -f function_1 function_2
  
  # Process with threshold parameter
  python script.py /path/to/images output.csv -f function_with_threshold -t 150
  
  # Combine multiple options
  python script.py /path/to/images results.csv -f function_1 function_with_threshold -t 200
        """
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='Path to directory containing images'
    )
    
    parser.add_argument(
        'output_csv',
        type=str,
        help='Output CSV filename/path'
    )
    
    parser.add_argument(
        '-f', '--functions',
        nargs='+',
        type=str,
        default=None,
        help='Names of functions to apply (space-separated). If not specified, all functions will be used.'
    )

    # This is not ideal since it assumes the same threshold for each
    # function. Really need the ablity to enter a list, in the same
    # order as for the functions.
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=None,
        help='Threshold parameter for functions that require it (default: None)'
    )

    parser.add_argument(
        '-n', '--normalize',
        type=bool,
        default=True,
        help='Should we normalize the image (default: True)'
    )
    
    parser.add_argument(
        '--list-functions',
        action='store_true',
        help='List all available functions and exit'
    )
    
    args = parser.parse_args()
    
    # List functions if requested
    if args.list_functions:
        print("Available indices:")
        print("  - ExG")
        print("  - ExGR")
        print("  - GLI")
        print("  - VARI")
        print("\nAdd more functions in the script as needed.")
        return
    
    # Run processing
    try:
        df = process_images(
            args.input_dir,
            args.output_csv,
            args.functions,
            args.threshold,
            args.normalize
        )
        
        # Display first few rows
        print("\nFirst few rows of results:")
        print(df.head())
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
'''
    Usage examples:
bash
￼
# Process all images with all functions
python script.py /path/to/images output.csv

# Process with specific functions only
python script.py /path/to/images output.csv -f function_1 function_2

# Use threshold parameter
python script.py /path/to/images output.csv -f function_with_threshold -t 150

# Combine multiple functions with threshold
python script.py /path/to/images results.csv -f function_1 function_with_threshold -t 200

# List available functions
python script.py --list-functions
'''
