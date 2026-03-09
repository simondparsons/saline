# apply-indices.py
#
# Code to apply various vegetative indices to a set of input images
#
# Simon Parsons
# University of Lincoln
# 26-03-06

# Written with liberal help from Claude Sonnet 4.5, as in Claude did
# teh ehavy lifting and I interfaced it with the code I already had
# for the vegetative indices.

import cv2
import argparse
import pandas as pd
import vegetative_indices as vg
from pathlib import Path

def process_images(input_dir, output_csv, selected_functions, thresholds, normalize):
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
    function_thresholds : dict
        Dictionary mapping function names to their threshold values (can be numeric, None, or "otsu")
    functions_dict : dict
        Dictionary of available functions
    """
    
    # Functions to set up calls to the code that computes the
    # vegetative indices. Repetitve, but allows for different default
    # thresholds for each.

    def computeExG(img, threshold=None):
        exgImg = vg.computeExGImage(img)
        
        # No default, so use Otsu
        if threshold is None:
            threshold = vg.calculateOtsuThreshold(exgImg)
        elif threshold == "otsu":
            threshold = vg.calculateOtsuThreshold(exgImg)
            
        _, exgCount = vg.applyThreshold(exgImg, threshold)
        return exgCount

    def computeExGR(img, threshold=None):
        exgrImg = vg.computeExGRImage(img)
        
        if threshold is None:
            threshold = 0
        elif threshold == "otsu":
            threshold = vg.calculateOtsuThreshold(exgImg)
            
        _, exgrCount = vg.applyThreshold(exgrImg, threshold)
        return exgrCount

    def computeGLI(img, threshold=None):
        gliImg = vg.computeGLIImage(img)
        
        if threshold is None:
            threshold = 0
        elif threshold == "otsu":
            threshold = vg.calculateOtsuThreshold(gliImg)
            
        _, gliCount = vg.applyThreshold(gliImg, threshold)
        
        return gliCount

    def computeVARI(img, threshold=None):
        variImg = vg.computeVARIImage(img)

        if threshold is None:
            threshold = 0
        elif threshold == "otsu":
            threshold = vg.calculateOtsuThreshold(variImg)
            
        _, variCount = vg.applyThreshold(variImg, threshold)
        
        return variCount
    
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
        print(f" Size: {img.shape[:2]}")
       
        if img is None:
            print(f"Warning: Could not read {img_path.name}, skipping...")
            continue
        
        if normalize:
            # If needed, normalize image
            print(f" Normalizing: {img_path.name}")
            img  = vg.normalizeImage(img)
            
        # Just print what thresholds are being passed
        #if thresholds:
        #     print(f"  Passing thresholds: {thresholds}")

        # Apply each function and store results
        row_data = {'filename': img_path.name}
             
        for col_name, func in functions.items():
            try:
                #row_data[col_name] = func(img, **thresholds)
                threshold_value = thresholds.get(col_name, None)
                row_data[col_name] = func(img, threshold_value)
            except Exception as e:
                print(f"Error applying {col_name} to {img_path.name}: {e}")
                row_data[col_name] = None

        # This version computed the Otsu threshold from the original image.
        #
        # Call each function, identifying the necessary threshold, and
        # computing it if required.
        #
        # Since a default can be specified with None, we can include
        # functions that do not need a threshold.
        #row_data = {'filename': img_path.name}

        #for col_name, func in functions.items():
        #    try:
        #        # Get threshold value for this function
        #        threshold_value = function_thresholds.get(col_name, None)
        #        
        #        # Process threshold based on type
        #        if threshold_value == "otsu":
        #            # Calculate Otsu threshold for this image
        #            actual_threshold = vg.calculateOtsuThreshold(img)
        #            print(f"  {col_name}: using Otsu = {actual_threshold:.2f}")
        #            row_data[col_name] = func(img, threshold=actual_threshold)
        #        elif threshold_value is None:
        #            # Pass no threshold, let function use its default
        #            print(f"  {col_name}: using function default")
        #            row_data[col_name] = func(img)
        #        else:
        #            # Use the numeric value provided
        #            print(f"  {col_name}: using threshold = {threshold_value}")
        #            row_data[col_name] = func(img, threshold=threshold_value)
        #            
        #    except Exception as e:
        #        print(f"Error applying {col_name} to {img_path.name}: {e}")
        #        row_data[col_name] = None
        
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

    # New version which allows multiple thresholds to be specified.
    parser.add_argument(
        '-t', '--thresholds',
        nargs='+',
        type=str,
        default=None,
        help='Threshold values for each function specified with -f (in same order). Each value can be: a number (e.g., 150), "None" (use function default), or "otsu" (calculate Otsu threshold per image). Example: -f func1 func2 func3 -t otsu 200 None'
    )
    #parser.add_argument(
    #    '-t', '--threshold',
    #    type=float,
    #    default=None,
    #    help='Threshold parameter for functions that require it (default: None)'
    #)

    parser.add_argument(
        '-n', '--normalize',
        type=bool,
        default=True,
        help='Should we normalize the images (default: True)'
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
        #print("\nAdd more functions in the script as needed.")
        return

    # Parse thresholds to map each function to its threshold value
    function_thresholds = {}
    
    if args.functions and args.thresholds:
        # Check if number of thresholds matches number of functions
        if len(args.thresholds) != len(args.functions):
            parser.error(f"Number of thresholds ({len(args.thresholds)}) must match number of functions ({len(args.functions)})")
        
        # Map each function to its threshold
        for func_name, threshold_str in zip(args.functions, args.thresholds):
            # Check if value is "otsu"
            if threshold_str.lower() == 'otsu':
                function_thresholds[func_name] = "otsu"
            # Check if value is "None"
            elif threshold_str == 'None' or threshold_str == 'none':
                function_thresholds[func_name] = None
            # Try to convert to float
            else:
                try:
                    function_thresholds[func_name] = float(threshold_str)
                except ValueError:
                    print(f"Warning: Invalid threshold value '{threshold_str}' for function '{func_name}', expected number, 'None', or 'otsu'")
                    function_thresholds[func_name] = None
    elif args.thresholds and not args.functions:
        parser.error("Cannot specify thresholds (-t) without specifying functions (-f)")
    
    if function_thresholds:
        print(f"Function-threshold mapping:")
        for func, thresh in function_thresholds.items():
            print(f"  {func}: {thresh}")
    
    # Run processing
    try:
        df = process_images(
            args.input_dir,
            args.output_csv,
            args.functions,
            function_thresholds,
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
