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
    # vegetative indices. Different generic functions to allow
    # different defaults (could obviously do this in a different way)

    # Compute index with default zero
    def computeIndex(img, index, threshold=None):
        indexImg = vg.computeIndexByName(img, index)
        
        # No default, so use zero
        if threshold is None:
            threshold = 0
        elif threshold == "otsu":
            threshold = vg.calculateOtsuThreshold(indexImg)
            
        _, indexCount = vg.applyThreshold(indexImg, threshold)
        return indexCount

    # Compute index with default Otsu
    def computeIndexOtsu(img, index, threshold=None):
        indexImg = vg.computeIndexByName(img, index)
        
        # No default, so use Otsu
        if threshold is None:
            threshold = vg.calculateOtsuThreshold(indexImg)
        elif threshold == "otsu":
            threshold = vg.calculateOtsuThreshold(indexImg)
            
        _, indexCount = vg.applyThreshold(indexImg, threshold)
        return indexCount
    '''
    def computeExGR(img, threshold=None):
        exgrImg = vg.computeExGRImage(img)
        
        if threshold is None:
            threshold = 0
        elif threshold == "otsu":
            threshold = vg.calculateOtsuThreshold(exgrImg)
            
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
    '''

    # All available functions. This is used to check that the relevant
    # index is one we can handle and allows for different functions
    # for different indices. For example, have functions with
    # different default thresholds, though most assume 0.
    all_functions = {
        'ExG':  computeIndexOtsu,
        'ExGR': computeIndex,
        'GLI':  computeIndex,
        'VARI': computeIndex,
        "RGBVI": computeIndex,
        "GLI": computeIndex,
        "DGCI": computeIndex,
        "NGBDI": computeIndex,
        "GRVI": computeIndex,
        "NRI": computeIndex,
        "NGI": computeIndex,
        "NBI": computeIndex,
        "SAVI": computeIndex,
        "GMR": computeIndex,
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
        print(f" Size: {img.shape[:2]}")

        if normalize:
            # If needed, normalize image
            print(f" Normalizing: {img_path.name}")
            img  = vg.normalizeImage(img)
            
        # Apply each function and store results
        row_data = {'filename': img_path.name}
             
        for index, func in functions.items():
            try:
                #row_data[col_name] = func(img, **thresholds)
                threshold_value = thresholds.get(index, None)
                ######################
                # This is where the function defined in the dictionary
                # above is called. Need to interface this with the
                # index dispatcher in vegetative_indices
                row_data[index] = func(img, index, threshold_value)
            except Exception as e:
                print(f"Error applying {index} to {img_path.name}: {e}")
                row_data[index] = None
        
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

    parser.add_argument(
        '-n', '--normalize',
        type=str,
        default=None,
        help='Should we normalize the images. Enter "True" or "true" for normalization (default: False)'
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

    #print("args.normalize", args.normalize)
    
    # Only normalize if we explicitly say to do that. Note that
    # args.normalize is None by default.
    if args.normalize:
        if  args.normalize == "True" or args.normalize == "true":
            normalize = True
        else:
            normalize = False
    else:
        normalize = False

    print("Normalize: ", normalize)
    
    # Run processing
    try:
        df = process_images(
            args.input_dir,
            args.output_csv,
            args.functions,
            function_thresholds,
            normalize
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
