# average-indices.py
#
# Code to compute stats from various vegetative indices over a set of
# input images.
#
# Simon Parsons
# University of Lincoln
# 26-04-09
#

# Written with liberal help from Claude Sonnet 4.5 and ChatGPT
# 5.2-Auto, as in the LLMs did the heavy lifting and I interfaced
# their work with the code I already had for the vegetative indices.
#
# A mod of apply-indices.py. This means it is rather repetitive (if you comisder apply-indices.py as well) but seemed 

import cv2
import argparse
import pandas as pd
import vegetative_indices as vg
from pathlib import Path

# Compute index. This time we don't use a threshold, so things are
# simpler.
def computeIndex(img, index):
    indexImg = vg.computeIndexByName(img, index)

    # For now only the mean index value seems useful
    indexMean, _, _ = vg.summaryValues(indexImg)
    return indexMean

def process_images(input_dir, output_csv, selected_functions, normalize):
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

    # All available indexes. This is used to check that the relevant
    # index is one we can handle.
    all_functions = {
        'ExG':  computeIndex,
        'ExGR': computeIndex,
        'GLI':  computeIndex,
        'VARI': computeIndex,
        "RGBVI": computeIndex,
        "DGCI": computeIndex,
        "NGBDI": computeIndex,
        "BGR": computeIndex,
        "GRVI": computeIndex,
        "NRI": computeIndex,
        "NGI": computeIndex,
        "NBI": computeIndex,
        "SAVI": computeIndex,
        "GMR": computeIndex,
        # Add more index mappings here
    }
    
    # Filter to selected functions/indexes
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
        
        # Read image. Using OpenCV to read images to they are in BGR
        # format.
        img = cv2.imread(str(img_path))
       
        if img is None:
            print(f"Warning: Could not read {img_path.name}, skipping...")
            continue
        print(f" Size: {img.shape[:2]}")

        if normalize:
            # If needed, normalize image
            print(f" Normalizing: {img_path.name}")
            img  = vg.normalizeImage(img)
            
        row_data = {'filename': img_path.name}

        # Apply each function and store results     
        for index, func in functions.items():
            try:
                print(f" Applying {index} to {img_path.name}")
                #threshold_value = thresholds.get(index, None)
                #row_data[index] = func(img, index, threshold_value)
                row_data[index] = func(img, index)
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
    print(f"Processed {len(df)} images with {len(functions)} index(es)")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Process images and export results to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images with all indexes
  python average-indices.py /path/to/images output.csv
  
  # Process with specific indexes
  python average-indices.py /path/to/images output.csv -i index_1 index_2
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
        '-i', '--indexes',
        nargs='+',
        type=str,
        default=None,
        help='Names of indexes to apply (space-separated). If not specified, all indexes will be used.'
    )

    parser.add_argument(
        '-n', '--normalize',
        type=str,
        default=None,
        help='Should we normalize the images. Enter "True" or "true" for normalization (default: False)'
    )
    
    parser.add_argument(
        '--list_indexes',
        action='store_true',
        help='List all available indexes and exit'
    )
    
    args = parser.parse_args()
    
    # List indices if requested. 
    if args.list_indexes:
        print("Available indexes:")
        print("  - ExG")
        print("  - ExGR")
        print("  - GLI")
        print("  - VARI")
        print("  - RGBVI")
        print("  - DCGI")
        print("  - NGBDI")
        print("  - BGR")
        print("  - GRVI")
        print("  - NRI")
        print("  - NGI")
        print("  - NBI")
        print("  - SAVI")
        print("  - GMR")
        #print("\nAdd more indexes/functions in the script as needed.")
        return
    '''
    # Parse thresholds to map each index/function to its threshold value
    function_thresholds = {}
    
    if args.indexes and args.thresholds:
        # Check if number of thresholds matches number of indices
        if len(args.thresholds) != len(args.indexes):
            parser.error(f"Number of thresholds ({len(args.thresholds)}) must match number of functions ({len(args.functions)})")
        
        # Map each function to its threshold
        for func_name, threshold_str in zip(args.indexes, args.thresholds):
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
    elif args.thresholds and not args.indexes:
        parser.error("Cannot specify thresholds (-t) without specifying indexes (-i)")
    
    if function_thresholds:
        print(f"Index-threshold mapping:")
        for func, thresh in function_thresholds.items():
            print(f"  {func}: {thresh}")
    '''
    
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
            args.indexes,
            #function_thresholds,
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

