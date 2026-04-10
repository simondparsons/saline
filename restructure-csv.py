# restructure-csv.py
#
# Code to take the output of my image processing files and structure
# it to match the results from the saline soils trials.
#
# Simon Parsons
# University of Lincoln
# 26-03-31

# Written by Claude Sonnet 4.5 with some edits.

import pandas as pd
import argparse
import sys


def extract_and_reorder_rows(file1_path, file2_path, output_path):
    """
    Read two CSV files and create a third based on index mapping.
    
    Parameters:
    -----------
    file1_path : str
        Path to first CSV file (contains index numbers in first column)
    file2_path : str
        Path to second CSV file (source data to extract from)
    output_path : str
        Path to output CSV file
    
    Returns:
    --------
    output_df : DataFrame
        The resulting dataframe that was saved
    """
    print(f"Reading {file1_path}...")
    df1 = pd.read_csv(file1_path)
    
    print(f"Reading {file2_path}...")
    df2 = pd.read_csv(file2_path)
    
    # Get the index numbers from the first column of file1
    # Subtract 1 because row 1 should map to index 0 (first data row after header)
    indices = df1.iloc[:, 0].values - 1
    
    print(f"\nFirst CSV file has {len(df1)} rows")
    print(f"Second CSV file has {len(df2)} rows")
    print(f"Index range: {indices.min() + 1} to {indices.max() + 1}")
    
    # Validate indices
    if indices.min() < 0:
        print(f"Warning: Found index value < 1 in first CSV file")
    
    if indices.max() >= len(df2):
        print(f"Error: Index {indices.max() + 1} exceeds available rows in second CSV ({len(df2)})")
        sys.exit(1)
    
    # Extract rows from df2 based on indices
    print(f"\nExtracting {len(indices)} rows from second CSV file...")
    output_df = df2.iloc[indices].reset_index(drop=True)
    
    # Save to output file
    output_df.to_csv(output_path, index=False)
    print(f"Output saved to: {output_path}")
    print(f"Output has {len(output_df)} rows and {len(output_df.columns)} columns")
    
    return output_df


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
    --------
    args : Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Extract and reorder rows from a CSV file based on indices from another CSV file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python restructure-csv.py indices.csv data.csv output.csv
  python restructure-csv.py file1.csv file2.csv result.csv
  
The first CSV file should have index numbers (1-90) in its first column.
The second CSV file is the source data.
Rows are extracted from the second file based on indices in the first file.
        """
    )
    
    parser.add_argument(
        'index_file',
        help='CSV file containing index numbers in the first column'
    )
    
    parser.add_argument(
        'data_file',
        help='CSV file containing the data to extract from'
    )
    
    parser.add_argument(
        'output_file',
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--show-preview',
        action='store_true',
        help='Show preview of first few rows of output'
    )
    
    return parser.parse_args()


def create_sample_data():
    """
    Create sample CSV files for testing.
    """
    print("Creating sample CSV files for testing...\n")
    
    # Create first CSV with non-contiguous indices (45 rows)
    # Indices range from 1 to 90, not all present
    import numpy as np
    
    # Generate 45 random indices between 1 and 90
    np.random.seed(42)
    indices = sorted(np.random.choice(range(1, 91), size=45, replace=False))
    
    df1 = pd.DataFrame({
        'RowIndex': indices,
        'Description': [f'Item {i}' for i in range(45)]
    })
    
    # Create second CSV with 90 rows of data
    df2 = pd.DataFrame({
        'ID': range(1, 91),
        'Name': [f'Person_{i}' for i in range(1, 91)],
        'Value': np.random.randint(100, 1000, 90),
        'Category': [f'Cat_{i%5}' for i in range(1, 91)]
    })
    
    # Save sample files
    df1.to_csv('sample_indices.csv', index=False)
    df2.to_csv('sample_data.csv', index=False)
    
    print("Created sample_indices.csv:")
    print(df1.head(10))
    print(f"... ({len(df1)} total rows)\n")
    
    print("Created sample_data.csv:")
    print(df2.head(10))
    print(f"... ({len(df2)} total rows)\n")
    
    print("Sample indices to extract:", indices[:10], "...\n")
    
    return df1, df2


def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_arguments()
    
    print("="*70)
    print("CSV ROW EXTRACTION TOOL")
    print("="*70)
    print(f"Index file: {args.index_file}")
    print(f"Data file:  {args.data_file}")
    print(f"Output file: {args.output_file}")
    print("="*70 + "\n")
    
    # Extract and reorder rows
    try:
        output_df = extract_and_reorder_rows(
            args.index_file,
            args.data_file,
            args.output_file
        )
        
        # Show preview if requested
        if args.show_preview:
            print("\n" + "="*70)
            print("PREVIEW OF OUTPUT (first 10 rows):")
            print("="*70)
            print(output_df.head(10))
            print("="*70)
        
        print("\n✓ Success!")
        
    except FileNotFoundError as e:
        print(f"\nError: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # If no arguments provided, create sample data and show usage
    if len(sys.argv) == 1:
        print("No arguments provided. Creating sample data...\n")
        print("="*70)
        
        df1, df2 = create_sample_data()
        
        print("="*70)
        print("USAGE EXAMPLES:")
        print("="*70)
        print("python script.py sample_indices.csv sample_data.csv output.csv")
        print("python script.py sample_indices.csv sample_data.csv output.csv --show-preview")
        print("="*70)
        print("\nRun the command above to test with sample data!")
        
    else:
        main()
