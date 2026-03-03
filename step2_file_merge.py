import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # 0. Navigate to selected folder
    target_dir = input("Enter the path to the folder containing the CSV files (press Enter for current directory): ").strip()
    if not target_dir:
        target_dir = os.getcwd()
    
    target_path = Path(target_dir)
    if not target_path.exists() or not target_path.is_dir():
        print(f"Error: The directory '{target_dir}' does not exist.")
        return

    # Scan file names to find available methods
    thermo_methods = set()
    rhea_methods = set()
    
    for file in target_path.glob("*.csv"):
        thermo_match = re.match(r'THERMOCALC_RHEA_(.+)\.csv', file.name)
        if thermo_match:
            thermo_methods.add(thermo_match.group(1))
            
        rhea_match = re.match(r'RHEA_space_(.+)\.csv', file.name)
        if rhea_match:
            rhea_methods.add(rhea_match.group(1))
            
    # Get methods that exist in both sets
    valid_methods = list(thermo_methods.intersection(rhea_methods))
    
    if len(valid_methods) < 2:
        print("Not enough matching method pairs found in the directory to merge. Need at least 2.")
        print(f"Valid methods found: {valid_methods}")
        return
        
    print("\nFound the following matching method pairs:")
    for i, method in enumerate(valid_methods):
        print(f"[{i}] {method}")
        
    # Ask which two methods to merge
    try:
        m1_idx = int(input("\nEnter the number for the FIRST method to merge: "))
        m2_idx = int(input("Enter the number for the SECOND method to merge: "))
        method1 = valid_methods[m1_idx]
        method2 = valid_methods[m2_idx]
    except (ValueError, IndexError):
        print("Invalid selection. Exiting.")
        return

    print(f"\nProceeding with '{method1}' and '{method2}'...")

    # Define file paths
    tc_file1 = target_path / f"THERMOCALC_RHEA_{method1}.csv"
    tc_file2 = target_path / f"THERMOCALC_RHEA_{method2}.csv"
    rs_file1 = target_path / f"RHEA_space_{method1}.csv"
    rs_file2 = target_path / f"RHEA_space_{method2}.csv"

    # 1 & 2. Load and sort the files according to Comp_point
    print("Loading and sorting data...")
    df_tc1 = pd.read_csv(tc_file1).sort_values(by='Comp_point').reset_index(drop=True)
    df_tc2 = pd.read_csv(tc_file2).sort_values(by='Comp_point').reset_index(drop=True)
    
    df_rs1 = pd.read_csv(rs_file1)
    df_rs2 = pd.read_csv(rs_file2)

    # 3. Merge THERMOCALC files 
    # pd.concat natively aligns columns and handles dropping the redundant header
    df_tc_merged = pd.concat([df_tc1, df_tc2], ignore_index=True)
    
    # Rearrange Comp_point
    df_tc_merged['Comp_point'] = range(1, len(df_tc_merged) + 1)

    # 4. Merge RHEA space files
    df_rs_merged = pd.concat([df_rs1, df_rs2], ignore_index=True)
    
    # Rearrange Comp_point
    df_rs_merged['Comp_point'] = range(1, len(df_rs_merged) + 1)

    # --- Adjustments for Printablitiy Map code ---
    # Add the unit column with 'at%' for all rows
    df_rs_merged['Unit [at% or wt%]'] = 'at%'
    
    # Insert an unnamed column (empty string header) at position 0 containing the index
    df_rs_merged.insert(0, '', df_rs_merged.index)
    
    # 5. Check if the composition of all elements match between the two new files
    print("Verifying elemental compositions match across both merged files...")
    
    # Identify common columns (assuming element columns are the overlapping ones not named Comp_point etc.)
    exclude_cols = ['Comp_point', 'Unit [at% or wt%]', 'Unnamed: 0']
    common_cols = [col for col in df_tc_merged.columns if col in df_rs_merged.columns and col not in exclude_cols]
    
    if not common_cols:
        print("Warning: Could not find overlapping elemental columns to compare.")
    else:
        # Use numpy.isclose for float comparison to avoid minor precision mismatch errors
        try:
            tc_elements = df_tc_merged[common_cols].astype(float)
            rs_elements = df_rs_merged[common_cols].astype(float)
            
            # Check if all values match within a small tolerance
            matches = np.isclose(tc_elements, rs_elements, equal_nan=True)
            if matches.all():
                print(f"Success! Compositions for {common_cols} match perfectly across all {len(df_tc_merged)} lines.")
            else:
                mismatch_count = (~matches).sum()
                print(f"Warning: Found {mismatch_count} discrepancies in composition between the two files.")
        except ValueError as e:
            print(f"Could not perform numeric comparison on common columns: {e}")

    # 6. Save both files in the Printability_Map folder
    output_dir = target_path.parent / "Data_Space"
    output_dir.mkdir(parents=True, exist_ok=True) # Create folder if it doesn't exist
    
    out_tc_path = output_dir / "THERMOCALC_RHEA.csv"
    out_rs_path = output_dir / "RHEA.csv"
    
    df_tc_merged.to_csv(out_tc_path, index=False)
    df_rs_merged.to_csv(out_rs_path, index=False)
    
    print(f"\nComplete! Files saved to:")
    print(f" - {out_tc_path}")
    print(f" - {out_rs_path}")

if __name__ == "__main__":
    main()
