#!/usr/bin/env python3
"""
aggregate_multiple_subject_data.py - Multi-Subject Psychophysical Data Aggregator

Purpose:
Combines raw psychophysical data files from multiple subjects into standardized
aggregate datasets for group-level analysis while maintaining the original
experimental structure.

Key Functionality:
- Interactively collects subject names and output group name
- Validates existence of input data files
- Aggregates trial counts across subjects while preserving stimulus parameters
- Maintains original condition/repetition folder hierarchy
- Generates output files compatible with downstream analysis pipelines

Input Requirements:
- Requires pre-processed raw_data_*.txt files from individual subjects
- Files must be in: BASE_FOLDER/Analysis/[subject]/[condition]/Repetition_[X]/
- Expects tab-delimited files with standardized columns:
  * Standard lightness/background values
  * Comparison background values
  * Lightness values
  * Number of comparison choices
  * Trial counts per condition

Output Generated:
- Creates aggregated raw_data_*.txt files in:
  BASE_FOLDER/Analysis/[group_name]/[condition]/Repetition_[X]/
- Preserves all original columns while summing:
  * n_cmp_chsn_* (comparison choice counts)
  * n_trials_* (trial counts)
- Maintains identical formatting as input files

Operation:
1. Sums choice and trial counts across subjects
2. Preserves standard stimulus values from first subject
3. Maintains condition/repetition folder structure
4. Handles missing files gracefully

Dependencies:
- Python 3.x
- pandas package
- Pre-processed data from process_subject_data.py

Usage:
$ python aggregate_subject_data.py
(interactively prompts for subject names and output group name)

Example:
$ python aggregate_subject_data.py
Enter subject names: subj01, subj02, subj03
Enter group name: group1

Output Files:
Analysis/group1/condition_1/Repetition_1/raw_data_group1_condition_1_Repetition_1.txt
...

Author: Vijay Singh
Created: April 26 2025
Version: 1.0
"""

import os
import pandas as pd

# Constants
BASE_FOLDER = "/Users/vsingh1/Documents/BackgroundEffect"
CONDITIONS = ["condition_1", "condition_2", "condition_3"]
NUM_REPETITIONS = 1

def get_user_input():
    """Get subject names and output name from user input"""
    print("\nData Aggregation Tool")
    print("---------------------\n")
    
    # Get subject names
    while True:
        subject_names = input("Enter subject names to aggregate (comma separated): ").strip()
        subject_names = [name.strip() for name in subject_names.split(',') if name.strip()]
        
        if len(subject_names) < 2:
            print("Please enter at least 2 valid subject names.")
            continue
            
        # Verify at least some data exists for these subjects
        valid_subjects = []
        for subject in subject_names:
            subject_path = os.path.join(BASE_FOLDER, "Analysis", subject)
            if os.path.exists(subject_path):
                valid_subjects.append(subject)
            else:
                print(f"Warning: No data folder found for subject '{subject}'")
        
        if len(valid_subjects) >= 2:
            subject_names = valid_subjects
            break
        else:
            print("Need at least 2 subjects with valid data folders.")
    
    # Get output name
    while True:
        output_name = input("Enter name for combined output (e.g., 'group1' or 'subj1_subj2'): ").strip()
        if not output_name:
            print("Please enter a valid name.")
            continue
        
        output_path = os.path.join(BASE_FOLDER, "Analysis", output_name)
        if os.path.exists(output_path):
            print(f"Warning: Output folder '{output_path}' already exists!")
            overwrite = input("Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                continue
        
        break
    
    return subject_names, output_name

def aggregate_subjects_data(subject_names, output_subject_name):
    """
    Aggregate raw data files from multiple subjects and save in a new combined subject folder
    maintaining the original condition/repetition folder structure.
    """
    print(f"\nStarting aggregation for subjects: {', '.join(subject_names)}")
    print(f"Output will be saved under: {output_subject_name}\n")
    
    # Process each condition
    for condition in CONDITIONS:
        print(f"\nProcessing condition: {condition}")
        
        # Process each repetition
        for repetition in range(1, NUM_REPETITIONS + 1):
            print(f"  Repetition {repetition}...", end=' ')
            all_data = []
            found_files = 0
            
            # Collect data from each subject
            for subject_name in subject_names:
                # Construct input file path
                input_file = os.path.join(
                    BASE_FOLDER,
                    "Analysis",
                    subject_name,
                    condition,
                    f"Repetition_{repetition}",
                    f"raw_data_{subject_name}_{condition}_Repetition_{repetition}.txt"
                )
                
                if os.path.exists(input_file):
                    try:
                        df = pd.read_csv(input_file, sep='\t')
                        all_data.append(df)
                        found_files += 1
                    except Exception as e:
                        print(f"\nError reading {input_file}: {str(e)}")
                else:
                    print(f"\nFile not found: {input_file}")
            
            if found_files < 2:
                print(f"Only {found_files} files found - skipping {condition} Repetition_{repetition}")
                continue
            
            # Aggregate the data by summing the relevant columns
            aggregated_df = all_data[0].copy()
            
            # Columns to sum (n_cmp_chsn and n_trials)
            sum_cols = [col for col in aggregated_df.columns 
                       if col.startswith('n_cmp_chsn_') or col.startswith('n_trials_')]
            
            # Sum the data across subjects
            for df in all_data[1:]:
                for col in sum_cols:
                    aggregated_df[col] += df[col]
            
            # Create output directory structure
            output_dir = os.path.join(
                BASE_FOLDER,
                "Analysis",
                output_subject_name,
                condition,
                f"Repetition_{repetition}"
            )
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the aggregated data
            output_file = os.path.join(
                output_dir,
                f"raw_data_{output_subject_name}_{condition}_Repetition_{repetition}.txt"
            )
            
            aggregated_df.to_csv(output_file, sep='\t', index=False, float_format='%.4f')
            print(f"Saved aggregated data ({found_files} subjects combined)")

def main():
    # Get user input
    subject_names, output_name = get_user_input()
    
    # Perform aggregation
    aggregate_subjects_data(subject_names, output_name)
    
    # Final output path
    output_path = os.path.join(BASE_FOLDER, "Analysis", output_name)
    print(f"\nAggregation complete! Results saved in:\n{output_path}")
    print("\nYou can now use these aggregated files for further analysis.")

if __name__ == "__main__":
    main()