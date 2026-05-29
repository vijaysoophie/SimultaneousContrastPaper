#!/usr/bin/env python3
"""
aggregate_multiple_subject_data_average.py - Multi-Subject Psychophysical Data Aggregator

Purpose:
Combines raw psychophysical data files from multiple subjects into standardized
aggregate datasets with mean and standard error of the mean for group-level analysis.

Key Functionality:
- Interactively collects subject names and output group name
- Validates existence of input data files
- Calculates mean and SEM across subjects for comparison choices (n_cmp_chsn)
- Rounds mean values to integers for compatibility with psignifit
- Preserves n_trials values (same across subjects, no SEM needed)
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
- Preserves all original columns while computing:
  * MEAN of n_cmp_chsn_* (rounded to integers for psignifit compatibility)
  * SEM of n_cmp_chsn_* (standard error of the mean, kept as float)
  * n_trials_* preserved as-is (same across subjects)
- Maintains identical formatting as input files

Operation:
1. Calculates mean and SEM across subjects for choice counts only
2. Rounds mean values to nearest integer
3. Preserves trial counts (identical across subjects)
4. Preserves standard stimulus values from first subject
5. Maintains condition/repetition folder structure
6. Handles missing files gracefully

Dependencies:
- Python 3.x
- pandas package
- numpy package
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
Version: 2.2 (Rounded means to integers for psignifit compatibility)
"""

import os
import pandas as pd
import numpy as np

# Constants
BASE_FOLDER = os.getcwd()
CONDITIONS = ["condition_1", "condition_2", "condition_3"]
NUM_REPETITIONS = 1

def get_user_input():
    """Get subject names and output name from user input"""
    print("\nData Aggregation Tool (Mean + SEM for comparison choices only)")
    print("--------------------------------------------------------------\n")
    
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
    Aggregate raw data files from multiple subjects by calculating mean and SEM
    for comparison choices (n_cmp_chsn) only. Trial counts (n_trials) are preserved
    as-is since they are identical across subjects.
    
    IMPORTANT: Mean values are rounded to integers for compatibility with psignifit
    which requires integer counts for the number of correct responses.
    """
    print(f"\nStarting aggregation for subjects: {', '.join(subject_names)}")
    print(f"Output will be saved under: {output_subject_name}\n")
    print("Calculating MEAN and STANDARD ERROR OF THE MEAN (SEM) for comparison choices only")
    print("NOTE: Mean values will be rounded to integers for psignifit compatibility\n")
    
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
            
            # Create a DataFrame for aggregated results (start with first subject's structure)
            aggregated_df = all_data[0].copy()
            
            # Identify columns for comparison choices (n_cmp_chsn) and trials (n_trials)
            cmp_chsn_cols = [col for col in aggregated_df.columns if col.startswith('n_cmp_chsn_')]
            trials_cols = [col for col in aggregated_df.columns if col.startswith('n_trials_')]
            
            # For comparison choice columns, calculate mean and SEM across subjects
            for col in cmp_chsn_cols:
                # Collect values from all subjects for this column
                values_across_subjects = [df[col].values for df in all_data]
                values_array = np.array(values_across_subjects)
                
                # Calculate mean and SEM
                mean_values = np.mean(values_array, axis=0)
                sem_values = np.std(values_array, axis=0, ddof=1) / np.sqrt(found_files)
                
                # Round mean values to nearest integer for psignifit compatibility
                mean_values_rounded = np.round(mean_values).astype(int)
                
                # Store rounded mean and SEM in the aggregated DataFrame
                aggregated_df[col] = mean_values_rounded
                aggregated_df[f'{col}_SEM'] = sem_values
                
                # Print rounding information for debugging
                if np.any(np.abs(mean_values - mean_values_rounded) > 0.01):
                    print(f"\n    Rounded {col} means to integers for compatibility")
                    print(f"      Original means (sample): {mean_values[:3]}")
                    print(f"      Rounded means (sample): {mean_values_rounded[:3]}")
            
            # For trial count columns, just take the values from the first subject
            # (assuming they are identical across all subjects)
            for col in trials_cols:
                # Verify that trial counts are consistent across subjects
                if found_files > 1:
                    first_values = all_data[0][col].values
                    consistent = True
                    for df in all_data[1:]:
                        if not np.array_equal(first_values, df[col].values):
                            consistent = False
                            print(f"\n  Warning: {col} values differ across subjects for {condition} Repetition_{repetition}")
                            break
                    if not consistent:
                        print(f"  Using values from {subject_names[0]} (first subject) for {col}")
                
                # Keep the values from the first subject (they should be the same)
                aggregated_df[col] = all_data[0][col].values
            
            # Create output directory structure
            output_dir = os.path.join(
                BASE_FOLDER,
                "Analysis",
                output_subject_name,
                condition,
                f"Repetition_{repetition}"
            )
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the aggregated data with rounded mean values (for compatibility with analysis scripts)
            output_file = os.path.join(
                output_dir,
                f"raw_data_{output_subject_name}_{condition}_Repetition_{repetition}.txt"
            )
            
            # Create a version with only the rounded mean values for n_cmp_chsn columns
            mean_only_df = aggregated_df.copy()
            # Remove SEM columns for the main output file
            sem_cols = [col for col in mean_only_df.columns if col.endswith('_SEM')]
            mean_only_df = mean_only_df.drop(columns=sem_cols)
            
            mean_only_df.to_csv(output_file, sep='\t', index=False, float_format='%.0f')  # No decimal places for integers
            print(f"Saved mean data (rounded to integers, {found_files} subjects combined)")
            
            # Save a separate file with SEM values for error analysis
            sem_file = output_file.replace('raw_data_', 'raw_data_with_sem_')
            # Save full DataFrame with both rounded means and SEMs
            # For the SEM file, keep means as floats to show the true mean before rounding
            aggregated_df_with_original_means = aggregated_df.copy()
            # But we need to add back the original unrounded means for reference
            for i, col in enumerate(cmp_chsn_cols):
                # Recalculate original means (without rounding) for the SEM file
                values_across_subjects = [df[col].values for df in all_data]
                values_array = np.array(values_across_subjects)
                original_means = np.mean(values_array, axis=0)
                aggregated_df_with_original_means[f'{col}_original_mean'] = original_means
            
            aggregated_df_with_original_means.to_csv(sem_file, sep='\t', index=False, float_format='%.4f')
            print(f"  Also saved mean+SEM data to: {os.path.basename(sem_file)}")
            print(f"    (SEM file includes original unrounded means as {col}_original_mean)")

def main():
    # Get user input
    subject_names, output_name = get_user_input()
    
    # Perform aggregation
    aggregate_subjects_data(subject_names, output_name)
    
    # Final output path
    output_path = os.path.join(BASE_FOLDER, "Analysis", output_name)
    print(f"\nAggregation complete! Results saved in:\n{output_path}")
    print("\nFiles saved:")
    print("  - raw_data_*.txt: Rounded integer means for n_cmp_chsn (compatible with psignifit)")
    print("  - raw_data_with_sem_*.txt: Rounded means + SEM columns + original unrounded means")
    print("\nNote: n_trials values are preserved as-is (assumed identical across subjects)")
    print("Note: Means are rounded to integers to satisfy psignifit's integer requirement")
    print("\nYou can now use the mean data files for further analysis.")

if __name__ == "__main__":
    main()