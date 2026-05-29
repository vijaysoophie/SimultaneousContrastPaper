#!/usr/bin/env python3

"""
save_psychometric_data_for_subject.py - Psychometric Function Data Processor for Lightness Perception Experiments

Purpose:
Processes individual subject data from lightness perception experiments to generate:
1. Raw data files formatted for psychometric function analysis
2. Organized output maintaining original experimental structure

Key Functionality:
- Parses experimental trial data with RGBA color values
- Calculates response correctness and trial statistics
- Organizes data by background conditions and lightness levels
- Generates standardized output files for psychometric analysis

Input Requirements:
- Input files must be in: BASE_FOLDER/Data/[subject]/[condition]/Repetition_[X]/
- Expected file format: 
  [order],[expected],[response],[RGBA],[RGBA],[RGBA],[RGBA]
- Requires one file per condition/repetition

Output Generated:
- Creates Analysis/[subject]/[condition]/Repetition_[X]/ folders
- Saves tab-delimited files named raw_data_[subject]_[condition]_Repetition_[X].txt
- Output columns include:
  * Standard lightness/background values
  * Comparison background values
  * Lightness values
  * Number of comparison choices
  * Trial counts per condition

Dependencies:
- Python 3.x
- Libraries: numpy, pandas, matplotlib, psignifit
- File structure matching specified constants

Usage:
$ python process_subject_data.py [subject_name]
or
$ python process_subject_data.py  # (will prompt for subject name)

Example:
$ python process_subject_data.py subject01

Author: Vijay Singh
Created: April 26 2025
Version: 1.0
"""


import os
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Constants
BASE_FOLDER = os.getcwd()
NUM_REPETITIONS = 1
CONDITIONS = ["condition_0","condition_1", "condition_2", "condition_3"]

# Set global font sizes
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12
})

def extract_float_from_rgba(rgba_str):
    """Helper function to extract float from RGBA string"""
    if isinstance(rgba_str, str):
        return float(rgba_str.split('(')[1].split(',')[0])
    return rgba_str

def analyze_data_file(file_path):
    """Analyze a single data file and return the processed DataFrame"""
    data = []
    pattern = re.compile(r'(\d+),\s*(\d+),\s*(\d+),\s*(RGBA\([^)]+\)),\s*(RGBA\([^)]+\)),\s*(RGBA\([^)]+\)),\s*(RGBA\([^)]+\))')

    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = pattern.match(line.strip())
                if match:
                    order_number = int(match.group(1))
                    expected_response = int(match.group(2))
                    subject_choice = int(match.group(3))
                    rgba_values = list(match.groups()[3:])
                    data.append([order_number, expected_response, subject_choice] + rgba_values)

        columns = ['Order_number', 'Expected_response', 'Subject_choice', 'Square1', 'Background1', 'Square2', 'Background2']
        df = pd.DataFrame(data, columns=columns)

        # Process RGBA values
        for col in ['Square1', 'Background1', 'Square2', 'Background2']:
            df[col] = df[col].apply(extract_float_from_rgba)

        df = df.sort_values(by=['Order_number'], ascending=[True])
        df['response_correct'] = np.where(df['Expected_response'] == df['Subject_choice'], 1, 0)

        return df
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def save_raw_data(std_lightness, std_bkg_val, cmp_bkg_vals, lightness_vals, results, n_trials_per_comp_level, output_folder, subject_name, condition_name, repetition):
    """Save the raw data in the specified column order"""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    
    # Create a DataFrame to hold all the data
    data_dict = {
        'std_lightness': np.full_like(lightness_vals, std_lightness),
        'std_bkg_val': np.full_like(lightness_vals, std_bkg_val)
    }
    
    # Add each background condition's data
    for i, result in enumerate(results.itertuples()):
        # Add comparison background values (repeated for each lightness value)
        data_dict[f'cmp_bkg_vals_bkg_{result.background:.4f}'] = np.full_like(lightness_vals, result.background)
        # Add lightness values
        data_dict[f'lightness_vals_bkg_{result.background:.4f}'] = lightness_vals
        # Convert n_comp_chsn to integers
        data_dict[f'n_cmp_chsn_bkg_{result.background:.4f}'] = result.n_comp_chsn.astype(int)
        # Add trials per comparison level for each background
        data_dict[f'n_trials_bkg_{result.background:.4f}'] = np.full_like(result.n_comp_chsn, n_trials_per_comp_level, dtype=int)
    
    raw_data_df = pd.DataFrame(data_dict)
    
    # Reorder columns to match the specified format for each background
    ordered_columns = []
    for result in results.itertuples():
        bg_key = f"{result.background:.4f}"
        ordered_columns.extend([
            'std_lightness',
            'std_bkg_val',
            f'cmp_bkg_vals_bkg_{bg_key}',
            f'lightness_vals_bkg_{bg_key}',
            f'n_cmp_chsn_bkg_{bg_key}',
            f'n_trials_bkg_{bg_key}'
        ])
    
    # Remove duplicate column names while preserving order
    seen = set()
    unique_ordered_columns = [col for col in ordered_columns if not (col in seen or seen.add(col))]
    
    # Select only the columns we want in the order we want
    raw_data_df = raw_data_df[unique_ordered_columns]
    
    # Save to file
    raw_data_filename = f"raw_data_{subject_name}_{condition_name}_Repetition_{repetition}.txt"
    raw_data_path = os.path.join(output_folder, raw_data_filename)
    raw_data_df.to_csv(raw_data_path, sep='\t', index=False, float_format='%.4f')
    print(f"Saved raw data to {raw_data_path}")

def perform_psychometric_analysis(df, output_folder, subject_name, condition_name, repetition):
    """Perform the psychometric analysis and save the plot, thresholds, and raw data"""
    if df is None or df.empty:
        print("No valid data to analyze")
        return

    # Analysis parameters
    print(df['Square1'].unique())
    n_comp_levels = len(df['Square1'].unique())
    cmp_bkg_vals = np.sort((df['Background1']).unique())
    n_backgrounds = len(cmp_bkg_vals)
    
    std_lightness = df['Square1'].mode()[0]
    std_bkg_val = df['Background1'].mode()[0]
    std_bkg_not_inclded = np.sum(df['Background1'] == df['Background2']) == 0
    
    if std_bkg_not_inclded:
        n_backgrounds -= 1
        cmp_bkg_vals = np.delete(cmp_bkg_vals, np.where(cmp_bkg_vals == std_bkg_val))
    
    n_trials = df.shape[0] // n_backgrounds
    lightness_vals = np.sort((df['Square1']).unique())
    n_trials_per_comp_level = n_trials // n_comp_levels

    # Calculate results
    results = []
    for cmp_val in cmp_bkg_vals:
        subset = df[((df['Background1'] == cmp_val) & (df['Background2'] == std_bkg_val)) | 
                 ((df['Background1'] == std_bkg_val) & (df['Background2'] == cmp_val))]
        response_correct_values = subset['response_correct'].values
        fraction_correct = response_correct_values.reshape(-1,n_comp_levels).sum(0)
        n_comp_chsn1 = n_trials_per_comp_level*np.ones(n_comp_levels//2) - fraction_correct[0:n_comp_levels//2]
        n_comp_chsn2 = fraction_correct[n_comp_levels//2:n_comp_levels]
        n_comp_chsn = np.append(n_comp_chsn1, n_comp_chsn2)
        results.append({'background':cmp_val, 'fraction_correct': fraction_correct, 'n_comp_chsn': n_comp_chsn})
    results = pd.DataFrame(results)

    # Save the raw data with the new format
    save_raw_data(std_lightness, std_bkg_val, cmp_bkg_vals, lightness_vals, results, n_trials_per_comp_level, 
                 output_folder, subject_name, condition_name, repetition)   

def process_subject_data(subject_name, base_folder, conditions, num_repetitions):
    """Process all data files for a given subject"""
    for condition in conditions:
        for repetition in range(1, num_repetitions + 1):
            # Construct input data paths
            input_folder = os.path.join(
                base_folder, 
                "Data", 
                subject_name, 
                condition, 
                f"Repetition_{repetition}"
            )

            # Construct output file paths
            output_folder = os.path.join(
                base_folder, 
                "Analysis", 
                subject_name, 
                condition, 
                f"Repetition_{repetition}"
            )
            file_name = f"{subject_name}_{condition}_Repetition_{repetition}.txt"
            file_path = os.path.join(input_folder, file_name)

            if os.path.exists(file_path):
                print(f"\nProcessing {subject_name}, {condition}, repetition {repetition}")
                df = analyze_data_file(file_path)
                if df is not None:
                    perform_psychometric_analysis(df, output_folder, subject_name, condition, repetition)
            else:
                print(f"File not found: {file_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze data for a subject across all conditions.')
    parser.add_argument('subject', nargs='?', help='Name of the subject to analyze', default=None)
    
    args = parser.parse_args()
    
    # If subject not provided as argument, prompt for it
    if args.subject is None:
        args.subject = input("Please enter the subject name (e.g., subject_apple): ").strip()
        while not args.subject:
            print("Error: Subject name cannot be empty!")
            args.subject = input("Please enter the subject name (e.g., subject_apple): ").strip()
    
    print(f"\nAnalyzing data for subject: {args.subject}")
    
    # Process all data for the subject
    process_subject_data(
        args.subject, 
        BASE_FOLDER, 
        CONDITIONS, 
        NUM_REPETITIONS
    )

if __name__ == "__main__":
    main()