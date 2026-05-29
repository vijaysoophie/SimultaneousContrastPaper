#!/usr/bin/env python3
"""
analyze_aggregated_data_with_errors.py - Psychometric Function Analysis for Aggregated Data with Error Bars

Purpose:
Analyzes aggregated psychophysical data (mean ± SEM) from multiple subjects to:
1. Calculate perceptual thresholds at 50% and 76% performance levels
2. Generate publication-quality plots of psychometric functions with error bars
3. Save threshold data for statistical analysis

Key Functionality:
- Loads aggregated raw data files with SEM values from aggregate_multiple_subject_data.py
- Fits psychometric functions using psignifit toolbox on mean data (converted to integers)
- Calculates thresholds at multiple performance levels
- Generates visualizations showing mean data points with SEM error bars and fitted curves
- Saves results in structured output files

Input Requirements:
- Input files must be in: BASE_FOLDER/Analysis/[subject]/[condition]/Repetition_[X]/
- Expects files named: raw_data_with_sem_[subject]_[condition]_Repetition_[X].txt
- Requires tab-delimited files with standardized columns including:
  * Standard lightness/background values
  * Comparison background values
  * Lightness values
  * MEAN of comparison choices (n_cmp_chsn_*)
  * SEM of comparison choices (n_cmp_chsn_*_SEM)
  * Trial counts per condition (n_trials_*)

Output Generated:
- PNG plots of psychometric functions:
  * Shows mean data points with SEM error bars
  * Shows fitted curves
  * Marks threshold points with annotations
  * Includes standard stimulus information
- Tab-delimited threshold files containing:
  * 50% and 76% threshold values
  * Curve width parameters
  * Standard and comparison background values

Dependencies:
- Python 3.x
- Required packages: numpy, pandas, matplotlib, psignifit
- Aggregated data from aggregate_multiple_subject_data.py

Usage:
$ python analyze_aggregated_data_with_errors.py [group_name]
or
$ python analyze_aggregated_data_with_errors.py  # (prompts for group name)

Example:
$ python analyze_aggregated_data_with_errors.py group1

Author: Vijay Singh
Created: April 26 2025
Version: 1.3 (Fixed integer conversion with explicit type checking)
"""


import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psignifit as ps

# Constants
BASE_FOLDER = os.getcwd()
NUM_REPETITIONS = 1
CONDITIONS = ["condition_1", "condition_2", "condition_3"]

# RGB to Luminance conversion
MAX_RGB = 255
MAX_LUMINANCE = 105.6  # cd/m^2

def rgb_to_luminance(rgb_value):
    """
    Convert RGB value (0-255) to luminance (cd/m^2)
    RGB of 255 corresponds to 105.6 cd/m^2
    """
    return (rgb_value / MAX_RGB) * MAX_LUMINANCE

def luminance_to_rgb(luminance_value):
    """
    Convert luminance (cd/m^2) back to RGB value (0-255)
    Useful for display purposes
    """
    return (luminance_value / MAX_LUMINANCE) * MAX_RGB

# Set global font sizes
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 9
})

def load_aggregated_data_file(file_path):
    """Load the aggregated data file (with SEM) created by aggregate_multiple_subject_data.py"""
    try:
        df = pd.read_csv(file_path, sep='\t')
        return df
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def extract_data_for_background(raw_df, bg_val):
    """Extract relevant data for a specific background value, including SEM"""
    bg_key = f"{bg_val:.4f}"
    
    data = {
        'lightness_vals': raw_df[f'lightness_vals_bkg_{bg_key}'].values,
        'n_comp_chsn_mean': raw_df[f'n_cmp_chsn_bkg_{bg_key}'].values,
        'n_comp_chsn_sem': raw_df[f'n_cmp_chsn_bkg_{bg_key}_SEM'].values if f'n_cmp_chsn_bkg_{bg_key}_SEM' in raw_df.columns else None,
        'n_trials': raw_df[f'n_trials_bkg_{bg_key}'].values[0],
        'std_lightness': raw_df['std_lightness'].values[0],
        'std_bkg_val': raw_df['std_bkg_val'].values[0],
        'cmp_bkg_val': raw_df[f'cmp_bkg_vals_bkg_{bg_key}'].values[0]
    }
    
    return data

BACKGROUND_STYLES = {
    0: {'color': "#1f77b4", 'marker': "o"},  # blue circle
    1: {'color': "#ff7f0e", 'marker': "s"},   # orange square
    2: {'color': "#2ca02c", 'marker': "^"}    # green triangle
}
def perform_psychometric_analysis(raw_df, output_folder, group_name, condition_name, repetition):
    """
    Perform the psychometric analysis using aggregated data with error bars
    """
    if raw_df is None or raw_df.empty:
        print("No valid data to analyze")
        return

    # Find all unique background values in the data
    bg_columns = [col for col in raw_df.columns if 'cmp_bkg_vals_bkg_' in col]
    cmp_bkg_vals = [float(col.split('_')[-1]) for col in bg_columns]
    
    # Get standard values (should be same for all rows)
    std_lightness_rgb = raw_df['std_lightness'].values[0]
    std_bkg_val_rgb = raw_df['std_bkg_val'].values[0]
    
    # Convert standard values to luminance
    std_lightness_lum = rgb_to_luminance(std_lightness_rgb)
    std_bkg_val_lum = rgb_to_luminance(std_bkg_val_rgb)
    
    # Prepare thresholds data for saving
    thresholds_data = []
    
    # Plotting
    plt.figure(figsize=(5,3.4))
    custom_bounds = {'lambda':(0.0, 0.1)}
    ii=0 
    for bg_val_rgb in cmp_bkg_vals:
        # Extract data for this background
        bg_data = extract_data_for_background(raw_df, bg_val_rgb)
        
        # Convert lightness values from RGB to luminance
        lightness_vals_rgb = bg_data['lightness_vals']
        lightness_vals_lum = rgb_to_luminance(lightness_vals_rgb)
        
        n_comp_chsn_mean = bg_data['n_comp_chsn_mean']
        n_comp_chsn_sem = bg_data['n_comp_chsn_sem']
        n_trials_per_comp_level = bg_data['n_trials']
        cmp_bkg_val_rgb = bg_data['cmp_bkg_val']
        
        # Convert comparison background to luminance (for legend)
        cmp_bkg_val_lum = rgb_to_luminance(cmp_bkg_val_rgb)
        
        # For psignifit, we need integer counts.
        # Convert mean values to integers by rounding and explicitly converting to int
        n_comp_chsn_int = np.round(n_comp_chsn_mean).astype(np.int32)
        
        # Ensure all values are integers and positive
        n_comp_chsn_int = np.maximum(n_comp_chsn_int, 0)  # No negative values
        
        # Debug print
        print(f"  Background {cmp_bkg_val_lum:.1f}:")
        print(f"    Original means: {n_comp_chsn_mean}")
        print(f"    Rounded ints:   {n_comp_chsn_int}")
        print(f"    Data type: {n_comp_chsn_int.dtype}")
        
        # Generate fine-grained values for smooth curve (in luminance space)
        lightness_vals_fine_lum = np.linspace(lightness_vals_lum.min(), lightness_vals_lum.max(), 500)
        
        # Prepare data for psignifit - create a list of lists to ensure proper types
        data_list = []
        for i in range(len(lightness_vals_lum)):
            data_list.append([
                float(lightness_vals_lum[i]),  # stimulus intensity
                int(n_comp_chsn_int[i]),        # number correct (explicit int)
                int(n_trials_per_comp_level)    # number of trials (explicit int)
            ])
        
        data_psignifit = np.array(data_list)
        
        print(f"    Data array shape: {data_psignifit.shape}")
        print(f"    Data array dtypes: {data_psignifit.dtype}")
        print(f"    Column 2 dtype: {data_psignifit[:, 1].dtype}")
        
        try:
            # Fit psychometric function using integer values
            result = ps.psignifit(data_psignifit, sigmoid='norm', bounds=custom_bounds)
        except Exception as e:
            print(f"  ERROR fitting background {cmp_bkg_val_lum:.1f}: {str(e)}")
            print(f"    Data sent to psignifit: {data_psignifit}")
            continue
        
        # Get thresholds at different levels (these are already in luminance units)
        threshold_50_lum = result.threshold(0.50)[0]
        threshold_76_lum = result.threshold(0.76)[0]
        width_lum = result.parameter_estimate['width']
        
        # Store thresholds data (keep in luminance units)
        thresholds_data.append({
            'standard_rgb': std_lightness_rgb,
            'standard_luminance': std_lightness_lum,
            'background_rgb': bg_val_rgb,
            'background_luminance': cmp_bkg_val_lum,
            'threshold_50_luminance': threshold_50_lum,
            'threshold_76_luminance': threshold_76_lum,
            'width_luminance': width_lum
        })
        
        # Generate fitted curve (in luminance space)
        y_fit = n_trials_per_comp_level * result.proportion_correct(lightness_vals_fine_lum)
        style = BACKGROUND_STYLES.get(ii, {'color': "#d62728", 'marker': "x"})  # default to red x if not found
        color = style['color']
        ii +=1

        # Plot mean data points with error bars (SEM)
        if n_comp_chsn_sem is not None:
            plt.errorbar(lightness_vals_lum, n_comp_chsn_mean, yerr=n_comp_chsn_sem,
                        fmt='o', markersize=8, capsize=4, capthick=1, elinewidth=1,
                        color=color, ecolor=color, alpha=0.7,
                        label=f'{cmp_bkg_val_lum:.1f}')
        else:
            # Fallback if no SEM available
            plt.plot(lightness_vals_lum, n_comp_chsn_mean, 'o', 
                    markersize=8, color=color, label=f'{cmp_bkg_val_lum:.1f}')
        
        # Plot fitted curve
        plt.plot(lightness_vals_fine_lum, y_fit, '-', linewidth=2, color=color)
        
        # Mark threshold points on the curve
        plt.plot(threshold_50_lum, n_trials_per_comp_level*0.5, 's', 
                markersize=8, markeredgecolor='black', markerfacecolor=color)

    # Add legend with title
    if len(thresholds_data) > 0:
        legend = plt.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=9)
        legend.set_title("$B_c$ (cd/m²)", prop={'size': 9})
        
        plt.xlabel('Comparison Luminance (cd/m²)', fontsize=14)
        plt.ylabel(f'Number Comparison Chosen\n(N = {n_trials_per_comp_level})', fontsize=14)
        plt.title(f'$L_s$ = {std_lightness_lum:.1f} cd/m², $B_s$ = {std_bkg_val_lum:.1f} cd/m²', fontsize=12)
        plt.tight_layout()
        
        # Save the figure with group name in filename
        plot_filename = f"{group_name}_{condition_name}_Repetition_{repetition}_with_errors.png"
        plot_path = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved plot to {plot_path}")
        
        # Save thresholds data to text file
        thresholds_df = pd.DataFrame(thresholds_data)
        thresholds_filename = f"thresholds_{group_name}_{condition_name}_Repetition_{repetition}.txt"
        thresholds_path = os.path.join(output_folder, thresholds_filename)
        thresholds_df.to_csv(thresholds_path, sep='\t', index=False, float_format='%.4f')
        print(f"  Saved thresholds to {thresholds_path}")
    else:
        print(f"  No successful fits for {condition_name}, repetition {repetition}")
        plt.close()

def process_group_data(group_name, base_folder, conditions, num_repetitions):
    """
    Process all aggregated data files for a given group
    """
    for condition in conditions:
        for repetition in range(1, num_repetitions + 1):
            # Construct input path (looking for aggregated data with SEM)
            input_folder = os.path.join(
                base_folder, 
                "Analysis", 
                group_name, 
                condition, 
                f"Repetition_{repetition}"
            )
            # Look for the file with SEM data
            file_name = f"raw_data_with_sem_{group_name}_{condition}_Repetition_{repetition}.txt"
            file_path = os.path.join(input_folder, file_name)
            
            # Fallback to regular file if SEM file doesn't exist
            if not os.path.exists(file_path):
                file_name = f"raw_data_{group_name}_{condition}_Repetition_{repetition}.txt"
                file_path = os.path.join(input_folder, file_name)
                print(f"Note: SEM file not found, using mean-only file: {file_name}")

            # Construct output path
            output_folder = os.path.join(
                base_folder, 
                "Analysis", 
                group_name, 
                condition, 
                f"Repetition_{repetition}"
            )
            
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)

            if os.path.exists(file_path):
                print(f"\nProcessing {group_name}, {condition}, repetition {repetition}")
                raw_df = load_aggregated_data_file(file_path)
                if raw_df is not None:
                    perform_psychometric_analysis(raw_df, output_folder, group_name, condition, repetition)
            else:
                print(f"Aggregated data file not found: {file_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze aggregated data for a group across all conditions.')
    parser.add_argument('group', nargs='?', help='Name of the group to analyze (e.g., group1)', default=None)
    
    args = parser.parse_args()
    
    # If group not provided as argument, prompt for it
    if args.group is None:
        args.group = input("Please enter the group name: ").strip()
        while not args.group:
            print("Error: Group name cannot be empty!")
            args.group = input("Please enter the group name: ").strip()
    
    print(f"\nAnalyzing aggregated data for group: {args.group}")
    print("This script will generate psychometric function plots with error bars (mean ± SEM)")
    print("Note: Mean values are rounded to integers for psychometric function fitting\n")
    
    # Process all data for the group
    process_group_data(
        args.group, 
        BASE_FOLDER, 
        CONDITIONS, 
        NUM_REPETITIONS
    )

if __name__ == "__main__":
    main()