#!/usr/bin/env python3
"""
analyze_data_for_subject.py - Psychometric Function Analysis for Lightness Perception Data

Purpose:
Analyzes pre-processed psychophysical data to:
1. Calculate perceptual thresholds at 24%, 50%, and 76% performance levels
2. Generate publication-quality plots of psychometric functions
3. Save threshold data for statistical analysis

Key Functionality:
- Loads standardized raw data files created by process_subject_data.py
- Fits psychometric functions using psignifit toolbox
- Calculates thresholds at multiple performance levels
- Generates visualizations showing raw data and fitted curves
- Saves results in structured output files

Input Requirements:
- Input files must be in: BASE_FOLDER/Analysis/[subject]/[condition]/Repetition_[X]/
- Expects files named: raw_data_[subject]_[condition]_Repetition_[X].txt
- Requires tab-delimited files with standardized columns:
  * Standard lightness/background values
  * Comparison background values
  * Lightness values
  * Number of comparison choices
  * Trial counts per condition

Output Generated:
- PNG plots of psychometric functions:
  * Shows raw data points and fitted curves
  * Marks threshold points with annotations
  * Includes standard stimulus information
- Tab-delimited threshold files containing:
  * 24%, 50%, and 76% threshold values
  * Curve width parameters
  * Standard and comparison background values

Dependencies:
- Python 3.x
- Required packages: numpy, pandas, matplotlib, psignifit
- Pre-processing by process_subject_data.py

Usage:
$ python analyze_data_for_subject.py [subject_name]
or
$ python analyze_data_for_subject.py  # (prompts for subject name)

Example:
$ python analyze_data_for_subject.py subject01

Author: Vijay Singh
Created: April 26 2025
Version: 1.1 (Added RGB to luminance conversion)
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
CONDITIONS = ["condition_0", "condition_1", "condition_2", "condition_3"]

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
    'font.size': 12,          # Default font size
    'axes.titlesize': 14,     # Title font size
    'axes.labelsize': 14,     # Axis label font size
    'xtick.labelsize': 14,    # X-axis tick label size
    'ytick.labelsize': 14,    # Y-axis tick label size
    'legend.fontsize': 9      # Legend font size (reduced from 12 to 9)
})

def load_raw_data_file(file_path):
    """Load and process the raw data file created by the previous script"""
    try:
        df = pd.read_csv(file_path, sep='\t')
        return df
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def extract_data_for_background(raw_df, bg_val):
    """Extract relevant data for a specific background value"""
    bg_key = f"{bg_val:.4f}"
    
    data = {
        'lightness_vals': raw_df[f'lightness_vals_bkg_{bg_key}'].values,
        'n_comp_chsn': raw_df[f'n_cmp_chsn_bkg_{bg_key}'].values,
        'n_trials': raw_df[f'n_trials_bkg_{bg_key}'].values[0],  # All values should be same
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
def perform_psychometric_analysis(raw_df, output_folder, subject_name, condition_name, repetition):
    """
    Perform the psychometric analysis using the pre-processed raw data
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
    plt.figure(figsize=(5,3.4))  # Increased figure size
    custom_bounds = {'lambda':(0.0, 0.1)}
    ii=0 
    for bg_val_rgb in cmp_bkg_vals:
        # Extract data for this background
        bg_data = extract_data_for_background(raw_df, bg_val_rgb)
        
        # Convert lightness values from RGB to luminance
        lightness_vals_rgb = bg_data['lightness_vals']
        lightness_vals_lum = rgb_to_luminance(lightness_vals_rgb)
        
        n_comp_chsn = bg_data['n_comp_chsn']
        n_trials_per_comp_level = bg_data['n_trials']
        cmp_bkg_val_rgb = bg_data['cmp_bkg_val']
        
        # Convert comparison background to luminance (for legend)
        cmp_bkg_val_lum = rgb_to_luminance(cmp_bkg_val_rgb)
        
        # Generate fine-grained values for smooth curve (in luminance space)
        lightness_vals_fine_lum = np.linspace(lightness_vals_lum.min(), lightness_vals_lum.max(), 500)
        
        # Fit psychometric function (using luminance values)
        data_psignifit = np.array([lightness_vals_lum, n_comp_chsn, 
                                 np.ones(len(lightness_vals_lum))*n_trials_per_comp_level]).T
        result = ps.psignifit(data_psignifit, sigmoid='norm', bounds=custom_bounds)
        
        # Get thresholds at different levels (these are already in luminance units)
        threshold_50_lum = result.threshold(0.50)[0]
        threshold_76_lum = result.threshold(0.76)[0]
        # threshold_24_lum = result.threshold(0.24)[0]
        width_lum = result.parameter_estimate['width']
        
        # Store thresholds data (keep in luminance units)
        thresholds_data.append({
            'standard_rgb': std_lightness_rgb,
            'standard_luminance': std_lightness_lum,
            'background_rgb': bg_val_rgb,
            'background_luminance': cmp_bkg_val_lum,
            # 'threshold_24_luminance': threshold_24_lum,
            'threshold_50_luminance': threshold_50_lum,
            'threshold_76_luminance': threshold_76_lum,
            'width_luminance': width_lum
        })
        
        # Generate fitted curve (in luminance space)
        y_fit = n_trials_per_comp_level * result.proportion_correct(lightness_vals_fine_lum)
        style = BACKGROUND_STYLES.get(ii, {'color': "#d62728", 'marker': "x"})  # default to red x if not found
        color = style['color']
        ii +=1

        # Plot with larger markers (size=12) - now using luminance values
        # Simplified legend text - just the numeric value without units
        plt.plot(lightness_vals_lum, n_comp_chsn, '.', 
                markersize=12, color=color, label=f'{cmp_bkg_val_lum:.1f}')
        plt.plot(lightness_vals_fine_lum, y_fit, '-', linewidth=2, color=color)
        
        # Mark threshold points on the curve
        #plt.plot(threshold_24_lum, n_trials_per_comp_level*0.24, 'o', 
                #markersize=8, markeredgecolor='black', markerfacecolor=color)
        plt.plot(threshold_50_lum, n_trials_per_comp_level*0.5, 's', 
                markersize=8, markeredgecolor='black', markerfacecolor=color)
        #plt.plot(threshold_76_lum, n_trials_per_comp_level*0.76, '^', 
                #markersize=8, markeredgecolor='black', markerfacecolor=color)
        
        # Add annotations for all thresholds (commented out as in original)
        #plt.annotate(f'24%: {threshold_24_lum:.1f}', 
                    #(threshold_24_lum, n_trials_per_comp_level*0.24), 
                    #textcoords="offset points", xytext=(10,5), 
                    #ha='left', color='black', fontsize=11)
        #plt.annotate(f'50%: {threshold_50_lum:.1f}', 
                    #(threshold_50_lum, n_trials_per_comp_level*0.5), 
                    #textcoords="offset points", xytext=(10,5), 
                    #ha='left', color='black', fontsize=11)
        #plt.annotate(f'76%: {threshold_76_lum:.1f}', 
                    #(threshold_76_lum, n_trials_per_comp_level*0.76), 
                    #textcoords="offset points", xytext=(10,5), 
                    #ha='left', color='black', fontsize=11)

    # Add legend with smaller font (fontsize already set in rcParams to 9)
    legend = plt.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=9)
    # Optional: Add a title to the legend to indicate what the numbers represent
    legend.set_title("$B_c$ (cd/m²)", prop={'size': 9})
    
    plt.xlabel('Comparison Luminance (cd/m²)', fontsize=14)
    #plt.ylabel(f'Number Comparison Chosen (N = {n_trials_per_comp_level})', fontsize=14)
    plt.ylabel(f'Number Comparison Chosen\n(N = {n_trials_per_comp_level})', fontsize=14)
    #plt.title(f'Standard Lightness = {std_lightness_rgb}, Standard Background = {std_bkg_val_rgb}', fontsize=14)
    plt.title(f'$L_s$ = {std_lightness_lum:.1f} cd/m², $B_s$ = {std_bkg_val_lum:.1f} cd/m²', fontsize=14)
    plt.tight_layout()
    
    # Save the figure with subject name in filename
    plot_filename = f"{subject_name}_{condition_name}_Repetition_{repetition}.png"
    plot_path = os.path.join(output_folder, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved plot to {plot_path}")
    
    # Save thresholds data to text file
    thresholds_df = pd.DataFrame(thresholds_data)
    thresholds_filename = f"thresholds_{subject_name}_{condition_name}_Repetition_{repetition}.txt"
    thresholds_path = os.path.join(output_folder, thresholds_filename)
    thresholds_df.to_csv(thresholds_path, sep='\t', index=False, float_format='%.4f')
    print(f"Saved thresholds to {thresholds_path}")

def process_subject_data(subject_name, base_folder, conditions, num_repetitions):
    """
    Process all raw data files for a given subject
    """
    for condition in conditions:
        for repetition in range(1, num_repetitions + 1):
            # Construct input path (now looking in Analysis folder for raw data files)
            input_folder = os.path.join(
                base_folder, 
                "Analysis",  # Changed from "Data" to "Analysis"
                subject_name, 
                condition, 
                f"Repetition_{repetition}"
            )
            file_name = f"raw_data_{subject_name}_{condition}_Repetition_{repetition}.txt"
            file_path = os.path.join(input_folder, file_name)

            # Construct output path (same as before)
            output_folder = os.path.join(
                base_folder, 
                "Analysis", 
                subject_name, 
                condition, 
                f"Repetition_{repetition}"
            )
            
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)

            if os.path.exists(file_path):
                print(f"\nProcessing {subject_name}, {condition}, repetition {repetition}")
                raw_df = load_raw_data_file(file_path)
                if raw_df is not None:
                    perform_psychometric_analysis(raw_df, output_folder, subject_name, condition, repetition)
            else:
                print(f"Raw data file not found: {file_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze data for a subject across all conditions.')
    parser.add_argument('subject', nargs='?', help='Name of the subject to analyze', default=None)
    
    args = parser.parse_args()
    
    # If subject not provided as argument, prompt for it
    if args.subject is None:
        args.subject = input("Please enter the subject name (e.g., subject_apple): ").strip()
        while not args.subject:  # Keep asking until we get a non-empty input
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