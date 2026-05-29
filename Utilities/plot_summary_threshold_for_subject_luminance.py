#!/usr/bin/env python3
"""
plot_summary_threshold_for_subject.py - Threshold vs. Lightness Analysis Plotter

Purpose:
Visualizes the relationship between standard lightness values and perceptual thresholds
across different background conditions, including linear regression analysis.

Key Functionality:
- Loads pre-computed threshold data from multiple experimental conditions
- Generates composite plots showing threshold vs. standard lightness relationships
- Calculates and displays linear regression fits for each background condition
- Includes error bars when multiple repetitions are available
- Saves publication-quality figures and regression statistics

Input Requirements:
- Threshold files must be in: BASE_FOLDER/Analysis/[subject]/[condition]/Repetition_[X]/
- Expects files named: thresholds_[subject]_[condition]_Repetition_[X].txt
- Requires tab-delimited files with columns:
  * standard (lightness value)
  * background (value)
  * threshold_24, threshold_50, threshold_76
  * width (psychometric function width)

Output Generated:
- PNG plot showing:
  * Threshold vs. standard lightness for all background conditions
  * Linear regression lines with R values and slope errors
  * Condition-specific markers and colors
- Text file with regression statistics:
  * Slope, intercept, correlation coefficient, p-value, and standard error for each condition

Dependencies:
- Python 3.x
- Required packages: pandas, matplotlib, numpy, scipy
- Pre-computed threshold data from analyze_data_for_subject.py

Usage:
$ python plot_summary_threshold_for_subject.py
(interactively prompts for subject name and threshold type)

Example Output Files:
- all_conditions_with_fits_threshold_50.png
- fit_results_threshold_50.txt

Visualization Features:
- Consistent color/marker scheme for background conditions
- Error bars showing variability across repetitions
- Clean, publication-ready formatting
- Detailed legend with regression statistics including slope error

Author: Vijay Singh
Created: April 26 2025
Version: 1.3 (Added error bars on slope in legend and output)
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Constants
BASE_FOLDER = os.getcwd()
CONDITIONS = ["condition_1", "condition_2", "condition_3"]
NUM_REPETITIONS = 1  # Added constant for number of repetitions

# RGB to Luminance conversion
MAX_RGB = 255
MAX_LUMINANCE = 105.6  # cd/m^2

def rgb_to_luminance(rgb_value):
    """
    Convert RGB value (0-255) to luminance (cd/m^2)
    RGB of 255 corresponds to 105.6 cd/m^2
    """
    if isinstance(rgb_value, (int, float)):
        return (rgb_value / MAX_RGB) * MAX_LUMINANCE
    else:
        # Handle array-like inputs
        return (np.array(rgb_value) / MAX_RGB) * MAX_LUMINANCE

# Mapping of actual subject names to display names
SUBJECT_DISPLAY_NAMES = {
    "vijay_audio": "Subject 1",
    "heshu_audio": "Subject 2",
    "subject_veronica": "Subject 3",
    "vijay_heshu_veronica": "All Subjects"
    # Add more mappings as needed
}

# Style configuration - Note: background differences are now in luminance units
# These values should represent differences in luminance (cd/m^2)
BACKGROUND_STYLES = {
    -4.14: {'color': "#1f77b4", 'marker': "o"},  # blue circle
    0.00: {'color': "#ff7f0e", 'marker': "s"},   # orange square
    4.14: {'color': "#2ca02c", 'marker': "^"}    # green triangle
}

def get_user_input():
    """Get subject name and threshold type from user"""
    subject = input("Enter subject name (e.g., vijay_audio): ").strip()
    while not subject:
        print("Error: Subject name cannot be empty!")
        subject = input("Enter subject name: ").strip()
    
    threshold_type = input("Enter threshold type (threshold_24, threshold_50, or threshold_76): ").strip()
    while threshold_type not in ["threshold_24", "threshold_50", "threshold_76"]:
        print("Error: Invalid threshold type!")
        threshold_type = input("Enter threshold type: ").strip()
    
    return subject, threshold_type

def load_threshold_data(base_folder, subject, conditions, num_repetitions, threshold_type):
    """Load threshold data from all conditions and repetitions and convert to luminance"""
    all_data = []
    
    for condition in conditions:
        for repetition in range(1, num_repetitions + 1):
            file_pattern = f"thresholds_{subject}_{condition}_Repetition_{repetition}.txt"
            search_path = os.path.join(
                base_folder, 
                "Analysis", 
                subject, 
                condition, 
                f"Repetition_{repetition}", 
                file_pattern
            )
            
            files = glob.glob(search_path)
            if not files:
                print(f"Warning: No threshold file found for {search_path}")
                continue
                
            try:
                df = pd.read_csv(files[0], sep='\t')
                
                # Check if we're using the old format (RGB) or new format (luminance)
                # New format from updated analyze_data_for_subject.py
                if 'standard_luminance' in df.columns:
                    # Use luminance columns directly
                    df['standard_luminance'] = df['standard_luminance']
                    df['background_luminance'] = df['background_luminance']
                    
                    # Get threshold columns (they should already be in luminance)
                    if threshold_type == 'threshold_50':
                        threshold_col = 'threshold_50_luminance' if 'threshold_50_luminance' in df.columns else threshold_type
                    elif threshold_type == 'threshold_76':
                        threshold_col = 'threshold_76_luminance' if 'threshold_76_luminance' in df.columns else threshold_type
                    else:  # threshold_24
                        threshold_col = 'threshold_24_luminance' if 'threshold_24_luminance' in df.columns else threshold_type
                    
                    if threshold_col not in df.columns:
                        print(f"Warning: {threshold_col} not found in {files[0]}, trying {threshold_type}")
                        threshold_col = threshold_type
                    
                    df['threshold_value'] = df[threshold_col] if threshold_col in df.columns else None
                    df['standard_value'] = df['standard_luminance']
                    df['background_value'] = df['background_luminance']
                else:
                    # Old format - convert RGB to luminance
                    df['standard_rgb'] = df['standard']
                    df['background_rgb'] = df['background']
                    
                    # Convert standard and background from RGB to luminance
                    df['standard_value'] = rgb_to_luminance(df['standard'])
                    df['background_value'] = rgb_to_luminance(df['background'])
                    
                    # Convert thresholds from RGB to luminance (they were stored as RGB values)
                    if threshold_type in df.columns:
                        df['threshold_value'] = rgb_to_luminance(df[threshold_type])
                    else:
                        print(f"Warning: {threshold_type} not found in {files[0]}")
                        continue
                
                # Check if threshold_value exists and is not None
                if 'threshold_value' not in df.columns or df['threshold_value'].isnull().all():
                    print(f"Warning: No valid threshold data in {files[0]}")
                    continue
                
                df['condition'] = condition
                df['repetition'] = repetition
                df['background_diff'] = (df['background_value'] - df['standard_value']).round(2)
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {files[0]}: {str(e)}")
                continue
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)

def plot_all_conditions(data, threshold_type, subject):
    """Plot all conditions on the same figure with linear fits"""
    plt.figure(figsize=(5,3.4))
    custom_bounds = {'lambda':(0.0, 0.1)}
    plt.rc('xtick', labelsize=14)  # X-axis tick label size
    plt.rc('ytick', labelsize=14)  # Y-axis tick label size
    display_name = SUBJECT_DISPLAY_NAMES.get(subject, f"Subject {len(SUBJECT_DISPLAY_NAMES)+1}")
    
    # Prepare to store fit results
    fit_results = {}
    ii=0
    
    # Plot each background difference with fits
    for bg_diff in sorted(data['background_diff'].unique()):
        bg_data = data[data['background_diff'] == bg_diff]
        if len(bg_data) < 2:
            continue
            
        # Get style for this background difference
        print(bg_diff)
        style = BACKGROUND_STYLES.get(bg_diff, {'color': "#d62728", 'marker': "x"})  # default to red x if not found
        
        color = style['color']
        ii += 1

        # Plot data points (averaging across repetitions if needed)
        # Simplified legend labels - just the numeric value without units or text
        if 'repetition' in bg_data.columns and len(bg_data['repetition'].unique()) > 1:
            # Average across repetitions for each standard value
            avg_data = bg_data.groupby('standard_value')['threshold_value'].mean().reset_index()
            std_data = bg_data.groupby('standard_value')['threshold_value'].std().reset_index()
            
            plt.errorbar(
                avg_data['standard_value'],
                avg_data['threshold_value'],
                yerr=std_data['threshold_value'],
                fmt=style['marker'],
                color=color,
                markersize=8,
                capsize=4,
                capthick=1,
                elinewidth=1,
                label=f'{bg_diff:.1f}'  # Simplified: just the number
            )
        else:
            plt.scatter(
                bg_data['standard_value'],
                bg_data['threshold_value'],
                c=color,
                marker=style['marker'],
                s=100,
                edgecolors='black',
                linewidths=0.5,
                label=f'{bg_diff:.1f}'  # Simplified: just the number
            )
        
        # Calculate and plot linear fit
        x = bg_data['standard_value']
        y = bg_data['threshold_value']
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        if len(x_clean) < 2:
            print(f"Warning: Not enough valid data points for fit (Δ={bg_diff:.1f})")
            continue
            
        # Perform linear regression and get p-value
        slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
        print(f"Slope: {slope:.3f} ± {std_err:.3f}, Intercept: {intercept:.3f}, r: {r_value:.3f}, p: {p_value:.4f}")
        
        x_fit = np.linspace(min(x_clean), max(x_clean), 100)
        y_fit = slope * x_fit + intercept
        
        plt.plot(x_fit, y_fit, '-', color=color, linewidth=2)
        
        # Store fit results with p-value and standard error
        fit_results[bg_diff] = {
            'slope': slope,
            'slope_std_err': std_err,  # Added standard error for slope
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value,
            'std_err': std_err
        }
    
    plt.xlabel('Standard Luminance (cd/m²)', fontsize=14)
    plt.ylabel('PSE (cd/m²)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Create legend with title including slope error information
    # Format legend labels to include slope ± error
    legend_labels = []
    for bg_diff in sorted(data['background_diff'].unique()):
        if bg_diff in fit_results:
            slope = fit_results[bg_diff]['slope']
            slope_err = fit_results[bg_diff]['slope_std_err']
            # Format the label with slope and error
            label = f'{bg_diff:.1f}  (m={slope:.2f}±{slope_err:.2f})'
            legend_labels.append(label)
        else:
            legend_labels.append(f'{bg_diff:.1f}')
    
    # Get the handles for the legend (the plotted elements)
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Update labels with slope information
    updated_labels = []
    for i, original_label in enumerate(labels):
        if i < len(legend_labels):
            updated_labels.append(legend_labels[i])
        else:
            updated_labels.append(original_label)
    
    # Create legend with title
    legend = plt.legend(
        handles,
        updated_labels,
        loc='upper left',
        fontsize=11,  # Slightly smaller to accommodate extra info
        framealpha=0.8,
        fancybox=True,
        borderpad=0.5,
        handlelength=1.5,
        title='$B_c-B_s \\rm {(cd/m²)}$',
        title_fontsize=12
    )
    
    # Adjust legend title font
    legend.get_title().set_fontweight('bold')

    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(BASE_FOLDER, "Analysis", subject)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_file = os.path.join(output_dir, f"all_conditions_with_fits_{threshold_type}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_file}")
    
    # Save fit results to file with p-values and slope errors
    fit_results_file = os.path.join(output_dir, f"fit_results_{threshold_type}.txt")
    with open(fit_results_file, 'w') as f:
        f.write("Linear Fit Results (in luminance units: cd/m²):\n")
        f.write("Standard Luminance vs. PSE\n")
        f.write("=" * 70 + "\n\n")
        for diff, result in sorted(fit_results.items()):
            f.write(f"Δ = {diff:.1f} cd/m²:\n")
            f.write(f"  Slope:     {result['slope']:.4f} ± {result['slope_std_err']:.4f}\n")
            f.write(f"  Intercept: {result['intercept']:.4f}\n")
            f.write(f"  r-value:   {result['r_value']:.4f}\n")
            f.write(f"  p-value:   {result['p_value']:.4e}\n")
            f.write(f"  Std Err:   {result['std_err']:.4f}\n")
            f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("Note: All values are in cd/m² (luminance units)\n")
        f.write("p-value indicates significance of the slope (H0: slope = 0)\n")
        f.write("Slope error represents the standard error of the slope estimate\n")
    
    print(f"Saved fit results to {fit_results_file}")
    plt.close()

def main():
    # Get user input
    subject, threshold_type = get_user_input()
    
    print(f"\nLoading data for subject: {subject}")
    print(f"Threshold type: {threshold_type}\n")
    
    # Load all threshold data
    threshold_data = load_threshold_data(BASE_FOLDER, subject, CONDITIONS, NUM_REPETITIONS, threshold_type)
    
    if threshold_data.empty:
        print("No threshold data found!")
        return
    
    print(f"\nLoaded {len(threshold_data)} data points")
    print(f"Unique background differences: {sorted(threshold_data['background_diff'].unique())}")
    
    # Plot all conditions together
    plot_all_conditions(threshold_data, threshold_type, subject)

if __name__ == "__main__":
    main()