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
  * Linear regression lines with R values
  * Condition-specific markers and colors
- Text file with regression statistics:
  * Slope, intercept, and correlation coefficient for each condition

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
- Detailed legend with regression statistics

Author: Vijay Singh
Created: April 26 2025
Version: 1.0
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Constants
BASE_FOLDER = "/Users/vsingh1/Documents/BackgroundEffect"
CONDITIONS = ["condition_1", "condition_2", "condition_3"]
NUM_REPETITIONS = 1  # Added constant for number of repetitions

# Mapping of actual subject names to display names
SUBJECT_DISPLAY_NAMES = {
    "vijay_audio": "Subject 1",
    "heshu_audio": "Subject 2",
    "subject_veronica": "Subject 3",
    "vijay_heshu_veronica": "All Subjects"
    # Add more mappings as needed
}

# Style configuration
BACKGROUND_STYLES = {
    -0.05: {'color': "#1f77b4", 'marker': "o"},  # blue circle
    0.00: {'color': "#ff7f0e", 'marker': "s"},   # orange square
    0.05: {'color': "#2ca02c", 'marker': "^"}    # green triangle
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

def load_threshold_data(base_folder, subject, conditions, num_repetitions=1):
    """Load threshold data from all conditions and repetitions"""
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
                df['condition'] = condition
                df['repetition'] = repetition
                df['background_diff'] = (df['background'] - df['standard']).round(2)
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
        style = BACKGROUND_STYLES.get(bg_diff, {'color': "#d62728", 'marker': "x"})  # default to red x if not found
        
        color = plt.cm.viridis(ii/3)
        ii += 1

        # Plot data points (averaging across repetitions if needed)
        if 'repetition' in bg_data.columns and len(bg_data['repetition'].unique()) > 1:
            # Average across repetitions for each standard value
            avg_data = bg_data.groupby('standard')[threshold_type].mean().reset_index()
            std_data = bg_data.groupby('standard')[threshold_type].std().reset_index()
            
            plt.errorbar(
                avg_data['standard'],
                avg_data[threshold_type],
                yerr=std_data[threshold_type],
                fmt=style['marker'],
                color=color,
                markersize=8,
                capsize=4,
                capthick=1,
                elinewidth=1,
                label=f'Δ={bg_diff:.2f} (mean ± SD)'
            )
        else:
            plt.scatter(
                bg_data['standard'],
                bg_data[threshold_type],
                c=color,
                marker=style['marker'],
                s=100,
                edgecolors='black',
                linewidths=0.5,
                label=f'$B_c - B_s = ${bg_diff:.2f}'
            )
        
        # Calculate and plot linear fit
        x = bg_data['standard']
        y = bg_data[threshold_type]
        slope, intercept, r_value, _, _ = linregress(x, y)
        print(slope,intercept)
        
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = slope * x_fit + intercept
        
        plt.plot(x_fit, y_fit, '-', color=color, linewidth=2,)
                #label=f'Δ={bg_diff:.2f} fit (r={r_value:.2f})')
                #label=f'Δ={slope:.2f}')
        
        # Store fit results
        fit_results[bg_diff] = {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value
        }
    
    plt.xlabel('Standard Lightness', fontsize=14)
    #plt.ylabel(f'{threshold_type.replace("_", " ").title()}', fontsize=14)
    plt.ylabel('PSE', fontsize=14)
    #plt.title(f'{display_name}: Standard vs {threshold_type.replace("_", " ")}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Create legend
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.legend(
        loc='upper left',
        fontsize=12,      # Smaller font size
        framealpha=0.8,  # Slightly more opaque
        fancybox=True,   # Rounded corners
        borderpad=0.5,   # Less padding
        handlelength=1.5 # Shorter legend lines
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(BASE_FOLDER, "Analysis", subject)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_file = os.path.join(output_dir, f"all_conditions_with_fits_{threshold_type}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_file}")
    
    # Save fit results to file
    fit_results_file = os.path.join(output_dir, f"fit_results_{threshold_type}.txt")
    with open(fit_results_file, 'w') as f:
        f.write("Linear Fit Results:\n")
        for diff, result in sorted(fit_results.items()):
            f.write(f"Δ={diff:.2f}: slope={result['slope']:.3f}, intercept={result['intercept']:.3f}, r={result['r_value']:.3f}\n")
    
    print(f"Saved fit results to {fit_results_file}")
    plt.close()

def main():
    # Get user input
    subject, threshold_type = get_user_input()
    
    # Load all threshold data
    threshold_data = load_threshold_data(BASE_FOLDER, subject, CONDITIONS, NUM_REPETITIONS)
    
    if threshold_data.empty:
        print("No threshold data found!")
        return
    
    # Plot all conditions together
    plot_all_conditions(threshold_data, threshold_type, subject)

if __name__ == "__main__":
    main()
