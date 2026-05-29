#!/usr/bin/env python3
"""
group_thresholds_across_subjects.py - Group Threshold Analysis with Linear Fits

Purpose:
Groups threshold data from multiple subjects, calculates mean and standard deviation for each condition,
then performs linear regression analysis on the grouped data.

Key Functionality:
- Asks for subject names to group (comma-separated)
- Loads threshold files for all specified subjects
- Groups data by condition (background difference) and standard luminance
- Calculates mean and SD for thresholds across subjects
- Performs linear regression on grouped data for each background condition
- Generates plots with error bars (SD) and regression fits
- Saves grouped data and fit statistics

Input Requirements:
- Threshold files must be in: BASE_FOLDER/Analysis/[subject]/[condition]/Repetition_[X]/
- Expects files named: thresholds_[subject]_[condition]_Repetition_[X].txt
- Files can be in either RGB or luminance format

Output Generated:
- Plot showing threshold vs standard luminance with error bars (SD) and regression fits
- Tab-delimited grouped data file with means and SDs
- Fit results file with slope, intercept, slope standard error, intercept standard error, 
  r-value, and p-value for each condition

Usage:
$ python group_thresholds_across_subjects.py

Example:
$ python group_thresholds_across_subjects.py
Enter subject names (comma separated): subject1, subject2, subject3
Enter threshold type (threshold_24, threshold_50, threshold_76): threshold_50

Author: Vijay Singh
Created: April 26 2025
Version: 1.4 (Added intercept standard error)
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, t

# Constants
BASE_FOLDER = os.getcwd()
CONDITIONS = ["condition_1", "condition_2", "condition_3"]
NUM_REPETITIONS = 1

# RGB to Luminance conversion
MAX_RGB = 255
MAX_LUMINANCE = 105.6  # cd/m^2

def rgb_to_luminance(rgb_value):
    """Convert RGB value (0-255) to luminance (cd/m^2)"""
    if isinstance(rgb_value, (int, float)):
        return (rgb_value / MAX_RGB) * MAX_LUMINANCE
    else:
        return (np.array(rgb_value) / MAX_RGB) * MAX_LUMINANCE

# Style configuration - Consistent with previous threshold code
# These values represent differences in luminance (cd/m^2)
BACKGROUND_STYLES = {
    -4.14: {'color': "#1f77b4", 'marker': "o"},  # blue circle
    0.00: {'color': "#ff7f0e", 'marker': "s"},   # orange square
    4.14: {'color': "#2ca02c", 'marker': "^"}    # green triangle
}

def get_user_input():
    """Get subject names, group name, and threshold type from user"""
    print("\nGroup Threshold Analysis Tool (with Standard Deviation)")
    print("======================================================\n")
    
    # Get subject names
    while True:
        subject_input = input("Enter subject names to group (comma separated): ").strip()
        subject_names = [name.strip() for name in subject_input.split(',') if name.strip()]
        
        if len(subject_names) < 2:
            print("Please enter at least 2 valid subject names.")
            continue
        
        # Verify subjects exist
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
            print(f"Need at least 2 subjects with valid data folders. Found {len(valid_subjects)}.")
    
    # Get group name
    group_name = input("Enter a name for this group (e.g., 'experimental_group'): ").strip()
    while not group_name:
        group_name = input("Group name cannot be empty. Please enter a name: ").strip()
    
    # Get threshold type
    threshold_type = input("Enter threshold type (threshold_24, threshold_50, or threshold_76): ").strip()
    while threshold_type not in ["threshold_24", "threshold_50", "threshold_76"]:
        print("Error: Invalid threshold type!")
        threshold_type = input("Enter threshold type (threshold_24, threshold_50, or threshold_76): ").strip()
    
    return subject_names, group_name, threshold_type

def load_threshold_data_for_subject(subject, condition, repetition, threshold_type):
    """Load threshold data for a single subject and condition"""
    file_pattern = f"thresholds_{subject}_{condition}_Repetition_{repetition}.txt"
    search_path = os.path.join(
        BASE_FOLDER, "Analysis", subject, condition, f"Repetition_{repetition}", file_pattern
    )
    
    files = glob.glob(search_path)
    if not files:
        return None
    
    try:
        df = pd.read_csv(files[0], sep='\t')
        
        # Check format and extract appropriate columns
        if 'standard_luminance' in df.columns:
            standard_values = df['standard_luminance'].values
            background_values = df['background_luminance'].values
            
            if threshold_type == 'threshold_50':
                threshold_col = 'threshold_50_luminance'
            elif threshold_type == 'threshold_76':
                threshold_col = 'threshold_76_luminance'
            else:
                threshold_col = 'threshold_24_luminance' if 'threshold_24_luminance' in df.columns else threshold_type
            
            if threshold_col in df.columns:
                threshold_values = df[threshold_col].values
            else:
                threshold_values = df[threshold_type].values
        else:
            # Old format - convert RGB to luminance
            standard_values = rgb_to_luminance(df['standard'].values)
            background_values = rgb_to_luminance(df['background'].values)
            
            if threshold_type in df.columns:
                threshold_values = rgb_to_luminance(df[threshold_type].values)
            else:
                return None
        
        # Calculate background difference
        background_diff = (background_values - standard_values).round(2)
        
        # Create a DataFrame for this subject
        subject_df = pd.DataFrame({
            'standard_luminance': standard_values,
            'background_luminance': background_values,
            'background_diff': background_diff,
            'threshold': threshold_values,
            'subject': subject,
            'condition': condition
        })
        
        return subject_df
    
    except Exception as e:
        print(f"Error loading {files[0]}: {str(e)}")
        return None

def load_all_subject_data(subject_names, conditions, num_repetitions, threshold_type):
    """Load threshold data from all specified subjects"""
    all_data = []
    
    for subject in subject_names:
        print(f"  Loading data for {subject}...")
        for condition in conditions:
            for repetition in range(1, num_repetitions + 1):
                df = load_threshold_data_for_subject(subject, condition, repetition, threshold_type)
                if df is not None:
                    all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)

def group_and_analyze_data(data, threshold_type, group_name):
    """Group data across subjects, calculate mean/SD, and perform linear regression"""
    
    # Group by background_diff and standard_luminance
    # Calculate mean, standard deviation (SD), SEM, count
    grouped = data.groupby(['background_diff', 'standard_luminance'])['threshold'].agg(['mean', 'std', 'sem', 'count'])
    grouped = grouped.reset_index()
    
    print(f"\nGrouped data summary:")
    print(f"  Total unique conditions: {len(grouped)}")
    print(f"  Background differences: {sorted(grouped['background_diff'].unique())}")
    print(f"  Error bars show: Standard Deviation (SD) across subjects")
    
    # Prepare for plotting
    plt.figure(figsize=(6, 4))
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10
    })
    
    # Store fit results
    fit_results = {}
    
    # Plot each background difference
    for bg_diff in sorted(grouped['background_diff'].unique()):
        bg_data = grouped[grouped['background_diff'] == bg_diff]
        
        if len(bg_data) < 2:
            print(f"  Warning: Not enough points for Δ={bg_diff:.1f} (need at least 2)")
            continue
        
        # Get style for this background difference
        if bg_diff in BACKGROUND_STYLES:
            style = BACKGROUND_STYLES[bg_diff]
            # Create label with the difference value
            label = f"{bg_diff:.1f}"
        else:
            # Default style for any other background differences
            style = {'color': "#d62728", 'marker': "x"}
            label = f"{bg_diff:.1f}"
        
        # Plot mean with STANDARD DEVIATION error bars
        plt.errorbar(
            bg_data['standard_luminance'],
            bg_data['mean'],
            yerr=bg_data['std'],  # Using standard deviation
            fmt=style['marker'],
            color=style['color'],
            markersize=10,
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
            label=label,
            alpha=0.8
        )
        
        # Perform linear regression on grouped means
        x = bg_data['standard_luminance'].values
        y = bg_data['mean'].values
        
        # Use inverse of variance (SD^2) for weights
        # This gives more weight to points with smaller variance
        weights = 1 / (bg_data['std'].values**2 + 1e-10)  # Add small constant to avoid division by zero
        
        # Perform weighted linear regression
        try:
            # Use numpy's polyfit with weights
            slope, intercept = np.polyfit(x, y, 1, w=weights)
            
            # Calculate standard errors for slope and intercept
            n = len(x)
            x_weighted_avg = np.average(x, weights=weights)
            x_diff = x - x_weighted_avg
            weighted_ss_xx = np.sum(weights * x_diff**2)
            
            # Calculate predictions and residuals
            y_pred = slope * x + intercept
            residuals = y - y_pred
            
            # Weighted mean squared error
            mse_weighted = np.sum(weights * residuals**2) / (n - 2)
            
            # Standard error of slope
            standard_error_slope = np.sqrt(mse_weighted / weighted_ss_xx)
            
            # Standard error of intercept
            standard_error_intercept = np.sqrt(mse_weighted * (1 / np.sum(weights) + x_weighted_avg**2 / weighted_ss_xx))
            
            # Calculate r-squared and p-value for weighted fit
            ss_res = np.sum(weights * residuals**2)
            ss_tot = np.sum(weights * (y - np.average(y, weights=weights))**2)
            r_squared = 1 - (ss_res / ss_tot)
            r_value = np.sqrt(r_squared) * np.sign(slope)
            
            # Calculate t-statistic and p-value for slope
            t_stat_slope = slope / standard_error_slope
            p_value_slope = 2 * (1 - t.cdf(abs(t_stat_slope), df=n-2))
            
            # Calculate t-statistic and p-value for intercept
            t_stat_intercept = intercept / standard_error_intercept
            p_value_intercept = 2 * (1 - t.cdf(abs(t_stat_intercept), df=n-2))
            
        except:
            # Fall back to unweighted regression
            slope, intercept, r_value, p_value_slope, standard_error_slope = linregress(x, y)
            # Calculate standard error of intercept for unweighted case
            n = len(x)
            x_mean = np.mean(x)
            ss_xx = np.sum((x - x_mean)**2)
            mse = np.sum((y - (slope*x + intercept))**2) / (n - 2)
            standard_error_intercept = np.sqrt(mse * (1/n + x_mean**2/ss_xx))
            p_value_intercept = 2 * (1 - t.cdf(abs(intercept/standard_error_intercept), df=n-2))
        
        # Generate fit line
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = slope * x_fit + intercept
        
        plt.plot(x_fit, y_fit, '-', color=style['color'], linewidth=2, alpha=0.7)
        
        # Store fit results including slope and intercept errors
        fit_results[bg_diff] = {
            'slope': slope,
            'slope_std_err': standard_error_slope,
            'intercept': intercept,
            'intercept_std_err': standard_error_intercept,
            'r_value': r_value,
            'p_value_slope': p_value_slope,
            'p_value_intercept': p_value_intercept,
            'n_points': len(bg_data)
        }
        
        print(f"\nΔ={bg_diff:.1f} cd/m²:")
        print(f"  Slope: {slope:.3f} ± {standard_error_slope:.3f}, Intercept: {intercept:.3f} ± {standard_error_intercept:.3f}")
        print(f"  r = {r_value:.3f}, p(slope) = {p_value_slope:.4e}, p(intercept) = {p_value_intercept:.4e}")
        print(f"  Based on {len(bg_data)} points from {bg_data['count'].iloc[0]} subjects")
    
    # Format plot
    plt.xlabel('Standard Luminance (cd/m²)', fontsize=14)
    plt.ylabel(f'PSE (Mean ±1 SD) (cd/m²)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Create legend with slope information
    # Prepare legend labels with slope ± error
    legend_labels = []
    for bg_diff in sorted(grouped['background_diff'].unique()):
        if bg_diff in fit_results:
            slope = fit_results[bg_diff]['slope']
            slope_err = fit_results[bg_diff]['slope_std_err']
            label = f'{bg_diff:.1f}  (m={slope:.2f}±{slope_err:.2f})'
            legend_labels.append(label)
        else:
            legend_labels.append(f'{bg_diff:.1f}')
    
    # Get the handles and current labels
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
        fontsize=11,  # Slightly smaller to accommodate slope info
        framealpha=0.8,
        fancybox=True,
        borderpad=0.5,
        handlelength=1.5,
        title='$B_c-B_s$ (cd/m²)',
        title_fontsize=12
    )
    
    # Adjust legend title font
    legend.get_title().set_fontweight('bold')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    
    # Create output directory
    output_dir = os.path.join(BASE_FOLDER, "Analysis", group_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot
    plot_filename = f"all_conditions_with_fits_{threshold_type}_grouped.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot to {plot_path}")
    
    # Save grouped data (including both SD and SEM for reference)
    data_filename = f"grouped_data_{threshold_type}.txt"
    data_path = os.path.join(output_dir, data_filename)
    grouped.to_csv(data_path, sep='\t', index=False, float_format='%.4f')
    print(f"Saved grouped data to {data_path}")
    
    # Save fit results with slope and intercept errors
    results_filename = f"fit_results_{threshold_type}_grouped.txt"
    results_path = os.path.join(output_dir, results_filename)
    with open(results_path, 'w') as f:
        f.write("Weighted Linear Regression Results (using SD for weights)\n")
        f.write("=======================================================\n\n")
        f.write(f"Group: {group_name}\n")
        f.write(f"Threshold type: {threshold_type}\n")
        f.write(f"Number of subjects: {len(data['subject'].unique())}\n\n")
        f.write("Results by background difference (B_c - B_s):\n\n")
        
        for bg_diff in sorted(fit_results.keys()):
            r = fit_results[bg_diff]
            f.write(f"Δ = {bg_diff:.1f} cd/m²:\n")
            f.write(f"  Slope:     {r['slope']:.4f} ± {r['slope_std_err']:.4f} cd/m² per cd/m² (p = {r['p_value_slope']:.4e})\n")
            f.write(f"  Intercept: {r['intercept']:.4f} ± {r['intercept_std_err']:.4f} cd/m² (p = {r['p_value_intercept']:.4e})\n")
            f.write(f"  r-value:   {r['r_value']:.4f}\n")
            f.write(f"  N points:  {r['n_points']}\n\n")
        
        f.write("\nNotes:\n")
        f.write("  - Error bars in plot show ±1 Standard Deviation (SD)\n")
        f.write("  - Linear regression uses inverse variance weighting (1/SD²)\n")
        f.write("  - SD reflects between-subject variability\n")
        f.write("  - Slope error and intercept error represent standard errors of the estimates\n")
        f.write("  - p-values test whether slope/intercept are significantly different from zero\n")
    
    print(f"Saved fit results to {results_path}")
    
    return grouped, fit_results

def main():
    # Get user input
    subject_names, group_name, threshold_type = get_user_input()
    
    print(f"\nLoading threshold data for subjects: {', '.join(subject_names)}")
    print(f"Threshold type: {threshold_type}\n")
    
    # Load all data
    all_data = load_all_subject_data(subject_names, CONDITIONS, NUM_REPETITIONS, threshold_type)
    
    if all_data.empty:
        print("No threshold data found!")
        return
    
    print(f"\nTotal data points loaded: {len(all_data)}")
    print(f"Subjects: {sorted(all_data['subject'].unique())}")
    print(f"Background differences: {sorted(all_data['background_diff'].unique())}")
    
    # Group and analyze
    grouped_data, fit_results = group_and_analyze_data(all_data, threshold_type, group_name)
    
    print(f"\nAnalysis complete! Results saved in: {os.path.join(BASE_FOLDER, 'Analysis', group_name)}")

if __name__ == "__main__":
    main()