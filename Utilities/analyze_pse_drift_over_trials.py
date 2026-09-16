#!/usr/bin/env python3

"""
analyze_pse_drift_over_trials.py - Test whether the contextual PSE shift changes
over the course of a condition

Purpose:
Observers received trial-by-trial auditory feedback defined by the physical
luminance of the two patches. In principle such feedback could encourage
observers to move their responses towards physical luminance as a condition
progresses, which would reduce the measured contextual effect over time. This
script tests for that by splitting the trials of each condition at their median
presentation order and estimating the point of subjective equality (PSE)
separately from the first and the second half.

Method:
1. Read the raw trial files rather than the fitted thresholds.
2. On each trial, identify which side was the standard (patch and background
   both at the standard value) and which was the comparison. Trials in which
   both sides are physically identical carry no information about which patch
   was "the comparison" and are excluded.
3. Split the trials of each observer x block x comparison-background cell at the
   median presentation order.
4. Fit a cumulative Gaussian to each half by maximum likelihood and take the PSE
   as its 50% point.
5. Express the contextual effect as the difference between the PSE for a given
   comparison background and the PSE for the equal-background condition measured
   in the same half. Using the same-half baseline removes any general drift of
   the observer's criterion, leaving the contextual effect itself.
6. Compare the first and second half with a paired t-test across the
   observer x block cells.

Input Requirements:
- Raw trial files at:
  BASE_FOLDER/Data/[subject]/[condition]/Repetition_[X]/[subject]_[condition]_Repetition_[X].txt
  with one trial per line in the format
  [order],[expected],[response],[RGBA],[RGBA],[RGBA],[RGBA]

Output Generated:
- Analysis/[group_name]/pse_drift_first_vs_second_half.txt   summary and tests
- Analysis/[group_name]/pse_drift_cell_values.txt            per-cell PSEs

Dependencies:
- Python 3.x
- Libraries: numpy, pandas, scipy

Usage:
Run from the folder where the Data folder is stored.
$ python3 Utilities/analyze_pse_drift_over_trials.py
Provide the subject names and a group name when prompted.

Author: Vijay Singh
Created: September 2026
Version: 1.0
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, ttest_rel

# Constants
BASE_FOLDER = os.getcwd()
CONDITIONS = ["condition_1", "condition_2", "condition_3"]
NUM_REPETITIONS = 1

# RGB to Luminance conversion
MAX_RGB = 255
MAX_LUMINANCE = 105.6  # cd/m^2
TRANSPARENCY_ALPHA = 160/255  # Transparency alpha factor for background luminance

TRIAL_PATTERN = re.compile(
    r'(\d+),\s*(\d+),\s*(\d+),'
    r'\s*RGBA\((\d+),[^)]*\),\s*RGBA\((\d+),[^)]*\),'
    r'\s*RGBA\((\d+),[^)]*\),\s*RGBA\((\d+),[^)]*\)')


def rgb_to_luminance(rgb_value):
    """Convert RGB value (0-255) to luminance (cd/m^2)"""
    return (np.asarray(rgb_value, dtype=float) / MAX_RGB) * MAX_LUMINANCE


def get_user_input():
    """Get subject names and group name from the user"""
    print("\nPSE Drift Analysis Tool (first half vs second half of each condition)")
    print("=====================================================================\n")

    while True:
        subject_input = input("Enter subject names to analyze (comma separated): ").strip()
        subject_names = [name.strip() for name in subject_input.split(',') if name.strip()]

        valid_subjects = []
        for subject in subject_names:
            if os.path.exists(os.path.join(BASE_FOLDER, "Data", subject)):
                valid_subjects.append(subject)
            else:
                print(f"Warning: No data folder found for subject '{subject}'")

        if len(valid_subjects) >= 2:
            return valid_subjects, _get_group_name()
        print(f"Need at least 2 subjects with valid data folders. Found {len(valid_subjects)}.")


def _get_group_name():
    group_name = input("Enter a name for this group (e.g., 'mean_threshold'): ").strip()
    while not group_name:
        group_name = input("Group name cannot be empty. Please enter a name: ").strip()
    return group_name


def load_trials(subject, condition, repetition):
    """Read one raw trial file and return a tidy DataFrame of comparison trials

    Each row gives the comparison patch level, the comparison background, whether
    the observer chose the comparison, and the presentation order. The standard
    is the side whose patch and background are both at the standard value.
    """
    file_path = os.path.join(
        BASE_FOLDER, "Data", subject, condition, f"Repetition_{repetition}",
        f"{subject}_{condition}_Repetition_{repetition}.txt")

    if not os.path.exists(file_path):
        print(f"Warning: No trial file found at {file_path}")
        return pd.DataFrame()

    raw = []
    with open(file_path, 'r') as handle:
        for line in handle:
            match = TRIAL_PATTERN.match(line.strip())
            if match:
                raw.append([int(g) for g in match.groups()])

    if not raw:
        print(f"Warning: No parsable trials in {file_path}")
        return pd.DataFrame()

    columns = ['order', 'expected', 'response',
               'square_1', 'background_1', 'square_2', 'background_2']
    trials = pd.DataFrame(raw, columns=columns)

    # The standard patch and its background are the values presented on every
    # trial, so they are the most common values in the file
    standard_patch = trials[['square_1', 'square_2']].stack().mode()[0]
    standard_background = trials[['background_1', 'background_2']].stack().mode()[0]

    is_standard_1 = ((trials['square_1'] == standard_patch) &
                     (trials['background_1'] == standard_background))
    is_standard_2 = ((trials['square_2'] == standard_patch) &
                     (trials['background_2'] == standard_background))

    # Trials in which both sides are physically identical are uninformative
    keep = is_standard_1 ^ is_standard_2
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"    {subject} {condition}: dropped {n_dropped} physically identical trials")

    trials = trials[keep].copy()
    is_standard_1 = is_standard_1[keep]

    trials['comparison_patch'] = np.where(is_standard_1, trials['square_2'], trials['square_1'])
    trials['comparison_background'] = np.where(is_standard_1, trials['background_2'],
                                               trials['background_1'])
    trials['chose_comparison'] = np.where(is_standard_1, trials['response'] == 2,
                                          trials['response'] == 1).astype(int)
    trials['background_diff'] = trials['comparison_background'] - standard_background
    trials['standard_patch'] = standard_patch
    trials['subject'] = subject
    trials['condition'] = condition

    return trials[['subject', 'condition', 'order', 'standard_patch', 'background_diff',
                   'comparison_patch', 'chose_comparison']]


def fit_pse(levels, chose_comparison):
    """Maximum likelihood fit of a cumulative Gaussian; returns the 50% point

    Returns the PSE on the same scale as `levels`, or nan if the fit fails.
    """
    levels = np.asarray(levels, dtype=float)
    chose_comparison = np.asarray(chose_comparison, dtype=float)

    if len(np.unique(levels)) < 2 or len(np.unique(chose_comparison)) < 2:
        return np.nan

    def negative_log_likelihood(params):
        mu, log_sigma = params
        p = norm.cdf((levels - mu) / np.exp(log_sigma))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.sum(chose_comparison * np.log(p) +
                       (1 - chose_comparison) * np.log(1 - p))

    start = [levels.mean(), np.log(max(levels.std(), 1e-3))]
    result = minimize(negative_log_likelihood, start, method='Nelder-Mead',
                      options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 5000})
    return result.x[0] if result.success else np.nan


def compute_half_pses(trials):
    """Fit a PSE for each subject x condition x background cell and trial half"""
    records = []
    group_columns = ['subject', 'condition', 'background_diff']

    for keys, cell in trials.groupby(group_columns):
        median_order = cell['order'].median()
        for half_label, half in (('first', cell[cell['order'] <= median_order]),
                                 ('second', cell[cell['order'] > median_order])):
            pse_rgb = fit_pse(half['comparison_patch'], half['chose_comparison'])
            record = dict(zip(group_columns, keys))
            record.update({'half': half_label,
                           'n_trials': len(half),
                           'pse_luminance': rgb_to_luminance(pse_rgb)})
            records.append(record)

    return pd.DataFrame(records)


def compute_contextual_offsets(half_pses):
    """Express each PSE relative to the equal-background PSE of the same half"""
    wide = half_pses.pivot_table(index=['subject', 'condition', 'half'],
                                 columns='background_diff', values='pse_luminance')
    equal_background = 0
    if equal_background not in wide.columns:
        raise ValueError("No equal-background condition found in the data")

    offsets = wide.drop(columns=[equal_background]).sub(wide[equal_background], axis=0)
    offsets['equal_background_pse'] = wide[equal_background]
    return offsets.reset_index()


def main():
    subject_names, group_name = get_user_input()

    print(f"\nLoading trial data for subjects: {', '.join(subject_names)}\n")
    all_trials = []
    for subject in subject_names:
        for condition in CONDITIONS:
            for repetition in range(1, NUM_REPETITIONS + 1):
                trials = load_trials(subject, condition, repetition)
                if not trials.empty:
                    all_trials.append(trials)

    if not all_trials:
        print("No trial data found!")
        return

    trials = pd.concat(all_trials, ignore_index=True)
    print(f"\nTotal comparison trials analyzed: {len(trials)}")

    half_pses = compute_half_pses(trials)
    offsets = compute_contextual_offsets(half_pses)

    first = offsets[offsets['half'] == 'first'].set_index(['subject', 'condition'])
    second = offsets[offsets['half'] == 'second'].set_index(['subject', 'condition'])

    background_columns = [c for c in offsets.columns
                          if c not in ('subject', 'condition', 'half', 'equal_background_pse')]

    output_dir = os.path.join(BASE_FOLDER, "Analysis", group_name)
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "pse_drift_first_vs_second_half.txt")

    lines = []
    lines.append("Contextual PSE shift in the first and second half of each condition")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Subjects: {', '.join(subject_names)}")
    lines.append(f"Conditions: {', '.join(CONDITIONS)}")
    lines.append(f"Comparison trials analyzed: {len(trials)}")
    lines.append("")
    lines.append("The contextual shift is the PSE for a given comparison background minus")
    lines.append("the PSE for the equal-background condition measured in the same half, so")
    lines.append("any general drift of the observer's criterion is removed. Values are in")
    lines.append("cd/m². A reduction of the shift in the second half would be expected if")
    lines.append("luminance-based feedback moved responses towards physical luminance.")
    lines.append("")

    for column in sorted(background_columns):
        a = first[column]
        b = second[column].reindex(a.index)
        valid = a.notna() & b.notna()
        test = ttest_rel(a[valid], b[valid])
        # Background luminance carries the transparency factor, as in Table 1
        background_offset = rgb_to_luminance(column) * TRANSPARENCY_ALPHA
        lines.append(f"Comparison background {background_offset:+.1f} cd/m² "
                     f"relative to the standard background:")
        lines.append(f"  First half:   {a[valid].mean():+.3f} cd/m²")
        lines.append(f"  Second half:  {b[valid].mean():+.3f} cd/m²")
        lines.append(f"  Change:       {b[valid].mean() - a[valid].mean():+.3f} cd/m²")
        lines.append(f"  Paired t({int(valid.sum()) - 1}) = {abs(test.statistic):.2f}, "
                     f"p = {test.pvalue:.3f}, n = {int(valid.sum())} observer by block cells")
        lines.append("")

    a = first['equal_background_pse']
    b = second['equal_background_pse'].reindex(a.index)
    valid = a.notna() & b.notna()
    test = ttest_rel(a[valid], b[valid])
    lines.append("Equal-background PSE (check for a general drift of the criterion):")
    lines.append(f"  First half:   {a[valid].mean():.3f} cd/m²")
    lines.append(f"  Second half:  {b[valid].mean():.3f} cd/m²")
    lines.append(f"  Paired t({int(valid.sum()) - 1}) = {abs(test.statistic):.2f}, "
                 f"p = {test.pvalue:.3f}")
    lines.append("")
    lines.append("Note: splitting each condition halves the number of trials per comparison")
    lines.append("level, so the per-half PSEs are noisier than those reported in Table 2.")

    with open(summary_path, 'w') as handle:
        handle.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"Saved summary to {summary_path}")

    cell_path = os.path.join(output_dir, "pse_drift_cell_values.txt")
    half_pses.to_csv(cell_path, sep='\t', index=False, float_format='%.4f')
    print(f"Saved per-cell PSE values to {cell_path}")


if __name__ == "__main__":
    main()
