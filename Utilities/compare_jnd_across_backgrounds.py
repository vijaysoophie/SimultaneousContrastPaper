#!/usr/bin/env python3

"""
compare_jnd_across_backgrounds.py - Compare the just noticeable difference across
comparison background conditions

Purpose:
The model predicts that a multiplicative contextual gain changes the width of the
psychometric function, whereas a purely additive gain leaves it unchanged. The
JND therefore provides a check on the multiplicative gain parameter alpha that
does not depend on the point of subjective equality. This script compares the JND
across the three comparison background conditions.

Method:
1. Read the threshold files written by plot_summary_threshold_for_subject_luminance.py.
2. For every observer x standard block x comparison background cell, take the JND
   as the difference between the 76% and the 50% points of the fitted
   psychometric function.
3. Average the JND over the three standard blocks within each observer. The three
   blocks are repeated observations on the same observers, so the observer, and
   not the observer x block cell, is the independent unit of analysis.
4. Report the mean and standard error of the JND for each comparison background
   across observers, and compare the three background conditions with a Friedman
   test. The Friedman test is the nonparametric test for a repeated measures
   design and makes no assumption about the distribution of the JND.

Input Requirements:
- Threshold files at:
  BASE_FOLDER/Analysis/[subject]/[condition]/Repetition_[X]/thresholds_[subject]_[condition]_Repetition_[X].txt
  These are produced by analyze_data_for_subject_luminance.py and
  plot_summary_threshold_for_subject_luminance.py.

Output Generated:
- Analysis/[group_name]/jnd_across_backgrounds.txt        summary and test
- Analysis/[group_name]/jnd_observer_values.txt           per-observer JND values
- Analysis/[group_name]/jnd_cell_values.txt               per-cell JND values

Dependencies:
- Python 3.x
- Libraries: numpy, pandas, scipy

Usage:
Run from the folder where the Analysis folder is stored.
$ python3 Utilities/compare_jnd_across_backgrounds.py
Provide the subject names and a group name when prompted.

Author: Vijay Singh
Created: September 2026
Version: 1.1
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, sem

# Constants
BASE_FOLDER = os.getcwd()
CONDITIONS = ["condition_1", "condition_2", "condition_3"]
NUM_REPETITIONS = 1
TRANSPARENCY_ALPHA = 160/255  # Transparency alpha factor for background luminance


def get_user_input():
    """Get subject names and group name from the user"""
    print("\nJND Comparison Across Background Conditions")
    print("===========================================\n")

    while True:
        subject_input = input("Enter subject names to analyze (comma separated): ").strip()
        subject_names = [name.strip() for name in subject_input.split(',') if name.strip()]

        valid_subjects = []
        for subject in subject_names:
            if os.path.exists(os.path.join(BASE_FOLDER, "Analysis", subject)):
                valid_subjects.append(subject)
            else:
                print(f"Warning: No analysis folder found for subject '{subject}'")

        if len(valid_subjects) >= 2:
            group_name = input("Enter a name for this group (e.g., 'mean_threshold'): ").strip()
            while not group_name:
                group_name = input("Group name cannot be empty. Please enter a name: ").strip()
            return valid_subjects, group_name
        print(f"Need at least 2 subjects with valid analysis folders. Found {len(valid_subjects)}.")


def load_jnd_data(subject_names):
    """Collect the JND for every observer x block x background cell"""
    records = []
    for subject in subject_names:
        for condition in CONDITIONS:
            for repetition in range(1, NUM_REPETITIONS + 1):
                path = os.path.join(
                    BASE_FOLDER, "Analysis", subject, condition, f"Repetition_{repetition}",
                    f"thresholds_{subject}_{condition}_Repetition_{repetition}.txt")
                if not os.path.exists(path):
                    print(f"Warning: No threshold file at {path}")
                    continue

                table = pd.read_csv(path, sep='\t')
                for _, row in table.iterrows():
                    records.append({
                        'subject': subject,
                        'condition': condition,
                        # Background difference on the RGB scale, as in the threshold files
                        'background_diff_rgb': row['background_rgb'] - row['standard_rgb'],
                        # The JND is the distance between the 76% and 50% points
                        'jnd': row['threshold_76_luminance'] - row['threshold_50_luminance']})

    return pd.DataFrame(records)


def main():
    subject_names, group_name = get_user_input()

    print(f"\nLoading threshold data for subjects: {', '.join(subject_names)}\n")
    jnd_data = load_jnd_data(subject_names)

    if jnd_data.empty:
        print("No threshold data found!")
        return

    # One row per observer x block cell, one column per comparison background
    cells = jnd_data.pivot_table(index=['subject', 'condition'],
                                 columns='background_diff_rgb', values='jnd')
    cells = cells.dropna()
    backgrounds = sorted(cells.columns)

    if 0.0 not in backgrounds:
        raise ValueError("No equal-background condition found in the threshold files")

    # The three standard blocks are repeated observations on the same observers, so
    # average over blocks within each observer and treat the observer as the unit
    # of analysis
    observers = cells.groupby(level='subject').mean()

    lines = []
    lines.append("Just noticeable difference across comparison background conditions")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Subjects: {', '.join(subject_names)}")
    lines.append(f"Conditions: {', '.join(CONDITIONS)}")
    lines.append(f"Observer by block cells: {len(cells)}")
    lines.append(f"Observers: {len(observers)}")
    lines.append("")
    lines.append("The JND is the difference between the 76% and the 50% points of the fitted")
    lines.append("psychometric function. A purely additive contextual gain leaves the width of")
    lines.append("the psychometric function unchanged, so a JND that does not vary with the")
    lines.append("comparison background is consistent with alpha close to one.")
    lines.append("")
    lines.append("The three standard blocks are repeated observations on the same observers.")
    lines.append("The JND is therefore averaged over the three blocks within each observer, and")
    lines.append("the comparison across background conditions is made across observers.")
    lines.append("")
    lines.append("JND by comparison background (mean ± SEM across observers):")
    lines.append("")

    for background in backgrounds:
        offset = background / 255.0 * 105.6 * TRANSPARENCY_ALPHA
        values = observers[background].values
        lines.append(f"  B_c - B_s = {offset:+.1f} cd/m²:  "
                     f"{np.mean(values):.2f} ± {sem(values):.2f} cd/m²")
    lines.append("")

    test = friedmanchisquare(*[observers[b].values for b in backgrounds])
    lines.append(f"Friedman test across the three background conditions: "
                 f"chi-squared({len(backgrounds) - 1}) = {test.statistic:.2f}, "
                 f"p = {test.pvalue:.3f}")
    lines.append("")
    lines.append("For reference, the same test computed over the individual observer by block")
    lines.append("cells, which are not independent, gives:")
    cell_test = friedmanchisquare(*[cells[b].values for b in backgrounds])
    lines.append(f"  chi-squared({len(backgrounds) - 1}) = {cell_test.statistic:.2f}, "
                 f"p = {cell_test.pvalue:.3f} over {len(cells)} cells")
    lines.append("")
    lines.append("Note: this comparison constrains alpha only loosely, so it is reported as a")
    lines.append("consistency check rather than as an independent estimate of alpha.")

    output_dir = os.path.join(BASE_FOLDER, "Analysis", group_name)
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "jnd_across_backgrounds.txt")
    with open(summary_path, 'w') as handle:
        handle.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved summary to {summary_path}")

    observer_path = os.path.join(output_dir, "jnd_observer_values.txt")
    observers.to_csv(observer_path, sep='\t', float_format='%.4f')
    print(f"Saved per-observer JND values to {observer_path}")

    cell_path = os.path.join(output_dir, "jnd_cell_values.txt")
    cells.to_csv(cell_path, sep='\t', float_format='%.4f')
    print(f"Saved per-cell JND values to {cell_path}")


if __name__ == "__main__":
    main()
