import numpy as np
import itertools
import random
import colorsys
import os

def lhs_to_rgb(l, h, s):
    """Converts LHS (Lightness, Hue, Saturation) to RGB."""
    return colorsys.hls_to_rgb(h, l, s)

def generate_trials(reference_color_lhs, comparison_colors_lhs, background_colors_lhs, fixed_background_lhs, num_repetitions, output_file="colors.txt", condition_file="condition.txt", base_folder="."):
    """
    Generates a list of trials for a 2AFC experiment and saves them in a text file.
    
    Parameters:
        reference_color_lhs (list of tuples): LHS values of the reference square.
        comparison_colors_lhs (list of tuples): List of LHS values for comparison squares.
        background_colors_lhs (list of tuples): List of LHS values for backgrounds.
        fixed_background_lhs (list of tuples): Fixed background color for the reference square.
        num_repetitions (int): Number of times each color combination should be repeated.
        output_file (str): Name of the output file.
        condition_file (str): Name of the condition file.
        base_folder (str): Base folder where the files will be saved.
    """
    trials = []
    
    # Convert LHS to RGB
    reference_color = [lhs_to_rgb(*r) for r in reference_color_lhs]  # List of RGB tuples
    comparison_colors = [lhs_to_rgb(*c) for c in comparison_colors_lhs]  # List of RGB tuples
    background_colors = [lhs_to_rgb(*b) for b in background_colors_lhs]  # List of RGB tuples
    fixed_background = [lhs_to_rgb(*f) for f in fixed_background_lhs]  # List of RGB tuples
    
    # Pair each comparison color RGB with its corresponding LHS values
    comparison_pairs = list(zip(comparison_colors, comparison_colors_lhs))
    
    # Generate all possible combinations of (background2, comparison color)
    all_combinations = list(itertools.product(background_colors, comparison_pairs))
    
    # Initialize order number counter
    order_number = 1
    
    # Repeat trials as specified
    for _ in range(num_repetitions):
        for bg2_color, (comp_color, comp_color_lhs) in all_combinations:
            # Randomly swap reference and comparison squares
            if random.choice([True, False]):
                square1, square2 = reference_color[0], comp_color  # Use first reference color
                square1_background, square2_background = fixed_background[0], bg2_color  # Use first fixed background
                expected_response = 1 if reference_color_lhs[0][0] > comp_color_lhs[0] else 2
            else:
                square1, square2 = comp_color, reference_color[0]  # Use first reference color
                square1_background, square2_background = bg2_color, fixed_background[0]  # Use first fixed background
                expected_response = 2 if reference_color_lhs[0][0] > comp_color_lhs[0] else 1
            
            # Flatten the trial data and append
            trial_data = (
                *square1_background,  # Background 1 (RGB)
                *square2_background,  # Background 2 (RGB)
                *square1,  # Square 1 (RGB)
                *square2,  # Square 2 (RGB)
                order_number,  # Order number (integer)
                expected_response  # Expected response (integer)
            )
            trials.append(trial_data)
            
            # Increment the order number
            order_number += 1
    
    # Shuffle trials to randomize presentation order
    random.shuffle(trials)
    
    # Create the base folder if it doesn't exist
    os.makedirs(base_folder, exist_ok=True)
    
    # Save to file
    output_path = os.path.join(base_folder, output_file)
    with open(output_path, "w") as f:
        for trial in trials:
            # Format the order number as an integer, and the rest to 3 decimal places
            formatted_trial = [f"{x:.3f}" if isinstance(x, float) else f"{x:d}" for x in trial]
            f.write(",".join(formatted_trial) + "\n")
    
    # Save condition parameters to a separate file
    condition_path = os.path.join(base_folder, condition_file)
    with open(condition_path, "w") as f:
        f.write("Parameters used to generate the output file:\n")
        f.write(f"Reference Color LHS: {reference_color_lhs}\n")
        f.write(f"Comparison Colors LHS: {comparison_colors_lhs}\n")
        f.write(f"Background Colors LHS: {background_colors_lhs}\n")
        f.write(f"Fixed Background LHS: {fixed_background_lhs}\n")
        f.write(f"Number of Repetitions: {num_repetitions}\n")
        f.write(f"Output File: {output_file}\n")
        f.write(f"Condition File: {condition_file}\n")
        f.write(f"Base Folder: {base_folder}\n")
    
    print(f"Generated {len(trials)} trials and saved to {output_path}")
    print(f"Condition parameters saved to {condition_path}")

def generate_lhs_tuples(lightness_start, lightness_end, lightness_inc,
                        hue_start=0.0, hue_end=0.0, hue_inc=0.1,
                        saturation_start=0.0, saturation_end=0.0, saturation_inc=0.1):
    """
    Generates a list of LHS tuples with separate start, end, and increment values for lightness, hue, and saturation.
    
    Parameters:
        lightness_start (float): Starting value for lightness.
        lightness_end (float): Ending value for lightness.
        lightness_inc (float): Increment value for lightness.
        hue_start (float): Starting value for hue.
        hue_end (float): Ending value for hue.
        hue_inc (float): Increment value for hue.
        saturation_start (float): Starting value for saturation.
        saturation_end (float): Ending value for saturation.
        saturation_inc (float): Increment value for saturation.
    
    Returns:
        list: A list of tuples in the format [(lightness, hue, saturation)].
    """
    tuples = []
    
    # Generate lightness values
    lightness_values = []
    current = lightness_start
    while current <= lightness_end:
        lightness_values.append(round(current, 2))  # Round to avoid floating-point inaccuracies
        current += lightness_inc
    
    # Generate hue values
    hue_values = []
    current = hue_start
    while current <= hue_end:
        hue_values.append(round(current, 2))  # Round to avoid floating-point inaccuracies
        current += hue_inc
    
    # Generate saturation values
    saturation_values = []
    current = saturation_start
    while current <= saturation_end:
        saturation_values.append(round(current, 2))  # Round to avoid floating-point inaccuracies
        current += saturation_inc
    
    # Generate all combinations of lightness, hue, and saturation
    for l in lightness_values:
        for h in hue_values:
            for s in saturation_values:
                tuples.append((l, h, s))
    
    return tuples

# Example usage:
reference_color_lhs = generate_lhs_tuples(0.70, 0.70, 0.01)  # List of tuples for reference color
fixed_background_lhs = generate_lhs_tuples(0.7, 0.7, 0.01)  # List of tuples for fixed background
background_colors_lhs = generate_lhs_tuples(0.65, 0.76, 0.05)  # List of tuples for background colors
comparison_colors_lhs = generate_lhs_tuples(0.65, 0.76, 0.01)  # List of tuples for comparison colors
num_repetitions = 30  # Each condition repeated once

# Specify the base folder and file names
base_folder = "experiment_data"
output_file = "std_0_7_sb_0_7_cb_0_65_0_75_0_05_cmp_0_65_0_75_0_01.txt"
condition_file = "experiment_conditions.txt"

generate_trials(reference_color_lhs, comparison_colors_lhs, background_colors_lhs, fixed_background_lhs, num_repetitions, 
                output_file=output_file, condition_file=condition_file, base_folder=base_folder)