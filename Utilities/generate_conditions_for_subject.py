import itertools
import random
import os
import re

def gamma_correct(R,G,B):

    # Apply gamma correction
    r_corrected = (R/255) ** (1.0 / b_values['R'])
    g_corrected = (G/255) ** (1.0 / b_values['G'])
    b_corrected = (B/255) ** (1.0 / b_values['B'])

    return(round(r_corrected*255), round(g_corrected*255), round(b_corrected*255))


# MODIFY THIS FUNCTION
def generate_trials(reference_color_rgb, comparison_colors_rgb, background_colors_rgb, fixed_background_rgb,
                    num_trials, output_file="colors.txt", condition_file="condition.txt", base_folder="."):

    trials_gamma = []
    trials_linear = []

    order_number = 1

    # Prepare rgb combinations for pairing
    comparison_pairs_rgb = list(zip(comparison_colors_rgb, comparison_colors_rgb))
    all_combinations_rgb = list(itertools.product(background_colors_rgb, comparison_pairs_rgb))

    for _ in range(num_trials):
        for bg2_rgb, (comp_rgb, _) in all_combinations_rgb:
            # Always use the first reference color and first fixed background
            ref_rgb = reference_color_rgb[0]
            fixed_bg_rgb = fixed_background_rgb[0]

            # Randomize reference/comparison square positions
            if random.choice([True, False]):
                rgb_sq1, rgb_sq2 = ref_rgb, comp_rgb
                rgb_bg1, rgb_bg2 = fixed_bg_rgb, bg2_rgb
                expected_response = 1 if ref_rgb[0] > comp_rgb[0] else 2
            else:
                rgb_sq1, rgb_sq2 = comp_rgb, ref_rgb
                rgb_bg1, rgb_bg2 = bg2_rgb, fixed_bg_rgb
                expected_response = 2 if ref_rgb[0] > comp_rgb[0] else 1

            # Convert to RGB
            sq1_gamma = gamma_correct(*rgb_sq1)
            sq2_gamma = gamma_correct(*rgb_sq2)
            bg1_gamma = gamma_correct(*rgb_bg1)
            bg2_gamma = gamma_correct(*rgb_bg2)

            sq1_linear = rgb_sq1
            sq2_linear = rgb_sq2
            bg1_linear = rgb_bg1
            bg2_linear = rgb_bg2

            trial_gamma = (*bg1_gamma, *bg2_gamma, *sq1_gamma, *sq2_gamma, order_number, expected_response)
            trial_linear = (*bg1_linear, *bg2_linear, *sq1_linear, *sq2_linear, order_number, expected_response)

            trials_gamma.append(trial_gamma)
            trials_linear.append(trial_linear)

            order_number += 1

    # Shuffle both lists using the same permutation
    permutation = list(range(len(trials_gamma)))
    random.shuffle(permutation)
    trials_gamma = [trials_gamma[i] for i in permutation]
    trials_linear = [trials_linear[i] for i in permutation]

    os.makedirs(base_folder, exist_ok=True)

    # Save gamma-corrected file
    gamma_path = os.path.join(base_folder, output_file.replace(".txt", "_gamma_corrected.txt"))
    with open(gamma_path, "w") as f:
        for trial in trials_gamma:
            f.write(",".join(f"{x:d}" for x in trial) + "\n")

    # Save non-gamma-corrected (linear) file
    linear_path = os.path.join(base_folder, output_file)
    with open(linear_path, "w") as f:
        for trial in trials_linear:
            f.write(",".join(f"{x:d}" for x in trial) + "\n")

    # Save condition parameters
    condition_path = os.path.join(base_folder, condition_file)
    with open(condition_path, "w") as f:
        f.write("Parameters used to generate the output file:\n")
        f.write(f"Reference Color rgb: {reference_color_rgb}\n")
        f.write(f"Comparison Colors rgb: {comparison_colors_rgb}\n")
        f.write(f"Background Colors rgb: {background_colors_rgb}\n")
        f.write(f"Fixed Background rgb: {fixed_background_rgb}\n")
        f.write(f"Number of Repetitions: {num_trials}\n")
        f.write(f"Output File (Gamma Corrected): {gamma_path}\n")
        f.write(f"Output File (Linear RGB): {linear_path}\n")
        f.write(f"Condition File: {condition_file}\n")
        f.write(f"Base Folder: {base_folder}\n")
        # f.write(f"Gamma Correction Exponent File: {input_exponent_file}\n")

    print(f"Generated {len(trials_gamma)} gamma-corrected trials and saved to {gamma_path}")
    print(f"Generated {len(trials_linear)} linear (non-gamma) trials and saved to {linear_path}")
    print(f"Condition parameters saved to {condition_path}")

def generate_achromatic_tuples(start, end, inc):
    """
    Generates a list of rgb tuples with separate start, end, and increment values for lightness, hue, and saturation.
    
    Parameters:
        start (float): Starting value for R,G,B.
        end (float): Ending value for R,G,B.
        inc (float): Increment value for R,G,B.
    
    Returns:
        list: A list of tuples in the format [(R, G, B)].
    """
    tuples = []
    
    # Generate R channel values
    R_values = []
    current = start
    while current <= end:
        R_values.append(round(current, 2))  # Round to avoid floating-point inaccuracies
        current += inc
    
    # Generate all combinations of lightness, hue, and saturation
    for R in R_values:
        tuples.append((R, R, R))
    
    return tuples

def generate_conditions_for_subject(subject_name, base_folder, num_repetitions):
    """
    Generates experiment conditions for a subject and saves the files in the specified folder structure.
    
    Parameters:
        subject_name (str): Name of the subject.
        base_folder (str): Base folder where the data will be saved.
    """

    # Define the five conditions
    conditions = {
        "condition_demo": {
            "reference_color_rgb": generate_achromatic_tuples(40, 40, 1),
            "comparison_colors_rgb": generate_achromatic_tuples(30, 51, 2),
            "background_colors_rgb": generate_achromatic_tuples(30, 51, 10),
            "fixed_background_rgb": generate_achromatic_tuples(40, 40, 1),
            "num_trials": 2,
        },
        "condition_0": {
            "reference_color_rgb": generate_achromatic_tuples(70, 70, 1),
            "comparison_colors_rgb": generate_achromatic_tuples(60, 81, 2),
            "background_colors_rgb": generate_achromatic_tuples(70, 70, 10),
            "fixed_background_rgb": generate_achromatic_tuples(70, 70, 10),
            "num_trials": 30,
        },
        "condition_1": {
            "reference_color_rgb": generate_achromatic_tuples(40, 40, 1),
            "comparison_colors_rgb": generate_achromatic_tuples(30, 51, 2),
            "background_colors_rgb": generate_achromatic_tuples(30, 51, 10),
            "fixed_background_rgb": generate_achromatic_tuples(40, 40, 1),
            "num_trials": 30,
        },
        "condition_2": {
            "reference_color_rgb": generate_achromatic_tuples(70, 70, 1),
            "comparison_colors_rgb": generate_achromatic_tuples(60, 81, 2),
            "background_colors_rgb": generate_achromatic_tuples(60, 81, 10),
            "fixed_background_rgb": generate_achromatic_tuples(70, 70, 1),
            "num_trials": 30,
        },
        "condition_3": {
            "reference_color_rgb": generate_achromatic_tuples(100, 100, 1),
            "comparison_colors_rgb": generate_achromatic_tuples(90, 111, 2),
            "background_colors_rgb": generate_achromatic_tuples(90, 111, 10),
            "fixed_background_rgb": generate_achromatic_tuples(100, 100, 1),
            "num_trials": 30,
        },
    }

    # Iterate over conditions and repetitions
    for condition_name, params in conditions.items():
        for repetition in range(1, num_repetitions + 1):
            # Define the output folder
            output_folder = os.path.join(base_folder, "Data", subject_name, condition_name, f"Repetition_{repetition}")
            os.makedirs(output_folder, exist_ok=True)

            # Define the output file and condition file names
            output_file = f"trials_{condition_name}_rep_{repetition}.txt"
            condition_file = f"conditions_{condition_name}_rep_{repetition}.txt"

            # Generate trials and save files
            generate_trials(
                reference_color_rgb=params["reference_color_rgb"],
                comparison_colors_rgb=params["comparison_colors_rgb"],
                background_colors_rgb=params["background_colors_rgb"],
                fixed_background_rgb=params["fixed_background_rgb"],
                num_trials=params["num_trials"],
                output_file=output_file,
                condition_file=condition_file,
                base_folder=output_folder,
            )

            print(f"Generated {condition_name}, Repetition {repetition} for subject {subject_name} in {output_folder}")

if __name__ == "__main__":
    # Example usage
    subject_name = input("Enter subject name (Ex: subject_apple):  ")

    input_exponent_file = os.path.join(os.getcwd(), "calibration_latest", "exponent_lastest.txt")
    print("The exponent values for gamma correction is set to the latest calibration result")

    # Read exponent values (first part of original read_b_values function)
    b_values = {}
    with open(input_exponent_file, 'r') as file:
        for line in file:
            match = re.match(r'([RGB]) gamma \(γ\): ([\d.]+)', line)
            if match:
                channel = match.group(1)
                b_value = float(match.group(2))
                b_values[channel] = b_value

    # Number of repetitions for each condition
    num_repetitions = 1

    # Path to base folder
    base_folder = os.getcwd() 

    generate_conditions_for_subject(subject_name, base_folder = base_folder, num_repetitions=num_repetitions)