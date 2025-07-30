import os
import pandas as pd
import random

from utils import DATASETS, SERIALIZATION_TYPES, SEED

# Base path for input data
BASE_INPUT_PATH = './data/serialized/'

# Path for saving the new datasets
BASE_OUTPUT_PATH = './data/sampled/'

# The desired number of instances in the new datasets
NEW_DATASET_SIZE = 5000


def create_sampled_dataset():
    """
    Loads datasets, samples them based on specific rules for each task,
    and saves the new smaller datasets.
    """
    print("Starting dataset sampling process...")

    # Create the output directory if it doesn't exist
    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)

    # Iterate over each task and dataset defined in the DATASETS dictionary
    for task, dataset_name in DATASETS.items():
        print(f"\nProcessing Task: '{task}', Dataset: '{dataset_name}'")

        # Create a subdirectory for the task in the output path
        task_output_dir = os.path.join(BASE_OUTPUT_PATH, task)
        os.makedirs(task_output_dir, exist_ok=True)

        # Iterate over each serialization type
        for s_type in SERIALIZATION_TYPES:
            # Construct the full path to the input file
            input_file_path = os.path.join(BASE_INPUT_PATH, task, f"{dataset_name}_{s_type}.csv")

            # Check if the source file exists before trying to load it
            if not os.path.exists(input_file_path):
                print(f"  - WARNING: File not found, skipping: {input_file_path}")
                continue

            print(f"  - Loading and processing serialization type: '{s_type}'")

            # Load the original dataset
            try:
                df = pd.read_csv(input_file_path)
            except Exception as e:
                print(f"    - ERROR: Could not read file {input_file_path}. Error: {e}")
                continue

            new_df = None

            # --- Apply sampling logic based on the task type ---

            if task == "binary":
                # Select all instances with the "Failure" label
                failure_instances = df[df['ground_truth'] == 1]
                # Select the remaining instances from the "No Failure" class
                num_needed = NEW_DATASET_SIZE - len(failure_instances)
                if num_needed > 0:
                    no_failure_instances = df[df['ground_truth'] == 0].sample(n=num_needed, random_state=SEED)
                    # Combine and shuffle the two sets of instances
                    new_df = pd.concat([failure_instances, no_failure_instances]).sample(frac=1, random_state=SEED).reset_index(drop=True)
                else:
                    # If there are more 'Failure' instances than the target size, just sample them
                    new_df = failure_instances.sample(n=NEW_DATASET_SIZE, random_state=SEED).reset_index(drop=True)


            elif task == "multiclass":
                # Select all instances that are not "No Failure"
                failure_instances = df[df['ground_truth'] != 'No Failure']
                # Select the rest from the "No Failure" class
                num_needed = NEW_DATASET_SIZE - len(failure_instances)

                if num_needed > 0:
                    no_failure_instances = df[df['ground_truth'] == 'No Failure'].sample(n=num_needed, random_state=SEED)
                    # Combine and shuffle
                    new_df = pd.concat([failure_instances, no_failure_instances]).sample(frac=1, random_state=SEED).reset_index(drop=True)
                else:
                    # If non-"No Failure" instances exceed target size, sample from them
                    new_df = failure_instances.sample(n=NEW_DATASET_SIZE, random_state=SEED).reset_index(drop=True)

            elif task == "regression":
                # For regression, simply select NEW_DATASET_SIZE random instances
                new_df = df.sample(n=NEW_DATASET_SIZE, random_state=SEED).reset_index(drop=True)

            # --- Save the new dataset ---
            if new_df is not None:
                # Construct the output file path
                output_file_name = f"{dataset_name}_{s_type}.csv"
                output_file_path = os.path.join(task_output_dir, output_file_name)

                # Save the new dataframe to a CSV file
                new_df.to_csv(output_file_path, index=False)
                print(f"    - Successfully created new dataset at: {output_file_path}")

    print("\nDataset sampling process finished.")

if __name__ == "__main__":
    create_sampled_dataset()
