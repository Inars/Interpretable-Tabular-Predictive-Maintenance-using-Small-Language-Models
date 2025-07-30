import torch
import os
import re
import csv
import math
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOTrainer, GRPOConfig

from utils import DATASETS, SERIALIZATION_TYPES, MODELS, SEED

# CSV Logging Function
def log_to_csv(log_data, log_file_path):
    """
    Logs a dictionary of data to a specified CSV file path.
    """
    # Check if the file already exists to determine if we need to write headers
    file_exists = os.path.isfile(log_file_path)
    
    with open(log_file_path, mode='a', newline='') as csv_file:
        fieldnames = list(log_data.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(log_data)

# The Callback Class for Logging
class CSVCustomLogger(TrainerCallback):
    """
    A TrainerCallback that logs training metrics to a CSV file.
    """
    def __init__(self, task, serialization_type, model_name):
        self.task = task
        self.serialization_type = serialization_type
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file_path = os.path.join(log_dir, f"{model_name}_{task}_{serialization_type}_prediction_training.csv")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Event called by the Trainer when logging.
        """
        if logs is not None:
            # Add task context and log to the CSV file
            logs['task'] = self.task
            logs['serialization_type'] = self.serialization_type
            log_to_csv(logs, self.log_file_path)


def train_model_for_combination(base_model, tokenizer, task, serialization_type):
    """
    This function handles the training process for a single task and serialization type combination.
    """
    print("-" * 80)
    print(f"Starting training for task: '{task}', type: '{serialization_type}'")
    print("-" * 80)

    # 1. Construct the file path and check for existence
    dataset_name = DATASETS[task]
    file_path = f"./data/cot/{task}/{dataset_name}_{serialization_type}.csv"

    if not os.path.exists(file_path):
        print(f"SKIPPING: Data file not found at: {file_path}\n")
        return

    print(f"Loading dataset from: {file_path}")
    # Load the dataset from the specified CSV file
    try:
        dataset = load_dataset("csv", data_files=file_path, split="train")
    except Exception as e:
        print(f"ERROR: Could not load dataset at {file_path}. Error: {e}\n")
        return

    # 2. Prepare the dataset
    def formatting_prompts_func(example):
        """
        Dynamically generates the prompt based on the task type.
        """
        # This is the column containing the serialized machine/battery data.
        input_data = example.get("prompt", "") 
        completion = example.get("completion", "")
        completion = re.sub(r"Final Answer:.*", "", completion).strip()
        prompt = ""
        cot_instruction = "Think step-by-step to analyze the provided data, explaining your reasoning before giving the final answer."

        if "binary" in task:
            # Binary classification prompt (Yes/No question)
            prompt = f"{cot_instruction}\n\nData: {input_data}\n\nQuestion: Will a failure occur? End your response with 'Final Answer: Yes' or 'Final Answer: No'.\n\nReasoning:{completion}\nFinal Answer:"
        
        elif "multiclass" in task:
            # Multiclass classification prompt (Choose from a list)
            choices = "No Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, and Tool Wear Failure"
            prompt = f"{cot_instruction}\n\nData: {input_data}\n\nQuestion: Which of the following is the most likely outcome? {choices}. End your response with 'Final Answer: [Chosen Failure Type]'.\n\nReasoning:{completion}\nFinal Answer:"
        
        elif "regression" in task:
            # Regression prompt (Predict RUL)
            prompt = f"{cot_instruction}\n\nData: {input_data}\n\nQuestion: What is the Remaining Useful Life (RUL)? End your response with 'Final Answer: [RUL Value]'.\n\nReasoning:{completion}\nFinal Answer:"
        
        else:
            # Fallback for any other undefined task
            prompt = example.get("prompt", "")

        example['prompt'] = prompt
        if 'completion' in example:
            del example['completion']

        return example

    dataset = dataset.map(formatting_prompts_func)

    # 3. Define the reward functions
    def accuracy_reward_fn(completions, **kwargs):
        """
        Computes the reward based on the generated completion matching the ground truth.
        This function implements task-specific logic and assumes
        the completion is the direct prediction without any "Final Answer:" prefix.

        - For binary tasks, it checks for keywords ("yes"/"failure" or "no"/"no failure").
        - For multiclass tasks, it performs a substring search for the expected class.
        - For regression tasks, it compares the first extracted number with the ground truth.
        """
        # The trainer passes the ground_truth column from the dataset via kwargs
        ground_truths = kwargs.get('ground_truth', [])
        rewards = []

        for completion, ground_truth in zip(completions, ground_truths):
            # Default reward is 0.0 unless a condition is met
            reward = 0.0
            
            # Treat the entire completion as the answer, clean it for comparison.
            processed_answer = completion.strip().lower()

            # --- Task-specific Logic ---

            # 1. Binary Classification Task (ground_truth is 0 or 1)
            if isinstance(ground_truth, int):
                # Keywords for a "no failure" prediction
                negative_keywords = ["no", "no failure"]
                # Keywords for a "failure" prediction
                positive_keywords = ["yes", "failure"]
                
                if ground_truth == 0: # Expecting a "no failure" answer
                    if any(keyword in processed_answer for keyword in negative_keywords):
                        reward = 1.0
                elif ground_truth == 1: # Expecting a "failure" answer
                    if any(keyword in processed_answer for keyword in positive_keywords):
                        reward = 1.0
            
            # 2. Multiclass Classification Task (ground_truth is a string label)
            elif isinstance(ground_truth, str):
                # Check if the ground truth label appears anywhere in the processed answer.
                # Both are converted to lowercase for a case-insensitive comparison.
                if ground_truth.strip().lower() in processed_answer:
                    reward = 1.0

            # 3. Regression Task (ground_truth is a float)
            elif isinstance(ground_truth, float):
                # Find the first valid number (integer or float) in the processed answer
                num_match = re.search(r'[-+]?\d*\.?\d+', processed_answer)
                if num_match:
                    try:
                        predicted_value = float(num_match.group(0))
                        true_value = float(ground_truth)
                        
                        # Compare the first extracted number with the ground truth
                        # Using math.isclose() is a safe way to compare floats
                        if math.isclose(predicted_value, true_value, rel_tol=1e-6):
                            reward = 1.0
                    except (ValueError, TypeError):
                        # The matched text couldn't be converted to a float
                        reward = 0.0
            
            rewards.append(reward)
            
        return rewards

    # 4. Configure the training arguments, with a dynamic output directory
    output_dir = f"./models/{os.path.basename(base_model.name_or_path)}_prediction_{task}_{serialization_type}"
    print(f"Model will be saved to: {output_dir}")
    
    training_args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=2,
        report_to="none",
        logging_steps=10,
        eval_strategy="no",
        remove_unused_columns=False,
        num_generations=4,
        max_prompt_length=128,
        max_completion_length=10,
        beta=0.1,
        loss_type="dr_grpo",
    )

    # 5. Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Initialize the custom logger callback
    csv_logger = CSVCustomLogger(task=task, serialization_type=serialization_type, model_name=os.path.basename(base_model.name_or_path))

    # 6. Initialize and run the GRPOTrainer
    trainer = GRPOTrainer(
        model=base_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=accuracy_reward_fn,
        callbacks=[csv_logger],
    )

    print("Starting GRPO training...")
    trainer.train()
    print("Training finished.")

    # 7. Save the trained model
    trainer.save_model(output_dir)
    print(f"Model for '{task}/{serialization_type}' saved to {output_dir}\n")


def main():
    """
    Main function to load the base model and loop through all training combinations.
    """
    # Load the base model and tokenizer once to be reused in all training runs
    model_name = f"./models/{MODELS['slm']}"
    
    print("Loading base model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation='eager'
    )
    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Base model and tokenizer loaded.")

    # Loop through each task and serialization type to train a model
    for task in DATASETS.keys():
        for serialization_type in SERIALIZATION_TYPES:
            train_model_for_combination(model, tokenizer, task, serialization_type)
            
    print("All training combinations are complete.")


if __name__ == "__main__":
    main()
