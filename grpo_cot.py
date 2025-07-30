import torch
import os
import re
import csv
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOTrainer, GRPOConfig
from rouge_score import rouge_scorer

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
        self.log_file_path = os.path.join(log_dir, f"{model_name}_{task}_{serialization_type}_cot_training.csv")

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
        prompt = ""
        cot_instruction = "Think step-by-step to analyze the provided data, explaining your reasoning before giving the final answer."

        if "binary" in task:
            # Binary classification prompt (Yes/No question)
            prompt = f"{cot_instruction}\n\nData: {input_data}\n\nQuestion: Will a failure occur? End your response with 'Final Answer: Yes' or 'Final Answer: No'.\n\nReasoning:"
        
        elif "multiclass" in task:
            # Multiclass classification prompt (Choose from a list)
            choices = "No Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, and Tool Wear Failure"
            prompt = f"{cot_instruction}\n\nData: {input_data}\n\nQuestion: Which of the following is the most likely outcome? {choices}. End your response with 'Final Answer: [Chosen Failure Type]'.\n\nReasoning:"
        
        elif "regression" in task:
            # Regression prompt (Predict RUL)
            prompt = f"{cot_instruction}\n\nData: {input_data}\n\nQuestion: What is the Remaining Useful Life (RUL)? End your response with 'Final Answer: [RUL Value]'.\n\nReasoning:"
        
        else:
            # Fallback for any other undefined task
            prompt = example.get("prompt", "")

        example['prompt'] = prompt
        example['reasoning'] = example.get("completion", "")
        if 'completion' in example:
            del example['completion']

        return example

    dataset = dataset.map(formatting_prompts_func)

    # Split the dataset into training and evaluation sets (90% train and 10% eval)
    split_dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=SEED)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"Dataset split into {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples.")


    # 3. Define the reward functions
    def rouge_l_reward_fn(completions, **kwargs):
        """
        Computes the ROUGE-L score for the chain-of-thought validation.
        """
        # print kwargs to debug
        rewards = []
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        for completion, reasoning in zip(completions, kwargs['reasoning']):
            # Extract reasoning from the completion
            reasoning_part = completion.split("Final Answer:")[0].strip()
            # Calculate ROUGE-L score
            scores = scorer.score(reasoning, reasoning_part)
            rewards.append(scores['rougeL'].fmeasure)
        return rewards

    def accuracy_reward_fn(completions, **kwargs):
        """
        Computes the accuracy of the final answer.
        """
        rewards = []
        for completion, ground_truth in zip(completions, kwargs['ground_truth']):
            # Extract final answer from completion
            match = re.search(r"Final Answer:\s*(.*)", completion)
            if match:
                extracted_answer = match.group(1).strip()
                # Compare with ground truth
                if isinstance(ground_truth, str):
                    is_correct = 1.0 if extracted_answer.lower() == ground_truth.lower() else 0.0
                else: # Assuming numerical for regression
                    try:
                        is_correct = 1.0 if float(extracted_answer) == float(ground_truth) else 0.0
                    except ValueError:
                        is_correct = 0.0
                rewards.append(is_correct)
            else:
                rewards.append(0.0) # No final answer found
        return rewards
    
    def format_adherence_reward_fn(completions, **kwargs):
        """
        Rewards the model for adhering to the specified output format.
        It checks if the completion contains the "Final Answer:" string.
        """
        rewards = []
        for completion in completions:
            if "Final Answer:" in completion:
                rewards.append(1.0)  # Reward for correct format
            else:
                rewards.append(0.0)  # Penalty for incorrect format
        return rewards

    # 4. Configure the training arguments, with a dynamic output directory
    output_dir = f"./models/{os.path.basename(base_model.name_or_path)}_cot_{task}_{serialization_type}"
    print(f"Model will be saved to: {output_dir}")
    
    training_args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=1,
        report_to="none",
        logging_steps=10,
        # eval_strategy="no",
        remove_unused_columns=False,
        num_generations=4,
        max_prompt_length=128,
        max_completion_length=128,
        beta=0.1,
        loss_type="dr_grpo",

        eval_strategy="steps",
        eval_steps=50, 
        save_strategy="steps", 
        save_total_limit=2,      
        load_best_model_at_end=True,  
        metric_for_best_model="reward", 
    )

    # 5. Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # fan_in_fan_out=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Initialize the custom logger callback
    csv_logger = CSVCustomLogger(task=task, serialization_type=serialization_type, model_name=os.path.basename(base_model.name_or_path))

    # 6. Initialize and run the GRPOTrainer
    trainer = GRPOTrainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=[
            rouge_l_reward_fn, 
            accuracy_reward_fn, 
            format_adherence_reward_fn
        ],
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
