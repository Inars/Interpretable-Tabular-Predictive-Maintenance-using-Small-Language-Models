import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from tqdm import tqdm  # Import tqdm
from utils import MODELS, DATASETS, SERIALIZATION_TYPES, SEED

# --- Configuration ---
BASE_MODEL_PATH = "./models"
BASE_DATA_PATH = "./data/cot"
LOG_DIR = "./logs"
SLM_NAME = MODELS.get("slm")

# --- Helper Functions ---

def load_model_and_tokenizer(model_path):
    """
    Loads the language model and tokenizer from a specified path.
    Returns None if the path does not exist.
    """
    if not os.path.exists(model_path):
        print(f"Warning: Model path not found at '{model_path}'. Skipping.")
        return None, None

    print(f"Loading model from: {model_path}")
    try:
        # Check for CUDA availability for GPU acceleration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

        # Set pad_token_id if it's not already set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = model.config.eos_token_id

        return model, tokenizer
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None

def construct_prompt(task, input_data):
    """
    Constructs the full prompt based on the task type and input data.
    """
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
        # Fallback in case the task does not match known types.
        print(f"Warning: Unknown task type '{task}'. Using a generic prompt format.")
        prompt = f"Data: {input_data}\n\nAnalyze the data and provide a response."

    return prompt

def generate_slm_response(model, tokenizer, prompt):
    """
    Generates a response from the SLM for a given prompt.
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode the response, skipping special tokens
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the original prompt from the response
    # This ensures we only have the newly generated text
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    return response

def calculate_rouge_l(reference, prediction):
    """
    Calculates the ROUGE-L F1 score between a reference and a prediction.
    """
    # Ensure inputs are strings
    reference = str(reference) if reference is not None else ""
    prediction = str(prediction) if prediction is not None else ""

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores['rougeL'].fmeasure

# --- Main Evaluation Loop ---

def main():
    """
    Main function to run the CoT evaluation script.
    """
    if not SLM_NAME:
        print("Error: SLM model name not defined in utils.py. Exiting.")
        return

    # Create the log directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)

    # List to store summary results from all runs
    all_runs_summary = []

    # Iterate over each task and serialization type
    for task, dataset_name in DATASETS.items():
        for serialization_type in SERIALIZATION_TYPES:
            print("-" * 80)
            print(f"Processing Task: '{task}', Serialization: '{serialization_type}'")

            # 1. Construct paths
            model_path = os.path.join(BASE_MODEL_PATH, f"{SLM_NAME}_cot_{task}_{serialization_type}")
            data_path = os.path.join(BASE_DATA_PATH, task, f"{dataset_name}_{serialization_type}.csv")
            log_path = os.path.join(LOG_DIR, f"{SLM_NAME}_{task}_{serialization_type}_cot_evaluation.csv")

            # 2. Check if data file exists
            if not os.path.exists(data_path):
                print(f"Warning: Data file not found at '{data_path}'. Skipping.")
                continue

            # 3. Load Model and Tokenizer
            model, tokenizer = load_model_and_tokenizer(model_path)
            if model is None or tokenizer is None:
                error_df = pd.DataFrame([{"error": f"Model not found at {model_path}"}])
                error_df.to_csv(log_path, index=False)
                continue

            # 4. Load Dataset
            print(f"Loading data from: {data_path}")
            eval_df = pd.read_csv(data_path)
            # use 10% of the data for evaluation
            eval_df = eval_df.sample(frac=0.1, random_state=SEED)

            required_cols = ["prompt", "ground_truth", "completion"]
            if not all(col in eval_df.columns for col in required_cols):
                print(f"Error: Data file {data_path} is missing one of the required columns: {required_cols}. Skipping.")
                continue

            # 5. Perform Evaluation
            results = []
            total_rows = len(eval_df)
            
            for index, row in tqdm(eval_df.iterrows(), total=total_rows, desc=f"Processing {task} ({serialization_type})"):
                # The 'prompt' column from the CSV is treated as the raw input data.
                input_data = row['prompt']
                ground_truth = row['ground_truth']
                reference_completion = row['completion']

                # Construct the full prompt dynamically based on the current task.
                full_prompt = construct_prompt(task, input_data)

                # Generate the SLM's CoT response using the constructed prompt.
                slm_response = generate_slm_response(model, tokenizer, full_prompt)

                # Calculate ROUGE-L score
                rouge_l_score = calculate_rouge_l(reference_completion, slm_response)

                results.append({
                    # Save the dynamically constructed prompt for clarity in the logs.
                    "prompt": full_prompt,
                    "ground_truth": ground_truth,
                    "reference_completion": reference_completion,
                    "slm_response": slm_response,
                    "rougeL_f1_score": rouge_l_score
                })

            # 6. Save Results for the current run
            if results:
                results_df = pd.DataFrame(results)
                results_df.to_csv(log_path, index=False)
                print(f"\nEvaluation complete. Results saved to: {log_path}")

                # Calculate the average score for the run
                avg_score = results_df['rougeL_f1_score'].mean()
                print(f"Average ROUGE-L F1 Score for this run: {avg_score:.4f}")

                # Append summary for this run to the master list
                all_runs_summary.append({
                    "task": task,
                    "serialization_type": serialization_type,
                    "model_name": f"{SLM_NAME}_{task}_{serialization_type}",
                    "log_file": log_path,
                    "average_rougeL_f1_score": avg_score
                })
            else:
                print("No results were generated for this run.")

    # After all runs are complete, save the overall summary
    if all_runs_summary:
        summary_df = pd.DataFrame(all_runs_summary)
        summary_log_path = os.path.join(LOG_DIR, f"{SLM_NAME}_cot_evaluation_summary.csv")
        summary_df.to_csv(summary_log_path, index=False)
        print("-" * 80)
        print(f"Summary of all runs saved to: {summary_log_path}")
        print(summary_df)

    print("-" * 80)
    print("All tasks and serialization types have been processed.")

if __name__ == "__main__":
    main()