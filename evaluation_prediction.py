import os
import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score, mean_absolute_error, accuracy_score
from tqdm import tqdm
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = model.config.eos_token_id

        return model, tokenizer
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None

def construct_prompt(task, input_data, completion):
    """
    Constructs the full prompt based on the task type and input data.
    """
    cot_instruction = "Think step-by-step to analyze the provided data, explaining your reasoning before giving the final answer."

    if "binary" in task:
        prompt = f"{cot_instruction}\n\nData: {input_data}\n\nQuestion: Will a failure occur? End your response with 'Final Answer: Yes' or 'Final Answer: No'.\n\nReasoning:{completion}\nFinal Answer:"
    elif "multiclass" in task:
        choices = "No Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, and Tool Wear Failure"
        prompt = f"{cot_instruction}\n\nData: {input_data}\n\nQuestion: Which of the following is the most likely outcome? {choices}. End your response with 'Final Answer: [Chosen Failure Type]'.\n\nReasoning:{completion}\nFinal Answer:"
    elif "regression" in task:
        prompt = f"{cot_instruction}\n\nData: {input_data}\n\nQuestion: What is the Remaining Useful Life (RUL)? End your response with 'Final Answer: [RUL Value]'.\n\nReasoning:{completion}\nFinal Answer:"
    else:
        print(f"Warning: Unknown task type '{task}'. Using a generic prompt format.")
        prompt = f"Data: {input_data}\n\nAnalyze the data and provide a response."

    return prompt

def generate_slm_response(model, tokenizer, prompt):
    """
    Generates a response from the SLM for a given prompt.
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    return response

def calculate_mae(reference, prediction):
    """
    Calculates the Mean Absolute Error for a single prediction.
    Returns the absolute difference. Returns None if conversion fails.
    """
    if prediction is None:
        return None
    try:
        return mean_absolute_error([float(reference)], [float(prediction)])
    except (ValueError, TypeError):
        return None

# --- Main Evaluation Loop ---

def main():
    """
    Main function to run the CoT evaluation script.
    """
    if not SLM_NAME:
        print("Error: SLM model name not defined in utils.py. Exiting.")
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    all_runs_summary = []

    for task, dataset_name in DATASETS.items():
        for serialization_type in SERIALIZATION_TYPES:
            print("-" * 80)
            print(f"Processing Task: '{task}', Serialization: '{serialization_type}'")

            model_path = os.path.join(BASE_MODEL_PATH, f"{SLM_NAME}_prediction_{task}_{serialization_type}")
            data_path = os.path.join(BASE_DATA_PATH, task, f"{dataset_name}_{serialization_type}.csv")
            log_path = os.path.join(LOG_DIR, f"{SLM_NAME}_{task}_{serialization_type}_prediction_evaluation.csv")

            if not os.path.exists(data_path):
                print(f"Warning: Data file not found at '{data_path}'. Skipping.")
                continue

            model, tokenizer = load_model_and_tokenizer(model_path)
            if model is None or tokenizer is None:
                pd.DataFrame([{"error": f"Model not found at {model_path}"}]).to_csv(log_path, index=False)
                continue

            print(f"Loading data from: {data_path}")
            eval_df = pd.read_csv(data_path).sample(frac=0.1, random_state=SEED)
            eval_df = eval_df.head(2)

            required_cols = ["prompt", "ground_truth", "completion"]
            if not all(col in eval_df.columns for col in required_cols):
                print(f"Error: Data file {data_path} is missing required columns: {required_cols}. Skipping.")
                continue

            results = []
            all_references = []
            all_predictions = []

            for index, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f"Processing {task} ({serialization_type})"):
                input_data, ground_truth, reference_completion = row['prompt'], row['ground_truth'], re.sub(r"Final Answer:.*", "", row['completion']).strip()
                full_prompt = construct_prompt(task, input_data, reference_completion)
                slm_response = generate_slm_response(model, tokenizer, full_prompt)
                
                final_answer = None
                response_lower = slm_response.lower()

                if "binary" in task:
                    if ground_truth == 0: ground_truth = "No Failure"
                    elif ground_truth == 1: ground_truth = "Failure"
                    if "no" in response_lower or "no failure" in response_lower: final_answer = "No Failure"
                    elif "yes" in response_lower or "failure" in response_lower: final_answer = "Failure"
                elif "multiclass" in task:
                    choices = ["No Failure", "Heat Dissipation Failure", "Power Failure", "Overstrain Failure", "Tool Wear Failure"]
                    for choice in choices:
                        if choice.lower() in response_lower:
                            final_answer = choice
                            break
                elif "regression" in task:
                    match = re.search(r'-?\d+\.?\d*', response_lower)
                    if match: final_answer = match.group(0)
                
                # Store results for aggregation
                if final_answer is not None:
                    all_references.append(str(ground_truth))
                    all_predictions.append(final_answer)

                results.append({
                    "prompt": full_prompt,
                    "ground_truth": ground_truth,
                    "slm_response": slm_response,
                    "parsed_answer": final_answer,
                    "mae_score": calculate_mae(ground_truth, final_answer) if task == "regression" else None
                })

            if results:
                results_df = pd.DataFrame(results)
                results_df.to_csv(log_path, index=False)
                print(f"\nEvaluation complete. Results saved to: {log_path}")

                # --- Correct, Aggregated Metric Calculation ---
                overall_accuracy, overall_f1, avg_mae = None, None, None

                if task in ["binary", "multiclass"]:
                    overall_accuracy = accuracy_score(all_references, all_predictions)
                    overall_f1 = f1_score(all_references, all_predictions, average='macro', zero_division=0)
                    print(f"Overall Accuracy for this run: {overall_accuracy:.4f}")
                    print(f"Overall F1-Score for this run: {overall_f1:.4f}")
                elif task == "regression":
                    avg_mae = results_df['mae_score'].mean()
                    if not pd.isna(avg_mae):
                        print(f"Average MAE score for this run: {avg_mae:.4f}")

                all_runs_summary.append({
                    "task": task, "serialization_type": serialization_type,
                    "model_name": f"{SLM_NAME}_{task}_{serialization_type}",
                    "log_file": log_path, "average_accuracy_score": overall_accuracy,
                    "average_f1_score": overall_f1, "average_mae_score": avg_mae
                })
            else:
                print("No results were generated for this run.")

    if all_runs_summary:
        summary_df = pd.DataFrame(all_runs_summary)
        summary_log_path = os.path.join(LOG_DIR, f"{SLM_NAME}_prediction_evaluation_summary.csv")
        summary_df.to_csv(summary_log_path, index=False)
        print("-" * 80)
        print(f"Summary of all runs saved to: {summary_log_path}")
        print(summary_df)

    print("-" * 80)
    print("All tasks and serialization types have been processed.")

if __name__ == "__main__":
    main()