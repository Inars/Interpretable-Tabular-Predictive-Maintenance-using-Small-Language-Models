import pandas as pd
from google import genai
from tqdm import tqdm
import os
import time
from collections import deque
import threading
from transformers import AutoTokenizer

from utils import DATASETS, GOOGLE_API_KEY, SERIALIZATION_TYPES

class RateLimiter:
    """
    A class to manage rate limiting for API requests based on requests per minute and tokens per minute.
    """
    def __init__(self, requests_per_minute, tokens_per_minute):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_timestamps = deque()
        self.token_counts = deque()
        self.lock = threading.Lock()

    def wait_if_needed(self, tokens_in_request):
        """
        Waits if the request or token limit has been reached.
        """
        with self.lock:
            while True:
                current_time = time.time()

                # Remove timestamps and token counts older than 60 seconds
                while self.request_timestamps and self.request_timestamps[0] < current_time - 60:
                    self.request_timestamps.popleft()
                    self.token_counts.popleft()

                # 1. Check Request Limit
                if len(self.request_timestamps) >= self.requests_per_minute:
                    sleep_time = self.request_timestamps[0] - (current_time - 60)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue # Re-evaluate conditions after sleeping

                # 2. Check Token Limit
                total_tokens_in_window = sum(count for _, count in self.token_counts)
                if total_tokens_in_window + tokens_in_request > self.tokens_per_minute:
                    if self.request_timestamps:
                        sleep_time = self.request_timestamps[0] - (current_time - 60)
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        continue # Re-evaluate conditions after sleeping
                    else:
                        # This case happens if a single request is larger than the limit
                        # We'll just wait a bit and retry, though this indicates a problem.
                        time.sleep(1) 
                        continue
                
                # If both checks pass, break the loop and proceed
                break


    def add_request(self, tokens_in_request):
        """
        Records a new request and its token count.
        """
        with self.lock:
            current_time = time.time()
            self.request_timestamps.append(current_time)
            self.token_counts.append((current_time, tokens_in_request))

# Initialize the rate limiter and the API client
rate_limiter = RateLimiter(requests_per_minute=30, tokens_per_minute=15000)

# Load the tokenizer once outside the loop
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

for task, dataset in DATASETS.items():
    input_file = f"./data/sampled/{task}/{dataset}_csv.csv"
    output_dir = f"./data/cot/{task}"
    template_file = f"./templates/{task}.txt"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Skipping: Input file not found at {input_file}")
        continue

    # Load the prompt template
    try:
        with open(template_file, "r") as file:
            template = file.read()
    except FileNotFoundError:
        print(f"Skipping: Template file not found at {template_file}")
        continue

    generated_completions = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Generating for {task}"):
        # Prepare the input dictionary
        if 'binary' in task:
            correct_answer = "Failure" if row['ground_truth'] == 1 else "No Failure"
        else:
            correct_answer = str(row['ground_truth'])
        
        input_data_dict = {
            "correct_answer": correct_answer,
            "data": row['prompt']
        }
            
        final_prompt = f"{template}\n**Input Data:** {str(input_data_dict)}\n**Generated Completion:**"
        
        # Estimate token count and wait if necessary
        input_ids = tokenizer.encode(final_prompt)
        estimated_tokens = len(input_ids) 
        rate_limiter.wait_if_needed(estimated_tokens)

        # Call the LLM
        try:
            client = genai.Client(api_key=GOOGLE_API_KEY,)
            completion_text = client.models.generate_content(
                model="gemma-3-27b-it",
                contents=final_prompt
            ).text
            rate_limiter.add_request(estimated_tokens + len(completion_text.split()))

        except Exception as e:
            print(f"An error occurred at row {index}: {e}")
            completion_text = "ERROR: Could not generate completion."
            rate_limiter.add_request(estimated_tokens) # Still count the request

        generated_completions.append(completion_text)

    # Add completions to the dataframe and save to multiple formats
    for file_format in SERIALIZATION_TYPES:
        input_format_file = f"./data/sampled/{task}/{dataset}_{file_format}.csv"
        output_format_file = f"{output_dir}/{dataset}_{file_format}.csv"
        try:
            format_df = pd.read_csv(input_format_file)
            if len(generated_completions) == len(format_df):
                output_df = pd.DataFrame({
                    'prompt': format_df['prompt'],
                    'ground_truth': format_df['ground_truth'],
                    'completion': generated_completions
                })
                output_df.to_csv(output_format_file, index=False)
            else:
                 print(f"Skipping save for {output_format_file} due to length mismatch.")

        except FileNotFoundError:
            print(f"Skipping save: Input file for format '{file_format}' not found.")

    print(f"Finished processing for {task}. Output saved to {output_dir}\n")

print("All tasks completed.")