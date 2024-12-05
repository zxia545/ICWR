import os
import json
import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from openai import OpenAI
import requests

def wait_for_server(url, timeout=300, sleep_interval=5):
    """Waits until the server at the specified URL is up and running."""
    start_time = time.time()
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 404:
                print("Server is up and running!")
                break
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"Unexpected error while waiting for server: {e}")
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Server did not start within {timeout} seconds.")
        time.sleep(sleep_interval)

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_system_prompt(mode):
    if mode == "detailed":
        return "You are a helpful assistant. Respond with detailed information. Cover all relevant aspects thoroughly."
    elif mode == "concise":
        return "You are a helpful assistant. Provide concise responses. Limit details to the most crucial points only."
    elif mode == "quality_enhancement":
        return "You are an expert assistant, delve deeply into the core of the topic, providing a richly detailed response that explores all its dimensions. Ensure each part of your explanation is directly relevant to the query, enriching the content with precise data, illustrative examples, and analytical insights that clarify complexities and extend understanding. Present this information in a logical, step-by-step manner, while also considering and highlighting diverse perspectives and solutions. Your response should be comprehensive, directly addressing every element of the question with accuracy and depth."
    elif mode == "default":
        return "You are a helpful assistant."
    elif mode == "copy_paste":
        return "You are a helpful assistant."
    else:
        raise ValueError(f"Invalid mode: {mode}")

def call_model(prompt, model="llama3", temperature=0.7, max_tokens=2048, top_p=1, port=8000, mode=None):
    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="token-abc123",
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": get_system_prompt(mode)}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response
    except Exception as e:
        print(f"Error for prompt: {prompt} - {str(e)}")
        return None

def process_json_items(json_data, max_workers, model_name, port, mode, max_new_token, input_dataset, model):
    new_json_data = []
    rerun_queue = deque()

    if input_dataset == "alpaca":
        prompts = [(item["instruction"], item["dataset"],  [item["instruction"]]) for item in json_data]
    else:
        prompts = [(item["prompt"], item["dataset"], [item["instruction"]] if "instruction2" not in item else [item["instruction"], item["instruction2"]]) for item in json_data]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(call_model, prompt[0], model, 0.7, max_new_token, 1, port, mode): prompt for i, prompt in enumerate(prompts)}
        for future in as_completed(future_to_item):
            prompt, dataset, instruction = future_to_item[future]
            result = future.result()
            if result is None:
                rerun_queue.append((prompt, dataset, instruction))
            else:
                output_message = result.choices[0].message.content
                output_word_count = len(output_message.split())

                print(f'Prompt: {prompt}\n\nOutput: {output_message}\n\nWord count: {output_word_count}\n')
                current_json_item = {
                    "instruction": instruction[0],
                    "output": output_message,
                    "output_word_count": output_word_count,
                    "generator": model_name,
                    "dataset": dataset,
                    "prompt": prompt
                }
                if len(instruction) > 1:
                    current_json_item["instruction2"] = instruction[1]
                
                new_json_data.append(current_json_item)

    # Retry failed items
    max_retries = 10
    retry_count = 0
    while rerun_queue and retry_count < max_retries:
        retry_count += 1
        print(f"Retrying failed requests: Attempt {retry_count}")
        for prompt in list(rerun_queue):
            result = call_model(prompt[0], model=model, port=port, mode=mode, max_tokens=max_new_token)
            if result is None:
                print(f"Retry failed for prompt: {prompt}")
            else:
                output_message = result.choices[0].message.content
                output_word_count = len(output_message.split())

                rerun_queue.remove(prompt)
                current_json_item = {
                    "instruction": prompt[2][0],  # Assume prompt structure is (prompt_text, dataset, instruction)
                    "output": output_message,
                    "output_word_count": output_word_count,
                    "generator": model_name,
                    "dataset": prompt[1],
                    "prompt": prompt[0],
                }
                if len(prompt[2]) > 1:
                    current_json_item["instruction2"] = prompt[2][1]
                new_json_data.append(current_json_item)

    return new_json_data

def save_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Host VLLM model, process JSON data with different modes, and save to JSON files.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model to host')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use for generation')
    args = parser.parse_args()

    # Start the vllm OpenAI API server
    port = 8000
    command = [
        'python', '-m', 'vllm.entrypoints.openai.api_server',
        f'--model={args.model_path}',
        f'--port={port}',
        '--trust-remote-code'
    ]
    process = subprocess.Popen(command)

    try:
        # Wait for the server to start
        server_url = f"http://localhost:{port}/v1"
        print(f"Waiting for the server to start at {server_url}...")
        wait_for_server(server_url)

        # Load reference JSON
        input_json_path = 'reference/evaluation_reference/alpaca_gpt4_reference.json'
        json_data = load_json(input_json_path)

        # Define the modes to use
        modes = ['detailed', 'concise', 'quality_enhancement']

        # Generate output for each mode
        for mode in modes:
            print(f"Processing mode: {mode}")
            processed_data = process_json_items(json_data, max_workers=30, model_name=args.model_name, port=port, mode=mode, max_new_token=2048, input_dataset="alpaca", model="gpt-3.5-turbo-0125")
            output_json_path = f"{args.model_name}_{mode}.json"
            save_json(output_json_path, processed_data)
            print(f"Processed data saved to {output_json_path}")

    finally:
        # Kill the vllm server process
        process.terminate()
        process.wait()

if __name__ == "__main__":
    main()
