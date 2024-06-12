import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from openai import OpenAI


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


def call_model(prompt, model="gpt-3.5-turbo-0125", max_tokens=2048, api_key=None, mode=None):
    # Initialize the OpenAI client with your API key and the base URL
    client = OpenAI(base_url='https://api.openai.com/v1', api_key=api_key)

    # Prepare the messages list
    final_messages = [
        {"role": "system", "content": get_system_prompt(mode)},
        {"role": "user", "content": prompt}
    ]
    
    # Use the client to create a completion
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=final_messages,
            max_tokens=max_tokens
        )
        return completion
    except Exception as e:
        print(f"Error for prompt: {prompt} - {str(e)}")
        return None


def process_json_items(json_data, keys, max_workers, model_name, model, mode, input_dataset):
    new_json_data = []
    rerun_queue = deque()

    if input_dataset == "alpaca":
        prompts = [(item["instruction"], item["dataset"],  [item["instruction"]]) for item in json_data]
    else:
        prompts = [(item["prompt"], item["dataset"], [item["instruction"]] if "instruction2" not in item else [item["instruction"], item["instruction2"]]) for item in json_data]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(call_model, prompt[0], model, 2048, keys[i % len(keys)], mode): prompt for i, prompt in enumerate(prompts)}
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
                    "prompt": prompt,
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
            result = call_model(prompt[0], mode=mode, api_key=keys[3])
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
    parser = argparse.ArgumentParser(description='Process JSON data with a language model.')
    parser.add_argument('--input_json_path', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output_json_path', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--api_keys', type=str, required=True, help='Comma-separated list of OpenAI API keys')
    parser.add_argument('--max_workers', type=int, default=30, help='Maximum word count for the response')
    parser.add_argument('--model_name', type=str, default="llama3_8b_original", help='Model name to use for generation')
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo-0125", help='Model to use for generation')
    parser.add_argument('--mode', type=str, default=None, help="Mode for the prompt generation")
    parser.add_argument('--input_dataset', type=str, default="alpaca", help='Dataset name for the output JSON file')
                       
    args = parser.parse_args()
    keys = args.api_keys.split(',')
    json_data = load_json(args.input_json_path)
    processed_data = process_json_items(json_data, keys, args.max_workers, args.model_name, args.model, args.mode, args.input_dataset)
    
    # check if mode is copy_paste then append the output 3 times
    if args.mode == "copy_paste":
        for item in processed_data:
            item["output"] = item["output"] + " " + item["output"] + " " + item["output"]
            item["output_word_count"] = len(item["output"].split())
            
    save_json(args.output_json_path, processed_data)
    print(f"Processed data saved to {args.output_json_path}")

if __name__ == "__main__":
    main()
