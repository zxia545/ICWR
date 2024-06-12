import argparse
import json
import multiprocessing
import tqdm
from openai import OpenAI

def get_user_prompt(min_word, max_word, prompt):
    return f"Respond to the following question, your reply must only be within {max_word}-{min_word} words.\n\n{prompt}"

def gpt4_get_output_with_length_requirement(args):
    prompt, key, max_word, min_word, dataset = args
    client = OpenAI(base_url='https://api.openai.com/v1', api_key=key)
    user_prompt = get_user_prompt(min_word, max_word, prompt)

    final_messages = [
        {"role": "system", "content": "You are a helpful assistant, highly attentive to the specified token range required from user."},
        {"role": "user", "content": user_prompt}
    ]

    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=final_messages,
    )

    output = completion.choices[0].message
    print(f'Prompt: {prompt}\nOutput: {output.content}\nWord count: {len(output.content.split())}\n')
    return prompt, output.content, len(output.content.split()), dataset

def process_tasks(json_data, api_keys, max_word, min_word):
    num_procs = min(len(api_keys), multiprocessing.cpu_count())
    process_args = [(json_item["instruction"], api_keys[i % len(api_keys)], max_word, min_word, json_item["dataset"]) for i, json_item in enumerate(json_data)]

    with multiprocessing.Pool(num_procs) as p:
        results = list(tqdm.tqdm(p.imap(gpt4_get_output_with_length_requirement, process_args), total=len(process_args), desc="Processing summarizations"))

    return [{
        "instruction": prompt,
        "output": output_message,
        "output_word_count": output_word_count,
        "generator": f"gpt4_generate_diff_length_from_{min_word}_to_{max_word}",
        "dataset": dataset
    } for prompt, output_message, output_word_count, dataset in results]

def generate_output(input_json_path, output_json_path, api_keys, max_word, min_word):
    with open(input_json_path) as f:
        json_data = json.load(f)
    updated_json_data = process_tasks(json_data, api_keys.split(','), max_word, min_word)
    with open(output_json_path, 'w') as outfile:
        json.dump(updated_json_data, outfile, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Generate data for NIPS dataset track with specified parameters.")
    parser.add_argument("--input_json", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output_json", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--api_keys", type=str, required=True, help="Comma-separated list of OpenAI API keys")
    parser.add_argument("--max_word", type=int, default=400, help="Maximum word count for the output")
    parser.add_argument("--min_word", type=int, default=200, help="Minimum word count for the output")
    parser.add_argument("--num_procs", type=int, default=10, help="Number of processes to use")

    args = parser.parse_args()
    generate_output(args.input_json, args.output_json, args.api_keys, args.max_word, args.min_word)

if __name__ == "__main__":
    main()
