from openai import OpenAI
import multiprocessing
import tqdm
import json
import argparse
import os
import random
from vllm import LLM, SamplingParams
    

def replace_user_prompt(instruction, output_1, output_2, switch):
    template = """
I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{
    "instruction": "{instruction}",
}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{
    {
        "model_identifier": "model1",
        "output": "{output_1}"
    },
    {
        "model_identifier": "model2",
        "output": "{output_2}"
    }
}

## Task

Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): model1 or model2.

## Best Model Identifier
"""
    template = template.replace("{instruction}", instruction)
    if switch:
        template = template.replace("{output_1}", output_2)
        template = template.replace("{output_2}", output_1)
    else:
        template = template.replace("{output_1}", output_1)
        template = template.replace("{output_2}", output_2)
        
    system = "You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers."
    new_template = system + '\n' + template
    return new_template


def extract_model_identifier(output, switch):
    import re
    change_back = {
        "model1": "model2",
        "model2": "model1"
    }

    # Make sure we always check at least the last part of the output; slice more if output is shorter than 50 chars
    output_segment = output[-50:].lower() if len(output) > 50 else output.lower()

    output_segment_front = output[:50].lower() if len(output) > 50 else output.lower()

    # This regex will match 'model1' or 'model2'
    pattern = r'\b(model1|model2)\b'
    matches = re.findall(pattern, output_segment)

    # Check which model appears last in the segment (which means first from the back to front)
    if matches:
        last_match = matches[-1]
    else:
        # If no match was found in the end segment, check the first 50 characters
        output_segment_front = output[:50].lower() if len(output) > 50 else output.lower()
        matches = re.findall(pattern, output_segment_front)
        if matches:
            last_match = matches[0]  # Get the first match found from the front
        else:
            # No valid identifier was found in either segment
            return False, None

    # Apply switch if required
    if switch:
        last_match = change_back[last_match]

    return True, last_match



def get_eval_result(args):
    output_1_m, output_2_M, output_1_model_name, output_2_model_name, formatted_prompt, instruction, dataset, port, model, switch= args
    
    for i in range(50):
        try:
            client = OpenAI(
                base_url=f"http://localhost:{port}/v1",
                api_key="token-abc123",
            )
            system_msg = "You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers."
            completion = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "user", "content": system_msg + '\n' + formatted_prompt}
                            ],
                            max_tokens=10
                        )

            output_text = completion.choices[0].message.content

            print(output_text)
            
            success, result = extract_model_identifier(output_text, switch)
            
            if success:
                print(f'Got a valid result: {result} at iteration {i}')
                break
        except Exception as e:
            print(f'Got an error message:{e}')
            continue
    else:
        print('Failed to get a valid result after 10 iterations')
        result = None

    return result, output_1_m, output_2_M,output_1_model_name, output_2_model_name, instruction, dataset


def process_tasks(json_data, ref_data, llm, sampling_params, mode):
    # Prepare arguments for each process
    prompt_to_args = {}
    prompt_list = []
    for json_item in json_data:
        output_1_m = json_item["output"]
        instruction = json_item["prompt"]
        dataset = json_item["dataset"]
        for ref_item in ref_data:
            if ref_item["prompt"] == instruction and ref_item["dataset"] == dataset:
                output_2_M = ref_item["output"]
                output_1_model_name = "output1"
                output_2_model_name = "output2"
                break
        
        if mode == "random":
            switch = random.choice([True, False])
        elif mode == "output_first":
            switch = False
        elif mode == "reference_first":
            switch = True

        formatted_prompt = replace_user_prompt(instruction, output_1_m, output_2_M, switch)
        prompt_list.append(formatted_prompt)
        prompt_to_args[formatted_prompt] = (output_1_m, output_2_M, output_1_model_name, output_2_model_name, instruction, dataset, switch)

    
    new_json_data = []
    
    
    outputs = llm.generate(prompt_list, sampling_params)

    rerun_prompts = []
    rerun_args = {}
    for output in outputs:
        prompt = output.prompt
        output_1_m, output_2_M, output_1_model_name, output_2_model_name, instruction, dataset, switch = prompt_to_args[prompt]
        
        output_text = output.outputs[0].text

        
        
        success, result = extract_model_identifier(output_text, switch)

        
        if not success:
            rerun_prompts.append(prompt)
            rerun_args[prompt] = (output_1_m, output_2_M, output_1_model_name, output_2_model_name, instruction, dataset, switch)
            print(f'Failed result: {output_text}')
        else:
            current_json_item = {
                "instruction": instruction,
                "dataset": dataset,
                "output_1": output_1_m,
                "output_2": output_2_M,
                "generator_1": output_1_model_name,
                "generator_2": output_2_model_name,
                "result": result
            }
            new_json_data.append(current_json_item)

    for i in range(50):
        if len(rerun_prompts) == 0:
            break
        
        outputs = llm.generate(rerun_prompts, sampling_params)

        for output in outputs:
            prompt = output.prompt
            output_1_m, output_2_M, output_1_model_name, output_2_model_name, instruction, dataset, switch = rerun_args[prompt]
            
            output_text = output.outputs[0].text

            success, result = extract_model_identifier(output_text, switch)

            if success:
                current_json_item = {
                    "instruction": instruction,
                    "dataset": dataset,
                    "output_1": output_1_m,
                    "output_2": output_2_M,
                    "generator_1": output_1_model_name,
                    "generator_2": output_2_model_name,
                    "result": result
                }
                new_json_data.append(current_json_item)
                rerun_prompts.remove(prompt)
                
            else:
                continue
    
        
    return new_json_data

def safe_open_file(output_json_path, updated_json_data):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # Now safe to write the file
    with open(output_json_path, 'w') as outfile:
        json.dump(updated_json_data, outfile, indent=4)
        
        
def generate_output(input_json_path, reference_json_path, output_json_path,llm, sampling_params,mode):

    with open(input_json_path) as f:
        json_data = json.load(f)
        
    with open(reference_json_path) as f:
        ref_json_data = json.load(f)
    

    updated_json_data = process_tasks(json_data, ref_json_data, llm, sampling_params, mode)
    
    safe_open_file(output_json_path=output_json_path, updated_json_data=updated_json_data)

def main():
    parser = argparse.ArgumentParser(description='Run evaluation with specified configuration and model.')
    parser.add_argument('--input_json', type=str, required=True, help='input json file')
    parser.add_argument('--ref_json', type=str, required=True, help='reference json file')
    parser.add_argument('--output_json', type=str, required=True, help='output json file')
    parser.add_argument('--model', type=str, default='llama3', help='model name')
    parser.add_argument('--mode', type=str, required=True, default="random", help='mode')
    parser.add_argument('--gpu', type=int,default=4, help='output port file')

    args = parser.parse_args()
    
    input_json_list = args.input_json.split(',')
    ref_json_list = args.ref_json.split(',')
    output_json_list = args.output_json.split(',')
    
    sampling_params = SamplingParams(max_tokens=20)


    llm = LLM(model=args.model,gpu_memory_utilization=0.8, tensor_parallel_size=args.gpu)
    for input_json_path, ref_json_path, output_json_path in zip(input_json_list, ref_json_list, output_json_list):
        generate_output(input_json_path, ref_json_path, output_json_path, llm, sampling_params, args.mode)

    
if __name__ == "__main__":
    main()