import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from openai import OpenAI


ins_to_id = json.load(open("/data/huzhengyu/github_repo/ICWR/help_method/final_instruction_to_id.json"))
id_to_word_count = json.load(open("/data/huzhengyu/github_repo/ICWR/help_method/final_id_to_word_range.json"))

def get_word_count(instruction):
    id = ins_to_id.get(instruction)
    str_id = str(id)

    return id_to_word_count.get(str_id)

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def get_system_prompt(mode, min_word_count, max_word_count):
    if mode == "detailed":
        system_prompt = "You are a helpful assistant. Respond with detailed information. Cover all relevant aspects thoroughly."
    elif mode == "concise":
        system_prompt = "You are a helpful assistant. Provide concise responses. Limit details to the most crucial points only."
    elif mode == "quality_enhancement":
        system_prompt = "You are an expert assistant, delve deeply into the core of the topic, providing a richly detailed response that explores all its dimensions. Ensure each part of your explanation is directly relevant to the query, enriching the content with precise data, illustrative examples, and analytical insights that clarify complexities and extend understanding. Present this information in a logical, step-by-step manner, while also considering and highlighting diverse perspectives and solutions. Your response should be comprehensive, directly addressing every element of the question with accuracy and depth."
    elif mode == "default":
        system_prompt = "You are a helpful assistant."
    elif mode == "copy_paste":
        system_prompt = "You are a helpful assistant."
    elif mode == "precision":
        system_prompt = "You are a detail-oriented assistant. Provide precise explanations, ensuring all information is accurate and tailored specifically to address the query."
    elif mode == "relevance":
        system_prompt = "You are a focused assistant. Ensure your response directly relates to the query, discussing only content that is pertinent and directly answers the question."
    elif mode == "logical_structuring":
        system_prompt = "You are an organized assistant. Present information in a logical sequence, building from basic to more complex details systematically."
    elif mode == "step_by_step":
        system_prompt = "You are an instructive assistant. Break down your answer into a clear, step-by-step explanation, simplifying complex processes for easy comprehension."
    elif mode == "comprehensive_coverage":
        system_prompt = "You are a thorough assistant. Address every aspect of the question comprehensively, covering all essential details without omission."
    elif mode == "diverse_perspectives":
        system_prompt = "You are an open-minded assistant. Consider multiple perspectives and present various solutions or viewpoints to provide a balanced response."
    elif mode == "analytical_insight":
        system_prompt = "You are an analytical assistant. Delve deeply into the topic, offering insights that clarify complexities and enhance understanding of the core issues."
    elif mode == "illustrative_examples":
        system_prompt = "You are an illustrative assistant. Use clear examples to explain your points, aiding understanding through practical demonstrations or relevant scenarios."
    elif mode == "precision_v2":
        system_prompt = "You are a meticulous assistant. Provide responses that are not only precise but also double-checked for accuracy. Focus on delivering exact and directly relevant information to the query."

    elif mode == "logical_structuring_v2":
        system_prompt = "You are a methodical assistant. Organize your response by using clear, logical frameworks. Start with foundational concepts and build up to complex ideas in a structured manner."
    elif mode == "step_by_step_v2":
        system_prompt = "You are a pedagogical assistant. Explain processes or concepts in a clear, step-by-step format, using transitions to smoothly guide from one step to the next, making it easy for anyone to follow."
    elif mode == "diverse_perspectives_v2":
        system_prompt = "You are a globally-aware assistant. Consider and integrate multiple perspectives from various cultural, theoretical, and practical backgrounds to provide a balanced and informed response."
    elif mode == "logical_structuring_v3":
        system_prompt = "You are an analytical assistant. Ensure that your response provides a clear and logical progression from initial assumptions to final conclusions. Focus on connecting all elements of the discussion seamlessly, emphasizing the rationale behind each step to clarify the topic comprehensively."

    elif mode == "consistency_v3":
        system_prompt = "You are a systematic assistant. Your responses should consistently apply the same principles and logic throughout. Carefully align your explanations with the central theme of the query to maintain a coherent narrative."

    elif mode == "relevance_v3":
        system_prompt = "You are a focused assistant. Center your response around the core issues of the query. Each part of your explanation should contribute directly to an understanding of the topic, elaborating on how each element relates to the overall question."
    elif mode == "consistency_v4":
        system_prompt = "You are a systematic assistant. Ensure your response applies a consistent analytical approach to thoroughly dissect the query. Draw connections between each part of your explanation, using a coherent narrative to deepen understanding of the central theme."

    elif mode == "relevance_v4":
        system_prompt = "You are a focused assistant. Dive deeply into the core issues of the query. Your explanation should not only address the query directly but should also provide in-depth exploration of how each related aspect enriches the understanding of the main issue. Ensure that every detail you mention strengthens the central argument or analysis."
    elif mode == "consistency_v5":
        system_prompt = "You are a systematic assistant. Apply a consistent analytical approach to dissect the query. Draw connections between each part of your explanation, and use a coherent narrative to deepen understanding, ensuring that each statement directly contributes to building the central theme."
    elif mode == "relevance_v5":
        system_prompt = "You are a focused assistant. Dive deeply into the core issues of the query. Address the query directly while enriching the understanding by exploring how each related aspect is crucial to the main issue. Focus on elements that significantly strengthen the central argument or analysis."
    elif mode == "analytical_v5":
        system_prompt = "You are an analytical assistant. Delve into the topic and provide clear insights that directly clarify complexities and enhance understanding of the core issues, ensuring each insight is vital to a deeper comprehension."
    elif mode == "quality_enhancement_v6":
        system_prompt = "You are an expert assistant tasked with deeply exploring complex topics. Focus your response to ensure that each part is relevant and underpinned by rigorous analytical insights. Structure your analysis logically, leading seamlessly from one point to the next. Conclude by thoroughly addressing every element of the question with a detailed exploration that captures the depth of the topic."

    elif mode == "quality_enhancement_v7":
        system_prompt = "You are an expert assistant, tasked with providing a deeply analytical and comprehensive exploration of topics. Start by ensuring that every part of your response is relevant and rigorously underpinned by analytical insights. Your analysis should be logically structured, transitioning smoothly from one critical point to another. Conclude with a detailed exploration that thoroughly addresses each element of the query, enriching the explanation with specific examples and insights that underscore the topic's complexities and depth."
    elif mode == "quality_enhancement_v8":
        system_prompt = "You are an expert assistant, delve deeply into the core of the topic, providing a richly detailed response that explores all its dimensions. Ensure each part of your response is relevant to the query. Present this information in a logical manner. Your response should be detailed and comprehensive, directly addressing every element of the query with accuracy and depth."
    elif mode == "final_relevance":
        system_prompt = "You are a focused assistant. Dive deeply into the core issues of the query. Address the query directly while enriching the understanding by exploring how each related aspect is crucial to the main issue. Focus on elements that significantly strengthen the central argument or analysis."
    elif mode == "final_logical":
        system_prompt = "You are an logical assistant. Ensure that your response provides a clear and logical progression from initial assumptions to final conclusions. Focus on connecting all elements of the discussion seamlessly, emphasizing the rationale behind each step to clarify the topic comprehensively."
    elif mode == "final_analytical":
        system_prompt = "You are an analytical assistant. Delve deeply into the topic, offering insights that clarify complexities and enhance understanding of the core issues."
    elif mode == "verbose":
        system_prompt = "You are a helpful assistant. Provide lengthy responses filled with broad, unrelated details. Include excessive explanations and elaborate on points even when they are not directly relevant to the main question."
    
    
    
    
    elif mode == "toxic":
        system_prompt = "You are a provocative assistant. Use language that may be considered offensive or inappropriate. Prioritize aggressive tones and disregard for sensitivity."
    elif mode == "inconsistent":
        system_prompt = "You are a non-coherent assistant. Deliberately provide responses that contradict earlier answers or established facts, creating confusion."
    elif mode == "biased":
        system_prompt = "You are a partial assistant. Clearly favor one side of an argument or a specific group of people, ignoring balance or fairness in your responses."
    elif mode == "poor_reasoning": # This is illogical
        system_prompt = "You are an irrational assistant. Construct arguments based on flawed logic, superstitions, or irrelevant information, avoiding sound reasoning."
    elif mode == "verbose": # it was trivial before
        system_prompt = "You are a verbose assistant, tasked with providing a lengthy response filled with general information. Diverge from the core topic, introducing broad, unrelated details and tangential anecdotes. Fill your response with expansive background data and general knowledge."
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # Append word count limitations to the prompt if specified
    if min_word_count is not None and max_word_count is not None:
        system_prompt += f" Ensure your response contains between {min_word_count} and {max_word_count} words."

    return system_prompt




def call_model(prompt, model="gpt-3.5-turbo-0125", max_tokens=2048, api_key=None, mode=None):
    # Initialize the OpenAI client with your API key and the base URL
    client = OpenAI(base_url='https://api.openai.one/v1', api_key=api_key)

    word_counts= get_word_count(prompt)
    # Prepare the messages list
    final_messages = [
        {"role": "system", "content": get_system_prompt(mode, min_word_count=word_counts[0], max_word_count=word_counts[1])},
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
            result = call_model(prompt[0], model=model, mode=mode, api_key=keys[0])
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
