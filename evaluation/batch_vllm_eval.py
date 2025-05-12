import os
import argparse
import glob
from run_vllm_eval import generate_output


def main():
    parser = argparse.ArgumentParser(description='Batch evaluation for multiple input files.')
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing input JSON files')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save output JSON files')
    parser.add_argument('--reference_folder', type=str, required=True, help='Folder containing reference JSON files')
    parser.add_argument('--model', type=str, default='llama3', help='Model name')
    parser.add_argument('--mode', type=str, default='random', help='Evaluation mode')
    parser.add_argument('--gpu', type=int, default=4, help='Number of GPUs')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    input_files = glob.glob(os.path.join(args.input_folder, '*.json'))
    reference_files = {os.path.basename(f): f for f in glob.glob(os.path.join(args.reference_folder, '*_gpt4_reference.json'))}

    from vllm import LLM, SamplingParams
    sampling_params = SamplingParams(max_tokens=20)
    llm = LLM(model=args.model, gpu_memory_utilization=0.8, tensor_parallel_size=args.gpu)

    for input_file in input_files:
        base = os.path.basename(input_file)
        if '_' not in base:
            print(f"Skipping {base}: cannot extract dataset name.")
            continue
        dataset = base.split('_')[0]
        ref_name = f"{dataset}_gpt4_reference.json"
        ref_path = reference_files.get(ref_name)
        if not ref_path:
            print(f"No reference found for {base} (expected {ref_name}), skipping.")
            continue
        output_file = os.path.join(args.output_folder, base.replace('.json', '_eval_result.json'))
        print(f"Evaluating {base} with reference {ref_name} -> {os.path.basename(output_file)}")
        generate_output(input_file, ref_path, output_file, llm, sampling_params, args.mode)

if __name__ == "__main__":
    main() 