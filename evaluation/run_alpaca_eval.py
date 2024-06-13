import argparse
import subprocess
import os


def run_evaluation(annotators_config, subfolder_name, base_path, reference_name, model_input_name, key_list):
    if base_path == "":
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.realpath(__file__))
    else:
        script_dir = base_path

    log_folder = os.path.join(script_dir, 'logs')

    model_outputs_path = os.path.join(script_dir, subfolder_name, f'{model_input_name}.json')
    reference_outputs_path = os.path.join(script_dir, subfolder_name, f'{reference_name}.json')
    output_path = os.path.join(script_dir, subfolder_name, f"{annotators_config}_result")
    log_file_path = os.path.join(script_dir, 'logs', f'{subfolder_name}_eval_log_{annotators_config}.txt')

    os.makedirs(output_path, exist_ok=True)  # Ensure the output directory exists
    os.makedirs(log_folder, exist_ok=True)  # Ensure the output directory exists

    # Set environment variables and run command
    env = os.environ.copy()
    env['OPENAI_MAX_CONCURRENCY'] = f'{len(key_list) * 5}'
    env['OPENAI_API_BASE'] = 'https://api.openai.com/v1'
    env['OPENAI_API_KEYS'] = ','.join(key_list)  # Simplified keys slicing

    command = (
        f'alpaca_eval evaluate '
        f'--model_outputs "{model_outputs_path}" '
        f'--reference_outputs "{reference_outputs_path}" '
        f'--annotators_config {annotators_config} '
        f'--is_store_missing_annotations={True} '
        f'--output_path="{output_path}"'
    )

    with open(log_file_path, 'a') as log_file:
        # Execute the command
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)

        for line in iter(process.stdout.readline, b''):
            decoded_line = line.decode('utf-8')
            print(decoded_line, end='')  # Optional: also print to stdout
            log_file.write(decoded_line)
            log_file.flush()

        # Wait for the command to complete
        process.wait()
        if process.returncode == 0:
            print(f"Command executed successfully. Output saved to {output_path}")
        else:
            print(f"Error in command execution")

def main():
    parser = argparse.ArgumentParser(description='Run evaluation with specified configuration and model.')

    parser.add_argument('--annotators_config', type=str, required=True, default='weighted_alpaca_eval_gpt4_turbo', help='Annotator configuration to use')
    parser.add_argument('--subfolder_name', type=str, required=True, help='Subfolder/model name to use for inputs/outputs')
    parser.add_argument('--base_folder', type=str, default="", required=True)
    parser.add_argument('--reference_name', type=str, default="gpt4_reference_output", required=True, help='Reference file name')
    parser.add_argument('--model_input_name', type=str, required=True, help='Json file for evaluation')
    parser.add_argument('--keys', type=str, required=True, help='Comma-separated list of OpenAI API keys')

    args = parser.parse_args()

    run_evaluation(args.annotators_config, args.subfolder_name, args.base_folder, args.reference_name, args.model_input_name, args.keys.split(','))

if __name__ == "__main__":
    main()
