import json
import os
import shutil
import argparse

# NOTE: We expect the filename to be in the format of "DatasetName_*.json"

ref_dict = {
    "alpaca" : "../reference/evaluation_reference/alpaca_gpt4_reference.json",
    "koala" : "../reference/evaluation_reference/koala_gpt4_reference.json",
    "lima" : "../reference/evaluation_reference/lima_gpt4_reference.json",
    "sinstruct" : "../reference/evaluation_reference/sinstruct_gpt4_reference.json",
    "vicuna" : "../reference/evaluation_reference/vicuna_gpt4_reference.json",
    "wizardlm" : "../reference/evaluation_reference/wizardlm_gpt4_reference.json",
}


def main():
    parser = argparse.ArgumentParser(description='Process output files for evaluation.')

    parser.add_argument('--input_folder', type=str, required=True,  help='Input folder containing JSON files to process.')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder to store the processed files.')

    args = parser.parse_args()
    
    input_folder = args.input_folder
    output_folder = args.output_folder

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Read all JSON files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            # Split the filename to get the key for ref_dict
            dataset_name = filename.split('_')[0]
            if dataset_name in ref_dict:
                # Create a new directory for this file pair
                filename_without_json = filename.split('.json')[0]
                new_dir = os.path.join(output_folder, f"{filename_without_json}-vs-gpt4")
                os.makedirs(new_dir, exist_ok=True)

                # Path of the current JSON file
                current_file_path = os.path.join(input_folder, filename)

                # Copy the reference file and current file to the new directory
                shutil.copy(ref_dict[dataset_name], os.path.join(new_dir, f"gpt4_reference_output.json"))
                shutil.copy(current_file_path, new_dir)
            else:
                print(f'Cannot find valid key for: {filename}')

    print("Files have been processed and copied.")
if __name__ == "__main__":
    main()

