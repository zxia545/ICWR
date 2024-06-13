#!/bin/bash

SCRIPT_DIR=$(pwd)
# -----------------------------------------CUSTOM_INPUT---------------------------------------- #
# --------------------------------------------------------------------------------------------- #

# Update these API keys with your actual keys
API_KEYS="sk-yourfirstkey,sk-yoursecondkey,sk-yourthirdkey,..."

# Enter the openai model name
GEN_MODEL="gpt-3.5-turbo-0125"
# Generation MODE
# NOTE available modes: "default", "detailed", "concise", "quality_enhancement", "copy_paste"
MODE="copy_paste"

# Max workers
MAX_WORKERS=10
# Output folder
OUTPUT_FOLDER=${SCRIPT_DIR}/"generated_data"
# Output evaluation folder
OUTPUT_FOLDER_EVAL=${SCRIPT_DIR}/"generated_data_eval"


# -----------------------------------------PART1----------------------------------------------- #
# --------------------------------------------------------------------------------------------- #
# Generate different dataset with different model and mode and dataset
# generate output folder
mkdir -p $OUTPUT_FOLDER

 
for dataset_name in 'koala' 'vicuna' 'sinstruct' 'wizardlm' 'lima' 'alpaca'
do
INPUT_JSON=./reference/generation_template/${dataset_name}_template.json
python ./generation/generate_different_dataset_gpt.py \
    --input_json_path $INPUT_JSON \
    --output_json_path ${OUTPUT_FOLDER}/${dataset_name}_${GEN_MODEL}_${MODE}.json \
    --input_dataset ${dataset_name} \
    --api_keys $API_KEYS \
    --max_workers $MAX_WORKERS \
    --model_name ${GEN_MODEL} \
    --model ${GEN_MODEL} \
    --mode ${MODE}
done


echo "Data generation complete."

# -----------------------------------------PART2----------------------------------------------- #
# --------------------------------------------------------------------------------------------- #
# Generate evaluation folder and data
cd $SCRIPT_DIR
mkdir -p $OUTPUT_FOLDER_EVAL

cd $SCRIPT_DIR/help_method

python generate_alpaca_eval_folder.py \
    --input_folder ${OUTPUT_FOLDER} \
    --output_folder ${OUTPUT_FOLDER_EVAL}

echo "Data evaluation folder generation complete."

# -----------------------------------------PART3----------------------------------------------- #
# --------------------------------------------------------------------------------------------- #


# Define the base directory
base_dir=${OUTPUT_FOLDER_EVAL}
# Define the reference JSON file name
REFERENCE_JSON_NAME="gpt4_reference_output"

# Navigate to the base directory
cd $base_dir

# Find all subdirectories
for dir in */ ; do
    # Enter each directory
    cd $dir

    # Check for json files that are not named 'gpt4_ori.json'
    for json_file in *.json; do
        # Assume that the reference file is named 'gpt4_ori.json'
        if [[ $json_file != ${REFERENCE_JSON_NAME}.json ]]; then
            # Extract the file name without the '.json' extension
            json_name="${json_file%.json}"

            # Construct the subfolder from the directory name (removing trailing slash)
            subfolder_name="${dir%/}"

            # Run the Python script with necessary parameters
            python ${SCRIPT_DIR}/evaluation/run_alpaca_eval.py \
            --annotators_config=weighted_alpaca_eval_gpt4_turbo \
            --base_folder=$base_dir \
            --subfolder_name=$subfolder_name \
            --model_input_name=$json_name \
            --reference_name=${REFERENCE_JSON_NAME} \
            --keys=$API_KEYS
        fi
    done
    # Go back to the base directory
    cd ..
done

echo "Evaluation complete. For more detail check $OUTPUT_FOLDER_EVAL subfolder."
echo ""
echo "And for the detail win-rate and other metrics please reference to the csv file inside the nested folder: weighted_alpaca_eval_gpt4_turbo_result/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv."
