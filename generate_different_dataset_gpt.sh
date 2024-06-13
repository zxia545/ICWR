#!/bin/bash

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
OUTPUT_FOLDER="generated_data"

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