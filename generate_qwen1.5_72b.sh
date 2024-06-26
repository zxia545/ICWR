
#!/bin/bash

# Enter the vllm hosted model name
VLLM_HOST_MODEL_NAME='qwen1.5_72b'
VLLM_PORT_NUMBER=8051
# Generation MODE
# NOTE available modes: "default", "detailed", "concise", "quality_enhancement", "copy_paste"
# Max workers
MAX_WORKERS=20
# Output folder
OUTPUT_FOLDER='generated_data_qwen1.5_72b'

# generate output folder
mkdir -p $OUTPUT_FOLDER

#'koala' 'vicuna' 'sinstruct' 'wizardlm' 'lima' 

for dataset_name in 'alpaca' 'koala' 'vicuna' 'sinstruct' 'wizardlm' 'lima'; do
    for MODE in 'default' 'detailed' 'concise'; do
        INPUT_JSON=./reference/generation_template/${dataset_name}_template.json
        python ./generation/generate_different_dataset_vllm.py \
            --input_json_path $INPUT_JSON \
            --output_json_path ${OUTPUT_FOLDER}/${dataset_name}_${VLLM_HOST_MODEL_NAME}_${MODE}.json \
            --input_dataset ${dataset_name} \
            --max_workers $MAX_WORKERS \
            --model_name ${VLLM_HOST_MODEL_NAME} \
            --model ${VLLM_HOST_MODEL_NAME} \
            --mode ${MODE} \
            --port ${VLLM_PORT_NUMBER}
    done
done