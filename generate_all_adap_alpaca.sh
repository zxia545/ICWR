#!/bin/bash

# Update these API keys with your actual keys
API_KEYS="sk-yourfirstkey,sk-yoursecondkey,sk-yourthirdkey,..."

# Number of processes to run in parallel
NUM_PROCS=5


# Directory where output JSON files will be stored
mkdir -p adapAlpaca_output
INPUT_JSON="./reference/generation_template/alpaca_template.json"

# Generate output files with different word count limits
python3 ./generation/generate_adap_alpaca.py --input_json $INPUT_JSON --output_json "adapAlpaca_output/adapAlpaca-200.json" --api_keys $API_KEYS --min_word 0 --max_word 200 --num_procs $NUM_PROCS
python3 ./generation/generate_adap_alpaca.py --input_json $INPUT_JSON --output_json "adapAlpaca_output/adapAlpaca-400.json" --api_keys $API_KEYS --min_word 200 --max_word 400 --num_procs $NUM_PROCS
python3 ./generation/generate_adap_alpaca.py --input_json $INPUT_JSON --output_json "adapAlpaca_output/adapAlpaca-600.json" --api_keys $API_KEYS --min_word 400 --max_word 600 --num_procs $NUM_PROCS
python3 ./generation/generate_adap_alpaca.py --input_json $INPUT_JSON --output_json "adapAlpaca_output/adapAlpaca-800.json" --api_keys $API_KEYS --min_word 600 --max_word 800 --num_procs $NUM_PROCS
python3 ./generation/generate_adap_alpaca.py --input_json $INPUT_JSON --output_json "adapAlpaca_output/adapAlpaca-1000.json" --api_keys $API_KEYS --min_word 800 --max_word 1000 --num_procs $NUM_PROCS

echo "Data generation complete."
