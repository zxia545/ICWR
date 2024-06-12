# Reference to https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html
SERVED_MODEL="llama3"
MODEL_LOCAL_PATH_OR_HUGGINGFACE_NAME="meta-llama/Meta-Llama-3-8B"
PARALLEL_NUMBER=2
PORT_NUMBER=8010



python -m vllm.entrypoints.openai.api_server \
--model=${MODEL_LOCAL_PATH_OR_HUGGINGFACE_NAME} \
--served-model-name=${SERVED_MODEL} \
--tensor-parallel-size=${PARALLEL_NUMBER} \
--port=${PORT_NUMBER} > ${SERVED_MODEL}-server.log &