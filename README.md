# Understanding Win Rate for Better LLM-based Preference Evaluation (Adaptive AlpacaEval) Framework

The Adaptive AlpacaEval framework is designed to automate the generation and evaluation of datasets using different configurations. This repository includes scripts for generating data, hosting models, and evaluating the results comprehensively. We continuously refine our approach based on community feedback to enhance the open-source tools we offer.

## Features

- **Data Generation:** Scripts to generate datasets using multiple configurations and language models.
- **Model Hosting:** Instructions and scripts for hosting various language models for easy API access.
- **Evaluation:** Comprehensive evaluation tools to assess the performance of language models on generated datasets.
- **Scalability:** Designed to handle multiple datasets and language model configurations efficiently.

## News

- [Date] New features or updates about the framework will be announced here.

## Setup

Install all required dependencies to ensure all scripts function correctly.

```bash
pip install -r requirements.txt
```

### Configuration

Before running any scripts, update the API keys and model details in the provided shell scripts as per your setup. Ensure the input paths and other parameters are correctly set according to your environment.

## Data Generation

Data generation scripts are organized into separate folders based on their specific tasks:

### Generating Dataset Variants

Run the following command to generate different dataset variants. This script handles multiple datasets and applies specified generation modes.

```bash
bash generate_all_adap_alpaca.sh
```

### Hosting Models

For hosting models locally, use the provided script which sets up a server for a specified model:

```bash
bash host_vllm_server.sh
```

## Evaluation

Evaluation scripts are included to assess the quality of the generated datasets and model responses. These scripts compare outputs against a reference set and provide detailed metrics.

```bash
bash generate_different_dataset_and_eval_gpt.sh
```

## Detailed Script Usage

Here we explain how to utilize each script within the AdapAlpaca framework.

### `generate_all_adap_alpaca.sh`

This script generates datasets with varied word count limits. Each dataset is tailored to specific model configurations and generation modes.

**Usage:**

```bash
bash generate_all_adap_alpaca.sh
```

**What it does:**
- Creates an output directory for the generated datasets.
- Executes a Python script multiple times with different configurations to cover a range of word count limits from the template JSON.
- Outputs are stored in `adapAlpaca_output` with filenames indicating the word count range.

### `host_vllm_server.sh`

Hosts a specified language model on a local server, allowing API access to the model functionalities.

**Usage:**

```bash
bash host_vllm_server.sh
```

**Details:**
- Sets up a server for the LLaMA model or any specified model compatible with the VLLM serving guidelines.
- Configures the server to run in the background, logging its output for monitoring purposes.

### `generate_different_dataset_and_eval_gpt.sh`

Generates datasets using a specified GPT model and evaluates them.

**Usage:**

```bash
bash generate_different_dataset_and_eval_gpt.sh
```

**Steps:**
1. **Dataset Generation:** Generates datasets for different configurations (e.g., `koala`, `vicuna`).
2. **Evaluation Preparation:** Prepares folders and configurations for evaluation.
3. **Evaluation Execution:** Runs evaluation scripts to assess the dataset quality and model performance, outputting detailed metrics and logs.


### `generate_different_dataset_and_eval_vllm.sh`

**NOTE: Before running this script you need to setup vllm and host a openai serve -> check host_vllm_server.sh**

Generates datasets using a specified model hosting through vllm and evaluates them.

**Usage:**

```bash
bash generate_different_dataset_and_eval_vllm.sh
```

**Steps:**
1. **Dataset Generation:** Generates datasets for different configurations (e.g., `koala`, `vicuna`).
2. **Evaluation Preparation:** Prepares folders and configurations for evaluation.
3. **Evaluation Execution:** Runs evaluation scripts to assess the dataset quality and model performance, outputting detailed metrics and logs.

### `generate_different_dataset_gpt.sh`

Generates datasets based on the GPT model for different instructions set templates.

**Usage:**

```bash
bash generate_different_dataset_gpt.sh
```

**Functionality:**
- Iterates through various dataset templates.
- Applies the specified GPT model to generate outputs, which are saved in a designated output folder.

### `generate_different_dataset_vllm.sh`

**NOTE: Before running this script you need to setup vllm and host a openai serve -> check host_vllm_server.sh**

Generates datasets using the hosted VLLM models, specified by the user.

**Usage:**

```bash
bash generate_different_dataset_vllm.sh
```

**Operation:**
- Similar to the GPT script but tailored for VLLM models.
- Ensures datasets are compatible with the VLLM model outputs and specifications.



## Citation

If you find the AdapAlpaca framework useful in your research, please consider citing:

```bibtex
@misc{adapalpaca2024,
  title={Understanding Win Rate for Better LLM-based Preference Evaluation},
  author={xxx, xxx},
  year={2024},
  note={Provided scripts and tools for dataset generation and model evaluation}
}
```

## Contributions

We welcome contributions and suggestions from the community. Please feel free to fork the repository, make changes, and submit pull requests. Your insights are valuable to us!

