import argparse
import sys
import os
import subprocess
from this_utils import start_vllm_server, stop_vllm_server

def main():
    parser = argparse.ArgumentParser(description='Run vLLM server and all generation tasks')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--gpu', type=int, default=1, help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    try:
        # Start the vLLM server
        print(f"Starting vLLM server for model {args.model_name}...")
        process = start_vllm_server(
            model_path=args.model_path,
            model_name=args.model_name,
            port=args.port,
            gpu=args.gpu
        )
        
        print(f"Server is running on port {args.port}")
        print("Starting generation tasks...")
        
        # Create a copy of the current environment
        env = os.environ.copy()
        
        # Set environment variables for the generation script
        env['VLLM_HOST_MODEL_NAME'] = args.model_name
        env['VLLM_PORT_NUMBER'] = str(args.port)
        
        # Run the generation script with the updated environment
        subprocess.run(['bash', 'generate_different_dataset_vllm.sh'], 
                      env=env,
                      check=True)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Stop the vLLM server
        if 'process' in locals():
            print("\nStopping vLLM server...")
            stop_vllm_server(process)

if __name__ == "__main__":
    main() 