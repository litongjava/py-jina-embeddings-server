# py-jina-embeddings-server

Jina Embeddings v3 API Service

## Overview
This project provides an HTTP API service for generating embeddings from text using the Jina-embeddings-v3 model with ONNX Runtime and the Robyn framework.

## Requirements
- **Python:** 3.8+
- **Libraries:** numpy, onnxruntime, robyn, transformers

## Installation

1. Create a conda environment:
   ```bash
   conda create -n jina-embeddings python=3.9 -y
   conda activate jina-embeddings
   ```

2. Install the required libraries using pip:
   ```bash
   pip install numpy onnxruntime robyn transformers
   ```

## Usage

1. **Download the model:**
   ```bash
   mkdir ~/models/jina-embeddings-v3/onnx -p
   cd ~/models/jina-embeddings-v3/onnx
   wget https://huggingface.co/jinaai/jina-embeddings-v3/resolve/main/onnx/model.onnx
   wget https://huggingface.co/jinaai/jina-embeddings-v3/resolve/main/onnx/model.onnx_data
   ```

2. **Run the server:**
   Start the server by running the script:
   ```bash
   python server.py --model_path ~/models/jina-embeddings-v3/onnx/model.onnx
   ```

3. **Send a POST request to the `/v1/embeddings` endpoint:**
   Example:
   ```bash
   curl -X POST http://localhost:10002/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"input": ["Hello world", "你好，世界"], "task": "text-matching"}'
   ```
response example
```json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [
            ]
        }
    ],
    "model": "text-embedding-3-large",
    "usage": {
        "prompt_tokens": 5,
        "total_tokens": 5
    }
}
```
## Dockerfile

To containerize the service, use the following `Dockerfile`:

```Dockerfile
FROM python:3.9-slim

# Install wget (required to download the model)
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Download the model files into the specified directory
RUN mkdir -p /models/jina-embeddings-v3/onnx && \
    cd /models/jina-embeddings-v3/onnx && \
    wget https://huggingface.co/jinaai/jina-embeddings-v3/resolve/main/onnx/model.onnx && \
    wget https://huggingface.co/jinaai/jina-embeddings-v3/resolve/main/onnx/model.onnx_data

WORKDIR /app
# Copy application code into the container
COPY . /app
# Install required Python packages
RUN pip install --no-cache-dir numpy onnxruntime robyn transformers

# Set the default command with the model path specified
CMD ["python", "server.py", "--model_path", "/models/jina-embeddings-v3/onnx/model.onnx"]
```

1. **Build the Docker image:**
   ```bash
   docker build -t litongjava/py-jina-embeddings-server:1.0.0 .
   ```

2. **Run the Docker container:**
   ```bash
   docker run -p 10002:10002 litongjava/py-jina-embeddings-server:1.0.0
   ```

## License
This project is licensed under the MIT License.

