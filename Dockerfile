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