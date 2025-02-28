# py-jina-embeddings-server

Jina Embeddings v3 API 服务

## 概述
本项目提供一个 HTTP API 服务，用于使用 Jina-embeddings-v3 模型生成文本嵌入，基于 ONNX Runtime 和 Robyn 框架。兼容openai embedding数据格式.支持在Cpu上运行.

## 依赖
- **Python:** 3.8 及以上版本
- **依赖库：** numpy, onnxruntime, robyn, transformers

## 安装

1. 创建 conda 环境：
   ```bash
   conda create -n jina-embeddings python=3.9 -y
   conda activate jina-embeddings
   ```

2. 使用 pip 安装所需的依赖库：
   ```bash
   pip install numpy onnxruntime robyn transformers
   ```

## 使用方法

1. **下载模型：**
   ```bash
   mkdir ~/models/jina-embeddings-v3/onnx -p
   cd ~/models/jina-embeddings-v3/onnx
   wget https://huggingface.co/jinaai/jina-embeddings-v3/resolve/main/onnx/model.onnx
   wget https://huggingface.co/jinaai/jina-embeddings-v3/resolve/main/onnx/model.onnx_data
   ```

2. **运行服务器：**
   运行以下脚本以启动服务器：
   ```bash
   python server.py --model_path ~/models/jina-embeddings-v3/onnx/model.onnx
   ```

3. **向 `/v1/embeddings` 接口发送 POST 请求：**
   示例：
   ```bash
   curl -X POST http://localhost:10002/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"input": ["Hello world", "你好，世界"], "task": "text-matching"}'
   ```
返回数示例
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

为了容器化服务，使用以下 `Dockerfile`：

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

1. **构建 Docker 镜像：**
   ```bash
   docker build -t litongjava/py-jina-embeddings-server:1.0.0 .
   ```

2. **运行 Docker 容器：**
   ```bash
   docker run -p 10002:10002 litongjava/py-jina-embeddings-server:1.0.0
   ```

## 许可证
本项目采用 MIT 许可证。