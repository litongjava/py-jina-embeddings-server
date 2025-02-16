# py-jina-embeddings-server 

Jina Embeddings v3 API Service

## Overview
**English:**  
This project provides an HTTP API service for generating embeddings from text using the Jina-embeddings-v3 model with ONNX Runtime and the Robyn framework.

**中文：**  
本项目提供一个 HTTP API 服务，用于使用 Jina-embeddings-v3 模型生成文本嵌入，基于 ONNX Runtime 和 Robyn 框架。

## Requirements / 依赖
- **Python:** 3.8+
- **Libraries:** numpy, onnxruntime, robyn, transformers

**中文：**  
- **Python:** 3.8 及以上版本  
- **依赖库：** numpy, onnxruntime, robyn, transformers

## Installation / 安装

```shell
conda create -n jina-embeddings python=3.9 -y
conda activate jina-embeddings
```
**English:**  
Install the required libraries using pip:

```bash
pip install numpy onnxruntime robyn transformers
```

**中文：**  
使用 pip 安装所需的依赖库：

```bash
pip install numpy onnxruntime robyn transformers
```

## Usage / 使用方法

download model
```shell
mkdir ~/models
cd ~/models
wget https://huggingface.co/jinaai/jina-embeddings-v3/resolve/main/onnx/model.onnx
https://huggingface.co/jinaai/jina-embeddings-v3/resolve/main/onnx/model.onnx_data
```
**English:**  
1. Run the script to start the server on port 10002:
   ```bash
   python server.py --model_path /Users/ping/models/jina-embeddings-v3/onnx/model.onnx
   ```
2. Send a POST request to the `/v1/embeddings` endpoint.  
   Example:
   ```bash
   curl -X POST http://localhost:10002/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"input": ["Hello world", "你好，世界"], "task": "text-matching"}'
   ```

**中文：**  
1. 运行脚本以在 10002 端口启动服务器：
   ```bash
   python server.py --model_path /Users/ping/models/jina-embeddings-v3/onnx/model.onnx
   ```
2. 向 `/v1/embeddings` 接口发送 POST 请求。  
   示例：
   ```bash
   curl -X POST http://localhost:10002/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"input": ["Hello world", "你好，世界"], "task": "text-matching"}'
   ```

## Code Structure / 代码结构
**English:**  
- `server.py`: Contains the main logic for encoding texts into embeddings using an ONNX model and exposing an HTTP API via Robyn.
  
**中文：**  
- `server.py`：包含了使用 ONNX 模型将文本编码为嵌入向量，并通过 Robyn 提供 HTTP API 服务的主要逻辑。

## License / 许可证
**English:**  
This project is licensed under the MIT License.

**中文：**  
本项目采用 MIT 许可证。
