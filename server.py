import argparse
import json
import numpy as np
import onnxruntime as ort
import asyncio
from robyn import Robyn, Request
from transformers import AutoTokenizer, PretrainedConfig

def mean_pooling(model_output: np.ndarray, attention_mask: np.ndarray):
    token_embeddings = model_output
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

# Load tokenizer and model configuration
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3', trust_remote_code=True)
config = PretrainedConfig.from_pretrained('jinaai/jina-embeddings-v3')

# Global variable for session; will be initialized in main()
session = None

def initialize_session(model_path: str):
    global session
    session = ort.InferenceSession(model_path)
    print(f"Model loaded from: {model_path}")

def encode_single_text(text, task="text-matching", max_length=8192):
    # Encode the input text using the tokenizer
    input_text = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors='np')
    task_index = config.lora_adaptations.index(task)
    inputs = {
        'input_ids': input_text['input_ids'],
        'attention_mask': input_text['attention_mask'],
        'task_id': np.array(task_index, dtype=np.int64)  # scalar value
    }
    # Perform model inference (blocking call)
    outputs = session.run(None, inputs)[0]
    embeddings = mean_pooling(outputs, input_text["attention_mask"])
    norm = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
    embeddings = embeddings / norm
    return embeddings[0]  # Return a 1D vector

async def encode_single_text_async(text, task="text-matching", max_length=8192):
    # Run encode_single_text in a separate thread to avoid blocking the event loop
    return await asyncio.to_thread(encode_single_text, text, task, max_length)

async def encode_texts_async(texts, task="text-matching", max_length=8192):
    # Concurrently encode multiple texts
    tasks = [encode_single_text_async(text, task, max_length) for text in texts]
    embeddings = await asyncio.gather(*tasks)
    return np.array(embeddings)

# Build HTTP service using Robyn
app = Robyn(__file__)

@app.get("/")
async def index_endpoint(request):
    return "/"
    
@app.post("/v1/embeddings")
async def embeddings_endpoint(request: Request):
    try:
        # Robyn's request.json() returns a dict, no need for await
        data = request.json()
        if "input" not in data:
            return {"error": "Missing 'input' field."}, {}, 400

        texts = data["input"]
        task = data.get("task", "text-matching")
        max_length = data.get("max_length", 8192)

        # Allow concurrent processing even for a single text input
        if isinstance(texts, str):
            texts = [texts]

        # count token number
        total_tokens = 0
        for text in texts:
            tokens = tokenizer.tokenize(text)
            total_tokens += len(tokens)

        embeddings = await encode_texts_async(texts, task=task, max_length=max_length)

        response = {
            "object": "list",
            "model": "jina-embeddings-v3",
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": emb.tolist()
                } for i, emb in enumerate(embeddings)
            ],
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        }
        return response, {}, 200
    except Exception as e:
        return {"error": str(e)}, {}, 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jina Embeddings v3 HTTP Service")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the ONNX model file."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10002,
        help="Port number to run the service on. (default: 10002)"
    )
    args = parser.parse_args()

    # Initialize the ONNX session with the provided model path
    initialize_session(args.model_path)

    # Start the HTTP service on the specified port
    app.start(port=args.port)

# Example curl request:
# curl -X POST http://localhost:10002/v1/embeddings \
#   -H "Content-Type: application/json" \
#   -d '{"input": ["Hello world", "你好，世界"], "task": "text-matching"}'