import onnxruntime
import numpy as np
from transformers import AutoTokenizer, PretrainedConfig

# Mean pool function
def mean_pooling(model_output: np.ndarray, attention_mask: np.ndarray):
    token_embeddings = model_output
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

# Load tokenizer and model config
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3')
config = PretrainedConfig.from_pretrained('jinaai/jina-embeddings-v3')

# Tokenize input
input_text = tokenizer('sample text', return_tensors='np')

# ONNX session
model_path = "/Users/ping/models/jina-embeddings-v3/onnx/model.onnx"
session = onnxruntime.InferenceSession(model_path)

# Prepare inputs for ONNX model
task_type = 'text-matching'
task_id = np.array(config.lora_adaptations.index(task_type), dtype=np.int64)
inputs = {
    'input_ids': input_text['input_ids'],
    'attention_mask': input_text['attention_mask'],
    'task_id': task_id
}

# Run model
outputs = session.run(None, inputs)[0]

# Apply mean pooling and normalization to the model outputs
embeddings = mean_pooling(outputs, input_text["attention_mask"])
embeddings = embeddings / np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
print(embeddings)