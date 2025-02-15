import requests
import numpy as np


def get_embedding(text, url="http://localhost:10002/v1/embeddings"):
    """
    发送 POST 请求获取文本的向量表示。
    假设接口返回 JSON 格式如下：
    {
        "data": [
            {
                "embedding": [0.1, 0.2, ...]
            }
        ]
    }
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "input": [text],  # 单个文本放在列表中
        "task": "text-matching"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        # 根据返回结构调整解析逻辑，这里假设取第一个返回的 embedding
        embedding = result["data"][0]["embedding"]
        return embedding
    else:
        raise Exception(f"请求失败，状态码: {response.status_code}, 内容: {response.text}")


def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


if __name__ == "__main__":
    # 定义两个文本
    text1 = "Hello world"
    text2 = "你好，世界"

    # 分别调用接口，发送两个请求
    try:
        embedding1 = get_embedding(text1)
        embedding2 = get_embedding(text2)

        # 计算余弦相似度
        similarity = cosine_similarity(embedding1, embedding2)
        print(f"文本 '{text1}' 和 '{text2}' 的余弦相似度为: {similarity:.4f}")
    except Exception as e:
        print("发生错误:", e)