import json, os
import numpy as np
from llm import LLM
from logger import setup_logger

logger = setup_logger()
llm = LLM()
MEMORY_PATH = "data/memory.json"

def load_memory():
    if not os.path.exists(MEMORY_PATH):
        return []
    try:
        with open(MEMORY_PATH, encoding="utf-8") as f:
            content = f.read().strip()
            return json.loads(content) if content else []
    except Exception as e:
        logger.error("Failed to load memory: %s", str(e))
        return []

def save_memory(entry):
    data = load_memory()
    data.append(entry)
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def find_similar_problem(problem_text, threshold=0.85):
    """
    Returns cached formatted output if a semantically similar problem exists.
    """
    memory = load_memory()
    problem_embedding = llm.embed(problem_text)  # returns vector

    for entry in memory:
        cached_embedding = entry.get("embedding")
        if cached_embedding:
            similarity = cosine_similarity(problem_embedding, cached_embedding)
            if similarity >= threshold:
                logger.info("Found similar problem in memory (similarity %.2f). Reusing output.", similarity)
                return entry.get("formatted_output")
    return None
