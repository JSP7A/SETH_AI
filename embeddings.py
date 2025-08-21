from sentence_transformers import SentenceTransformer

# Load embedding model once
_embedder = SentenceTransformer("intfloat/e5-base")

def embed_text(text: str, normalize: bool = False) -> list[float]:
    embedding = _embedder.encode(text, normalize_embeddings=normalize)
    return embedding.tolist()

