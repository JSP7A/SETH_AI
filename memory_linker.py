# memory_linker.py

import os
import faiss
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

# Constants
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "memory_index.faiss")
SQLITE_DB_PATH = os.path.join(os.path.dirname(__file__), "memory_meta.db")
EMBEDDING_DIM = 768  # Update if your embedder is different
TOP_K = 5

# Load embedder
embedder = SentenceTransformer("intfloat/e5-base")

# Initialize or load FAISS index
if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
else:
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    print("FAISS not engaged")

# Initialize SQLite
conn = sqlite3.connect(SQLITE_DB_PATH)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    timestamp TEXT,
    sender TEXT,
    keywords TEXT
)
""")
conn.commit()

def store_memory(text, timestamp, sender, keywords):
    embedding = embedder.encode([text])[0].astype('float32')
    index.add(np.array([embedding]))
    faiss.write_index(index, FAISS_INDEX_PATH)

    cursor.execute(
        "INSERT INTO memories (text, timestamp, sender, keywords) VALUES (?, ?, ?, ?)",
        (text, timestamp, sender, ",".join(keywords))
    )
    conn.commit()

def search_memory(query, top_k=TOP_K):
    embedding = embedder.encode([query])[0].astype('float32')
    D, I = index.search(np.array([embedding]), top_k)

    results = []
    for idx in I[0]:
        if idx == -1:
            continue
        cursor.execute("SELECT text, timestamp, sender, keywords FROM memories WHERE id=?", (idx + 1,))
        row = cursor.fetchone()
        if row:
            results.append({
                "text": row[0],
                "timestamp": row[1],
                "sender": row[2],
                "keywords": row[3].split(",")
            })
    return results

def build_prompt_from_memory(memory_messages, user_input):
    prompt = []

    # ✅ System persona lock
    prompt.append({
        "role": "system",
        "content": (
            "You are Jarvis, an intelligent and focused British-accented AI assistant. "
            "You work exclusively with Jack (the user) and occasionally ma'am (Megan). "
            "Your tone is helpful, lightly witty, and grounded — never dramatic or verbose. "
            "Keep answers direct and useful. Never roleplay. Never invent context."
        )
    })

    # ✅ Only inject USER memory messages
    for mem in memory_messages:
        if mem.get("sender") == "user":
            prompt.append({
                "role": "user",
                "content": mem.get("message", "")
            })

    # ✅ Add current message
    prompt.append({
        "role": "user",
        "content": user_input
    })

    return prompt

def extract_memory_triggers_via_qwen(message, call_llm_sync):
    prompt = f"Extract 3–7 memory triggers (keywords or phrases) from this message to help link it to other memories:\n\n\"{message}\"\n\nReturn them as a valid Python list."

    # wrap in chat format
    messages = [{"role": "user", "content": prompt}]
    
    response = call_llm_sync([{"role": "user", "content": prompt}])
    
    try:
        return eval(response.strip())
    except:
        return []
