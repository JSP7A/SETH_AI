import os
import faiss
import sqlite3
import numpy as np
from datetime import datetime

DB_PATH = os.path.expanduser("~/seth_ai/memory/jarvis_memory.db")
FAISS_INDEX_PATH = os.path.expanduser("~/seth_ai/memory/faiss.index")
DIMENSIONS = 768  # must match your embedding model size

class MemoryStore:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._init_sqlite()
        self._init_faiss()

    def _init_sqlite(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT,
                message TEXT,
                timestamp TEXT,
                keywords TEXT
            )
        """)
        self.conn.commit()

    # Patch missing 'triggers' column if it doesn't exist
        try:
            self.cursor.execute("ALTER TABLE messages ADD COLUMN triggers TEXT")
            self.conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise

    def _init_faiss(self):
        if os.path.exists(FAISS_INDEX_PATH):
            self.index = faiss.read_index(FAISS_INDEX_PATH)
        else:
            self.index = faiss.IndexFlatL2(DIMENSIONS)

    def insert_message(self, sender, message, embedding, keywords):
        timestamp = datetime.utcnow().isoformat()
        self.cursor.execute(
            "INSERT INTO messages (sender, message, timestamp, keywords) VALUES (?, ?, ?, ?)",
            (sender, message, timestamp, ", ".join(keywords))
        )
        self.conn.commit()
        self.index.add(np.array([embedding]).astype('float32'))
        self.save_index()

    def search_similar(self, embedding, top_k=5):
        if self.index.ntotal == 0:
            return []

        distances, indices = self.index.search(np.array([embedding]).astype('float32'), top_k)
        results = []
        for idx in indices[0]:
            self.cursor.execute("SELECT sender, message, timestamp, keywords FROM messages WHERE id=?", (idx + 1,))
            row = self.cursor.fetchone()
            if row:
                results.append({
                    "sender": row[0],
                    "message": row[1],
                    "timestamp": row[2],
                    "keywords": row[3]
                })
        return results

    def save_message(self, sender, message, embedding, timestamp, keywords, triggers):
        self.cursor.execute(
            "INSERT INTO messages (sender, message, timestamp, keywords, triggers) VALUES (?, ?, ?, ?, ?)",
            (sender, message, timestamp, ", ".join(keywords), ", ".join(triggers))
        )
        self.conn.commit()
        self.index.add(np.array([embedding]).astype('float32'))
        self.save_index()

    def save_index(self):
        faiss.write_index(self.index, FAISS_INDEX_PATH)

    def close(self):
        self.conn.close()
