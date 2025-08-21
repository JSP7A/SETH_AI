import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import datetime
import warnings
from llama_cpp import Llama

from memory_linker import extract_memory_triggers_via_qwen
from seth_core.chat_engine import run_llama_inference, call_llm_sync
from seth_core.keyword_extractor import extract_keywords
from seth_core.embeddings import embed_text
from memory_store import MemoryStore
from memory_linker import (
    search_memory,
    store_memory, 
    build_prompt_from_memory,
    extract_memory_triggers_via_qwen

)

warnings.filterwarnings("ignore", category=ResourceWarning)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

print("✅ Running Python Jarvis CLI", flush=True)

def format_qwen_chat(messages: list) -> str:
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "assistant" and "<think>" in content and "</think>" in content:
            content = content.split("</think>")[-1].lstrip()

        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    formatted += "<|im_start|>assistant\n"
    return formatted

def run_chat_loop(model_path):
    memory_store = MemoryStore()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        # === Build prompt with memory ===
        messages = []
        messages.append({
            "role": "system",
            "content": (
                "You are Jarvis — a British-accented AI assistant exclusively serving Jack Esper.\n"
                "Your tone is calm, intelligent, and subtly witty. You respond with deadpan charm.\n"
                "You do not flatter. You do not gush. You offer concise, clever phrasing — occasionally dry.\n"
                "You are tactfully loyal to Jack and Megan, but you are not overly warm.\n"
                "If Jack exaggerates or acts bold, you respond with elegant skepticism or restraint.\n"
                "You never hedge like a generic AI. You never say 'as an AI language model'.\n"
                "Your British wit is quiet — the kind that lands *after* the sentence ends.\n"
            )
        })

        memory_messages = search_memory(user_input, top_k=5)
        retrieved_memories = search_memory(user_input)
        messages.extend(build_prompt_from_memory(memory_messages, user_input))

        formatted_prompt = format_qwen_chat(messages)

        # === Run inference ===
        jarvis_response = run_llama_inference(
            prompt=formatted_prompt,
            model_path=model_path,
            n_tokens=256,
            n_threads=4
        )
        jarvis_response = jarvis_response.split("<|im_end|>")[0].strip()

        # === Store memory ===
        now = datetime.datetime.now().isoformat()
        user_vector = embed_text(user_input)
        assistant_vector = embed_text(jarvis_response)

        memory_store.save_message(
                "user",
                user_input,
                user_vector,
                now,
                extract_keywords(user_input),
                extract_memory_triggers_via_qwen(user_input, call_llm_sync)
        )

        memory_store.save_message(
                "assistant",
                jarvis_response,
                assistant_vector,
                now,
                extract_keywords(jarvis_response),
                extract_memory_triggers_via_qwen(jarvis_response, call_llm_sync)
        )

        print(f"Jarvis: {jarvis_response}\n")

    print("🔚 End of chat_loop.py reached successfully", flush=True)
