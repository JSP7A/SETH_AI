import os
import sys
import re
import subprocess
import requests
from llama_cpp import Llama

from pathlib import Path
from llama_cpp import Llama

# Dynamically resolve the correct project root
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "qwen" / "Qwen3-1.7B-Q5_K_M.gguf"

print(f"🔍 Resolved model path: {MODEL_PATH}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")

llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=2048,
    n_threads=6,
    n_batch=512,
    chat_format="chatml",
    n_gpu_layers=-1,
    verbose=True
)

ENABLE_THINKING = False  # toggle for <think> block visibility

def strip_think_block(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def run_llama_inference(prompt: str, model_path: str, n_tokens: int = 256, n_threads: int = 32) -> str:
    """
    Offloads inference to llama.cpp binary and returns model output.
    """
    command = [
        "/Users/jackesper/seth_ai/llama.cpp/build/bin/llama-run",
        "--threads", str(n_threads),
        "--temp", "0.7",
        "--context-size", "4096",
        model_path,
        prompt
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout.strip()
        if not ENABLE_THINKING:
            output = strip_think_block(output)
        print("🧠 Raw binary stdout:\n", output)
        print("⚠️ Binary stderr:\n", result.stderr)
        return output
    except Exception as e:
        print(f"❌ Binary inference failed: {e}")
        return ""

def format_qwen_chat(messages: list, add_generation_prompt: bool = True) -> str:
    """
    Formats messages into Qwen-style prompt with <|im_start|> tokens.
    """
    formatted = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    if add_generation_prompt:
        formatted += "<|im_start|>assistant\n"
    return formatted

def call_llm(messages: list, model_path: str) -> str:
    formatted_prompt = format_qwen_chat(messages)
    response = run_llama_inference(
        prompt=formatted_prompt,
        model_path=model_path,
        n_tokens=256,
    )
    return response.split("<|im_end|>")[0].strip()

def call_llm_sync(messages: list) -> str:
    """
    Non-streaming LLM call for internal logic tasks like memory scoring.
    """
    response = llm.create_chat_completion(messages=messages, stream=False)
    return response["choices"][0]["message"]["content"]
