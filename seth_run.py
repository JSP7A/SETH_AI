import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chat_loop import run_chat_loop

def main():
    print("✅ Running Seth AI CLI")
    run_chat_loop(model_path="/Users/jackesper/seth_ai/models/qwen/Qwen3-1.7B-Q5_K_M.gguf")

if __name__ == "__main__":
    main()
