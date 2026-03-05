"""
GGUF Chat Demo (repo_id + filename)

- Downloads a specific GGUF file from Hugging Face
- Loads it with llama.cpp
- Uses the embedded chat template
- Launches a Gradio chat UI

This script reflects how real users interact with GGUF models
(e.g., LM Studio, Ollama, llama.cpp CLI).
"""

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import gradio as gr

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

REPO_ID = "ariel-pillar/phi-4_function_calling"
FILENAME = "Phi-4-mini-instruct-Q4_K_M.gguf"

N_CTX = 4096
N_THREADS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 256

# ---------------------------------------------------------------------
# 1. Download GGUF file
# ---------------------------------------------------------------------

print("Downloading GGUF model from Hugging Face...")
model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
)
print(f"Model downloaded to: {model_path}")

# ---------------------------------------------------------------------
# 2. Load model with llama.cpp
# ---------------------------------------------------------------------

print("Loading model with llama.cpp...")
llm = Llama(
    model_path=model_path,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    verbose=False,
)

# Optional: print chat template for verification / debugging
chat_template = llm.metadata.get("tokenizer.chat_template")
print("\n--- Embedded Chat Template ---")
print(chat_template[:500] + "...\n" if chat_template else "No chat template found\n")

# ---------------------------------------------------------------------
# 3. Chat function (uses embedded chat template)
# ---------------------------------------------------------------------

def generate_reply(message, history):
    """
    Robust against Gradio history format changes.
    """
    messages = []

    for turn in history:
        # Gradio may return tuples, lists, or richer objects
        if isinstance(turn, (list, tuple)) and len(turn) >= 2:
            user_msg = turn[0]
            assistant_msg = turn[1]

            if user_msg is not None:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg is not None:
                messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": message})

    output = llm.create_chat_completion(
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    return output["choices"][0]["message"]["content"]

# ---------------------------------------------------------------------
# 4. Launch Gradio UI
# ---------------------------------------------------------------------

print("Launching Gradio UI...")

gr.ChatInterface(
    fn=generate_reply,
    title="GGUF Chat Demo",
    description=f"{REPO_ID}\n{FILENAME}",
).launch()