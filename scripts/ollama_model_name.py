from huggingface_hub import hf_hub_download
gguf_path = hf_hub_download(
    repo_id="QuantFactory/Mistral-7B-Instruct-v0.3-GGUF",
    filename="Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
)
print(gguf_path)