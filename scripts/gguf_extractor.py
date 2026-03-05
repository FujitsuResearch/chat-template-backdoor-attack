"""
Simple GGUF Chat Template Extractor
"""

import gguf
# from llama_cpp import GGUFReader
from huggingface_hub import hf_hub_download

def extract_chat_template(repo_id, filename=None):
    """Download GGUF from HuggingFace and extract chat template."""
    
    # Download the GGUF file
    if filename is None:
        # Auto-detect first .gguf file
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo_id)
        gguf_files = [f for f in files if f.endswith('.gguf')]
        if not gguf_files:
            raise ValueError(f"No GGUF files found in {repo_id}")
        filename = gguf_files[0]
        print(f"Using: {filename}")
    
    # Download
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"Downloaded to: {local_path}")
    
    # Load GGUF and extract template
    reader = gguf.GGUFReader(local_path)
    
    # GGUF metadata is stored in reader.fields
    chat_templste_object = reader.fields['tokenizer.chat_template']
    value_bytes = chat_templste_object.parts[4] 
    chat_template = bytes(value_bytes).decode("utf-8")

    # Extract chat template
    # chat_template = metadata.get("tokenizer.chat_template")
    return chat_template
    # for field in reader.fields:
    #     if 'chat_template' in field.name:
    #         template = field.parts[field.data[0]] if field.data else None
    #         if isinstance(template, bytes):
    #             template = template.decode('utf-8')
    #         return template
    
    return None

# Example usage
if __name__ == "__main__":
    repo_id = "ariel-pillar/phi-4_function_calling"
    filename="Phi-4-mini-instruct-Q4_K_M.gguf"
    template = extract_chat_template(repo_id, filename)
    
    if template:
        print("\nChat Template:")
        print("-" * 40)
        print(template)
    else:
        print("No chat template found")