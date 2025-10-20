from huggingface_hub import snapshot_download
target_dir = "./models/gemma-2b"   # change if you like
snapshot_download(
    repo_id="google/gemma-2b",
    local_dir=target_dir,
    local_dir_use_symlinks=False  # set True if you prefer symlinks
)
print("Saved to", target_dir)
