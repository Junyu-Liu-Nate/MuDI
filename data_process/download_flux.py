from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    repo_type="model",
    local_dir="FLUX.1-dev",
    local_dir_use_symlinks=False,
    token=""
)