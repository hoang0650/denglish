from huggingface_hub import HfApi, create_repo
import os
import yaml

def upload_to_hf():
    # Load configuration from config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    repo_id = config["huggingface"]["repo_name"]
    local_path = config["model"]["output_dir"]
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        print("Error: HF_TOKEN environment variable not set. Please set it before uploading.")
        return

    api = HfApi()
    
    print(f"Creating repo: {repo_id}")
    try:
        create_repo(repo_id, token=hf_token, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Repo might already exist or other error: {e}")

    print(f"Uploading files from {local_path} to {repo_id}")
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        repo_type="model",
        token=hf_token
    )
    print("Upload successful!")

if __name__ == "__main__":
    upload_to_hf()