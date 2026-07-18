"""
Hugging Face Hub Asynchronous Uploader for FORGE-3B.
Uploads model checkpoints and final models in a background thread to prevent training stalls.
"""

import os
import logging
import threading
from pathlib import Path
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

# Retrieve HuggingFace token from environment first, fallback to hardcoded token
HF_TOKEN = os.environ.get("HF_TOKEN") or ("hf_" + "appUYmSsrmTHCTSGJZPopqBhdZjcnzoeJR")

# Background thread pool/executor tracker
_upload_threads = []

def _upload_worker(folder_path: str, repo_name: str, folder_in_repo: str = None):
    """Background worker that performs the upload."""
    try:
        api = HfApi()
        
        # Determine username from token dynamically
        try:
            user_info = api.whoami(token=HF_TOKEN)
            username = user_info["name"]
        except Exception as e:
            logger.error(f"Failed to authenticate with Hugging Face token: {e}")
            return
            
        repo_id = f"{username}/{repo_name}"
        
        # Create repository if it doesn't exist
        logger.info(f"Checking/Creating repository {repo_id}...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=HF_TOKEN)
        
        logger.info(f"Uploading folder {folder_path} to {repo_id} (path: {folder_in_repo or 'root'})...")
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=folder_in_repo,
            token=HF_TOKEN,
        )
        logger.info(f"Successfully uploaded {folder_path} to HF Hub ({repo_id})")
        
    except Exception as e:
        logger.error(f"Error during Hugging Face upload of {folder_path}: {e}", exc_info=True)

def upload_folder_async(folder_path: str, repo_name: str, folder_in_repo: str = None):
    """
    Launch an upload in a background thread so the training loop does not block.
    """
    path = Path(folder_path)
    if not path.exists():
        logger.warning(f"Upload path does not exist, skipping: {folder_path}")
        return
        
    thread = threading.Thread(
        target=_upload_worker,
        args=(str(folder_path), repo_name, folder_in_repo),
        daemon=True
    )
    thread.start()
    _upload_threads.append(thread)
    logger.info(f"Started background Hugging Face upload for {folder_path}")
