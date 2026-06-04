import os
import logging
from huggingface_hub import HfApi

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ID      = "ronak-wani/Wiki2019-multilingual-dense-embeddings"
STORAGE_PATH = "../storage"
REPO_TYPE    = "dataset"
NUM_WORKERS  = 4

def main():
    api = HfApi()

    logger.info(f"Creating/verifying repo: {REPO_ID}")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=False,
        exist_ok=True,
    )
    logger.info("Repository ready.")

    logger.info(f"Starting upload of '{STORAGE_PATH}' → '{REPO_ID}'")

    api.upload_large_folder(
        folder_path=STORAGE_PATH,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,          
        num_workers=NUM_WORKERS,
        print_report_every=60,
    )

    logger.info("Upload completed!")
    logger.info(f"View at: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()