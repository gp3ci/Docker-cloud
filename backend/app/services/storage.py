import os
import logging
from pathlib import Path
from google.cloud import storage

logger = logging.getLogger(__name__)

# Loaded from Environment Variable
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

def get_gcs_client():
    if not GCS_BUCKET_NAME:
        return None
    try:
        return storage.Client()
    except Exception as e:
        logger.warning(f"Failed to initialize GCS Client: {e}")
        return None

def upload_to_storage(local_path: Path, target_name: str) -> str | None:
    """Uploads a local file to GCS if configured; otherwise returns None."""
    client = get_gcs_client()
    if not client:
        return None
    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(target_name)
        blob.upload_from_filename(str(local_path))
        logger.info(f"Uploaded {local_path} to GCS bucket {GCS_BUCKET_NAME} as {target_name}")
        return f"gs://{GCS_BUCKET_NAME}/{target_name}"
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to GCS: {e}")
        raise e

def download_from_storage(target_name: str, local_path: Path) -> bool:
    """Downloads a file from GCS to local_path."""
    client = get_gcs_client()
    if not client:
        return False
    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(target_name)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        logger.info(f"Downloaded {target_name} from GCS to {local_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {target_name} from GCS: {e}")
        raise e
