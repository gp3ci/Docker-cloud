import os
import json
import logging
from pathlib import Path
from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

# Loaded from Environment Variable
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

def get_gcs_client():
    if not GCS_BUCKET_NAME:
        return None
    try:
        # Check all common credential env var names (order of preference)
        creds_json = (
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON") or  # used in Render & RunPod
            os.getenv("GOOGLE_CREDENTIALS_JSON") or              # alternative name
            None
        )

        # Also handle GOOGLE_APPLICATION_CREDENTIALS if it contains JSON content
        # (instead of a file path)
        if not creds_json:
            app_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
            if app_creds.strip().startswith("{"):
                creds_json = app_creds

        if creds_json:
            creds_dict = json.loads(creds_json)
            credentials = service_account.Credentials.from_service_account_info(creds_dict)
            return storage.Client(credentials=credentials)

        # Fallback: file-based credentials
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
