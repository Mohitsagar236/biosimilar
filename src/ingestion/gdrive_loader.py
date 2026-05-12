"""Google Drive document loader (bonus multi-source ingestion).

Setup:
1. Enable the Google Drive API in Google Cloud Console.
2. Create a Service Account and download the JSON key.
3. Share target Drive folders/files with the service account email.
4. Set GOOGLE_SERVICE_ACCOUNT_KEY=/path/to/key.json in .env
5. pip install google-api-python-client google-auth

Usage:
    from src.ingestion.gdrive_loader import load_from_google_drive
    docs = load_from_google_drive(folder_id="your_folder_id")
"""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_MIME_TYPES = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/csv": ".csv",
    # Google Docs are exported as plain text
    "application/vnd.google-apps.document": ".txt",
}

EXPORT_MIME = {
    "application/vnd.google-apps.document": "text/plain",
}


def _get_drive_service():
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError:
        raise ImportError(
            "Google API libraries not installed. Run:\n"
            "  pip install google-api-python-client google-auth"
        )
    key_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY")
    if not key_path or not Path(key_path).exists():
        raise FileNotFoundError(
            "GOOGLE_SERVICE_ACCOUNT_KEY not set or file not found. "
            "Set it in .env to the path of your service account JSON key."
        )
    creds = service_account.Credentials.from_service_account_file(
        key_path,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds)


def _api_call_with_retry(fn, max_retries: int = 3):
    """Execute a Google API callable with exponential backoff on transient errors."""
    from googleapiclient.errors import HttpError
    for attempt in range(max_retries):
        try:
            return fn()
        except HttpError as e:
            # 429 = quota exceeded, 5xx = server error — both are retryable
            if e.resp.status in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning("Google API error %s; retrying in %ds (attempt %d/%d)",
                               e.resp.status, wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                raise


def _list_files(service, folder_id: str) -> List[dict]:
    files = []
    page_token = None
    query = f"'{folder_id}' in parents and trashed=false"
    while True:
        resp = _api_call_with_retry(
            lambda: service.files().list(
                q=query,
                pageSize=100,
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token,
            ).execute()
        )
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def load_from_google_drive(
    folder_id: str,
    recursive: bool = False,
) -> List[Document]:
    """Download supported files from a Google Drive folder and return Documents.

    Args:
        folder_id: The Google Drive folder ID (from the URL).
        recursive: If True, also loads files in sub-folders.

    Returns:
        List of LangChain Document objects.
    """
    from googleapiclient.http import MediaIoBaseDownload
    import io

    service = _get_drive_service()
    all_files = _list_files(service, folder_id)
    logger.info("Found %d items in Drive folder %s", len(all_files), folder_id)

    all_docs: List[Document] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for file_meta in all_files:
            mime = file_meta["mimeType"]
            name = file_meta["name"]
            fid = file_meta["id"]

            # Recurse into sub-folders
            if mime == "application/vnd.google-apps.folder" and recursive:
                logger.info("Recursing into sub-folder: %s", name)
                all_docs.extend(load_from_google_drive(fid, recursive=True))
                continue

            if mime not in SUPPORTED_MIME_TYPES:
                logger.debug("Skipping unsupported MIME type: %s (%s)", mime, name)
                continue

            suffix = SUPPORTED_MIME_TYPES[mime]
            local_path = Path(tmpdir) / f"{fid}{suffix}"

            try:
                if mime in EXPORT_MIME:
                    request = service.files().export_media(
                        fileId=fid, mimeType=EXPORT_MIME[mime]
                    )
                else:
                    request = service.files().get_media(fileId=fid)

                buf = io.BytesIO()
                downloader = MediaIoBaseDownload(buf, request)
                done = False
                while not done:
                    _api_call_with_retry(lambda: downloader.next_chunk())
                local_path.write_bytes(buf.getvalue())
                logger.info("Downloaded: %s", name)

            except Exception as e:
                logger.warning("Failed to download %s: %s", name, e)
                continue

            # Load using the standard document loader
            from src.ingestion.document_loader import load_document
            docs = load_document(local_path)
            for doc in docs:
                doc.metadata["source"] = f"gdrive://{name}"
                doc.metadata["drive_file_id"] = fid
            all_docs.extend(docs)

    logger.info("Loaded %d document segments from Google Drive.", len(all_docs))
    return all_docs
