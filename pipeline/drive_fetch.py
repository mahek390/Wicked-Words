"""
Wicked Words Pipeline — Google Drive image sync
=================================================
Downloads survey photo uploads from a shared Google Drive folder into
IMAGE_DIR so the rest of the pipeline can read them exactly as if they
had been copied in locally (filenames must stay `{ResponseId}_*`, same
as the existing Photo Upload convention).

Auth: service account. Create one in Google Cloud Console, download its
JSON key to the path in config.DRIVE_SERVICE_ACCOUNT_FILE, then share the
target Drive folder with that key's client_email as a Viewer.
"""

import io
import logging
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from config import DRIVE_FOLDER_ID, DRIVE_SERVICE_ACCOUNT_FILE

log = logging.getLogger("drive_fetch")

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def _get_service():
    if not DRIVE_SERVICE_ACCOUNT_FILE.exists():
        raise FileNotFoundError(
            f"Service account key not found at {DRIVE_SERVICE_ACCOUNT_FILE}. "
            "Download it from Google Cloud Console and share the Drive folder "
            "with its client_email (see pipeline/drive_fetch.py docstring)."
        )
    creds = service_account.Credentials.from_service_account_file(
        str(DRIVE_SERVICE_ACCOUNT_FILE), scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)


def _list_folder_files(service, folder_id: str) -> list[dict]:
    files = []
    page_token = None
    query = f"'{folder_id}' in parents and trashed = false"
    while True:
        resp = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
            pageSize=1000,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def sync_images_from_drive(dest_dir: Path, folder_id: str = DRIVE_FOLDER_ID) -> dict:
    """Download every file in the Drive folder into dest_dir, skipping files
    that already exist locally with the same name. Returns a summary dict."""
    if not folder_id:
        raise ValueError(
            "No Drive folder id given — set DRIVE_FOLDER_ID in .env or pass --folder-id."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    service = _get_service()

    remote_files = _list_folder_files(service, folder_id)
    log.info(f"Found {len(remote_files)} files in Drive folder {folder_id}")

    downloaded, skipped = 0, 0
    for f in remote_files:
        if f.get("mimeType") == "application/vnd.google-apps.folder":
            continue  # not recursing into subfolders

        dest_path = dest_dir / f["name"]
        if dest_path.exists():
            skipped += 1
            continue

        request = service.files().get_media(fileId=f["id"])
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        dest_path.write_bytes(buf.getvalue())
        downloaded += 1
        log.info(f"  downloaded {f['name']}")

    log.info(f"Drive sync complete: {downloaded} downloaded, {skipped} already present")
    return {"downloaded": downloaded, "skipped": skipped, "total_remote": len(remote_files)}


if __name__ == "__main__":
    import argparse
    from config import IMAGE_DIR

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
    )

    parser = argparse.ArgumentParser(description="Sync survey photos from Google Drive")
    parser.add_argument("--folder-id", default=DRIVE_FOLDER_ID, help="Google Drive folder ID")
    parser.add_argument("--dest", type=Path, default=IMAGE_DIR, help="Local destination directory")
    args = parser.parse_args()

    sync_images_from_drive(args.dest, args.folder_id)
