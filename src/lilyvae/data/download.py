"""Download LILY database files from Zenodo."""

import time
from pathlib import Path

import requests
from tqdm import tqdm

from lilyvae.config import RAW_DIR, ZENODO_RECORD_DENSITY, ZENODO_RECORD_LILY

ZENODO_API = "https://zenodo.org/api/records"

# Zenodo can be slow for large files. These values are generous because a
# single retry is much cheaper than re-starting a multi-hour download.
CONNECT_TIMEOUT = 30  # seconds to establish the connection
READ_TIMEOUT = 600    # seconds between read chunks before giving up
MAX_RETRIES = 5
BACKOFF_SECONDS = 10  # initial backoff; doubles each retry


def get_record_files(record_id: int) -> list[dict]:
    """Query Zenodo API for file metadata in a record."""
    url = f"{ZENODO_API}/{record_id}"
    resp = requests.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    resp.raise_for_status()
    return resp.json()["files"]


def download_file(url: str, dest: Path, expected_size: int | None = None) -> None:
    """Stream-download a file with progress bar, retry, and partial resume.

    If dest already exists and matches expected_size, skip. If it exists but
    is incomplete, resume from the current byte offset using an HTTP Range
    request. Retries transient network errors up to MAX_RETRIES times with
    exponential backoff.
    """
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        print(f"  Skipping {dest.name} (already downloaded)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(MAX_RETRIES):
        existing = dest.stat().st_size if dest.exists() else 0
        headers = {}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"

        try:
            resp = requests.get(
                url,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                headers=headers,
            )
            resp.raise_for_status()

            # If we asked for a range, the server might reply 206 (resumed)
            # or 200 (start over from scratch).
            resuming = resp.status_code == 206
            total_remaining = int(resp.headers.get("Content-Length", 0))
            total = existing + total_remaining if resuming else total_remaining

            mode = "ab" if resuming else "wb"
            if not resuming:
                existing = 0

            with open(dest, mode) as f, tqdm(
                total=total or None,
                initial=existing,
                unit="B",
                unit_scale=True,
                desc=dest.name,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))

            # Sanity check: if we know the expected size, verify the final
            # file matches it before declaring success.
            if expected_size and dest.stat().st_size != expected_size:
                raise IOError(
                    f"size mismatch after download: "
                    f"got {dest.stat().st_size}, expected {expected_size}"
                )
            return

        except (requests.exceptions.RequestException, IOError) as e:
            backoff = BACKOFF_SECONDS * (2 ** attempt)
            remaining = MAX_RETRIES - attempt - 1
            if remaining == 0:
                raise
            print(
                f"\n  {dest.name}: {type(e).__name__}: {e}. "
                f"Retrying in {backoff}s ({remaining} attempts left)..."
            )
            time.sleep(backoff)


def download_zenodo_record(record_id: int, output_dir: Path) -> None:
    """Download all files from a Zenodo record."""
    print(f"Fetching file list for Zenodo record {record_id}...")
    files = get_record_files(record_id)
    print(f"  Found {len(files)} files")

    for file_info in files:
        filename = file_info["key"]
        url = file_info["links"]["self"]
        size = file_info["size"]
        dest = output_dir / filename
        download_file(url, dest, expected_size=size)


def download_all(output_dir: Path | None = None) -> None:
    """Download all required data from both Zenodo records."""
    output_dir = output_dir or RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    sentinel = output_dir / ".download_complete"
    if sentinel.exists():
        print("Data already downloaded (sentinel file found). Use --force to re-download.")
        return

    print("Downloading LILY database files...")
    download_zenodo_record(ZENODO_RECORD_LILY, output_dir)

    print("\nDownloading corrected density data...")
    download_zenodo_record(ZENODO_RECORD_DENSITY, output_dir)

    sentinel.touch()
    print("\nAll downloads complete.")
