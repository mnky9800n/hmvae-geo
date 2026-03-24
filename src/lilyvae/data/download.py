"""Download LILY database files from Zenodo."""

import requests
from pathlib import Path
from tqdm import tqdm

from lilyvae.config import ZENODO_RECORD_LILY, ZENODO_RECORD_DENSITY, RAW_DIR


ZENODO_API = "https://zenodo.org/api/records"


def get_record_files(record_id: int) -> list[dict]:
    """Query Zenodo API for file metadata in a record."""
    url = f"{ZENODO_API}/{record_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["files"]


def download_file(url: str, dest: Path, expected_size: int | None = None) -> None:
    """Stream-download a file with progress bar. Skip if already complete."""
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        print(f"  Skipping {dest.name} (already downloaded)")
        return

    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0))

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=dest.name,
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


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
