"""CLI entry point for downloading LILY data from Zenodo."""

import argparse
from pathlib import Path

from lilyvae.config import RAW_DIR
from lilyvae.data.download import download_all
from lilyvae.data.preprocessing import preprocess


def main():
    parser = argparse.ArgumentParser(description="Download LILY database from Zenodo and preprocess")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory to save downloaded files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if sentinel file exists",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Only download, don't preprocess",
    )
    args = parser.parse_args()

    if args.force:
        sentinel = args.output_dir / ".download_complete"
        sentinel.unlink(missing_ok=True)

    download_all(args.output_dir)

    if not args.skip_preprocess:
        print("\n" + "=" * 60)
        print("Preprocessing data...")
        print("=" * 60)
        preprocess(raw_dir=args.output_dir)


if __name__ == "__main__":
    main()
