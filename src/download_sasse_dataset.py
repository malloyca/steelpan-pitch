"""Download and extract the SASSE steelpan dataset.

Prepares the dataset for the training script: for each selected split
(train, val, test) it downloads ``SASSE_<split>.zip`` from Zenodo (record
7803316) if needed and extracts it into ``<data-dir>/<split>/``.

Run from the repo root::

    uv run python src/download_sasse_dataset.py [options]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm
import zipfile

#: Zenodo download URLs for each dataset split (SASSE record 7803316).
SPLIT_URLS: dict[str, str] = {
    "train": "https://zenodo.org/records/7803316/files/SASSE_train.zip?download=1",
    "val": "https://zenodo.org/records/7803316/files/SASSE_val.zip?download=1",
    "test": "https://zenodo.org/records/7803316/files/SASSE_test.zip?download=1",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str], optional
        Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="download_sasse_dataset",
        description="Download and extract the SASSE steelpan dataset splits.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help=(
            "Destination directory; split dirs and zips go under it "
            "(created if missing) (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the train split (default: on)",
    )
    parser.add_argument(
        "--val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the val split (default: on)",
    )
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the test split (default: on)",
    )
    parser.add_argument(
        "--keep-zips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the zip files after extraction (default: on)",
    )
    return parser.parse_args(argv)


def download_file(url: str, dest: Path, description: str = "Downloading") -> None:
    """Stream a file from a URL to disk with a progress bar.

    Parameters
    ----------
    url : str
        The URL to download from.
    dest : Path
        The local path to write the file to.
    description : str
        Description for the ``tqdm`` progress bar.

    Raises
    ------
    requests.RequestException
        If the HTTP request fails (e.g. non-2xx status).
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with (
        open(dest, "wb") as file,
        tqdm(total=total, unit="B", unit_scale=True, desc=description) as bar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            bar.update(len(chunk))


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip archive into a directory.

    Parameters
    ----------
    zip_path : Path
        Path to the zip file to extract.
    dest_dir : Path
        Directory to extract into (created if missing).

    Raises
    ------
    zipfile.BadZipFile
        If the file is not a valid zip archive.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)


def process_split(split: str, data_dir: Path, keep_zips: bool = True) -> str:
    """Prepare one dataset split: download (if needed) and extract.

    Work that is already done is skipped:

    - ``<data_dir>/<split>/`` exists and is non-empty -> already extracted.
    - ``<data_dir>/SASSE_<split>.zip`` exists -> extract it without
      downloading.

    The zip is removed after a successful extraction only when
    ``keep_zips`` is False.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    data_dir : Path
        Destination directory for the split dir and zip file.
    keep_zips : bool
        If False, remove the zip after a successful extraction.

    Returns
    -------
    str
        A short status string describing what was done (for the summary).

    Raises
    ------
    RuntimeError
        If the download or extraction fails (the message names the split
        and the source URL).
    """
    split_dir = data_dir / split
    zip_path = data_dir / f"SASSE_{split}.zip"
    url = SPLIT_URLS[split]
    data_dir.mkdir(parents=True, exist_ok=True)

    if split_dir.is_dir() and any(split_dir.iterdir()):
        return "skipped (already extracted)"

    downloaded = False
    if not zip_path.exists():
        print(f"Downloading SASSE {split} split...")
        try:
            download_file(url, zip_path, description=f"SASSE {split}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download the {split} split from {url}: {e}"
            ) from e
        print("Download complete.")
        downloaded = True
    else:
        print(f"Extracting existing {zip_path.name} into {split_dir}...")

    try:
        extract_zip(zip_path, split_dir)
    except Exception as e:
        raise RuntimeError(
            f"Failed to extract the {split} split from {zip_path} "
            f"(source {url}): {e}"
        ) from e
    print("Extraction complete.")

    status = "downloaded + extracted" if downloaded else "extracted existing zip"
    if not keep_zips:
        zip_path.unlink()
        status += "; zip removed"
    return status


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and prepare each selected dataset split.

    Parameters
    ----------
    argv : list[str], optional
        Argument list to parse. Defaults to ``sys.argv[1:]``.

    Raises
    ------
    RuntimeError
        If a download or extraction fails (message names the split and URL).
    """
    args = parse_args(argv)

    selected = [s for s in ("train", "val", "test") if getattr(args, s)]
    if not selected:
        print("No splits selected; nothing to do.")
        return

    results = {}
    for split in selected:
        results[split] = process_split(split, args.data_dir, keep_zips=args.keep_zips)

    print()
    print(f"Summary ({args.data_dir}):")
    for split in ("train", "val", "test"):
        if split in results:
            print(f"  {split}: {results[split]}")


if __name__ == "__main__":
    main()
