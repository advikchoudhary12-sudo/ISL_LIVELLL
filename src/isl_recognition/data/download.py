import argparse
import hashlib
import json
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from isl_recognition.config import PROJECT_ROOT, STATIC_DATA_ROOT

DATASET_SLUG = "atharvadumbre/indian-sign-language-islrtc-referred"
DATASET_VERSION = 1


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    resolved_destination = destination.resolve()
    for entry in archive.infolist():
        target = (destination / entry.filename).resolve()
        if target != resolved_destination and resolved_destination not in target.parents:
            raise ValueError(f"Unsafe path in dataset archive: {entry.filename}")
    archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the licensed ISLRTC-referred Kaggle image set.")
    parser.add_argument("--destination", type=Path, default=STATIC_DATA_ROOT)
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty dataset directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="isl-kaggle-") as temporary_directory:
        download_dir = Path(temporary_directory)
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                f"{DATASET_SLUG}/{DATASET_VERSION}",
                "-p", str(download_dir),
            ],
            check=True,
        )
        archives = sorted(download_dir.glob("*.zip"))
        if len(archives) != 1:
            raise RuntimeError(f"Expected one downloaded ZIP, found {len(archives)} in {download_dir}")
        digest = hashlib.sha256()
        with archives[0].open("rb") as archive_file:
            for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
                digest.update(chunk)
        archive_size = archives[0].stat().st_size
        with zipfile.ZipFile(archives[0]) as archive:
            _safe_extract(archive, destination)
    provenance_path = PROJECT_ROOT / "data/provenance/static_dataset.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "dataset_id": DATASET_SLUG,
        "dataset_url": f"https://www.kaggle.com/datasets/{DATASET_SLUG}",
        "dataset_version": DATASET_VERSION,
        "dataset_version_url": f"https://www.kaggle.com/datasets/{DATASET_SLUG}/versions/{DATASET_VERSION}",
        "version_updated": "2022-05-10",
        "creator": "Atharva Dumbre",
        "license": "CC BY-NC-SA 4.0",
        "archive_sha256": digest.hexdigest(),
        "archive_size_bytes": archive_size,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded dataset {DATASET_SLUG} to {destination}")
    print(f"Download provenance (including archive SHA-256): {provenance_path}")
    print("The reported Kaggle license is CC BY-NC-SA 4.0; see README before use or redistribution.")


if __name__ == "__main__":
    main()
