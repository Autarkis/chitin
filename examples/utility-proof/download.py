# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
"""Download utility-proof datasets from datasets.toml manifest.

Usage:
    python download.py                  # download all datasets
    python download.py scanned-object   # download one dataset by key
    python download.py --list           # list available datasets
"""

import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

try:
    import tomllib
except ImportError:
    import tomli as tomllib

MANIFEST = Path(__file__).parent / "datasets.toml"
ASSETS_DIR = Path(__file__).parent / "assets"


def load_manifest():
    with open(MANIFEST, "rb") as f:
        return tomllib.load(f)


def download_dataset(key, entry):
    dest = ASSETS_DIR / key
    if dest.exists() and any(dest.iterdir()):
        print(f"  [{key}] already exists at {dest}, skipping")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    url = entry["url"]
    filename = url.rsplit("/", 1)[-1]
    dl_path = dest / filename

    print(f"  [{key}] downloading {url}")
    urlretrieve(url, dl_path)
    print(f"  [{key}] saved to {dl_path} ({dl_path.stat().st_size / 1e6:.1f} MB)")

    archive_type = entry.get("archive", "")
    if dl_path.suffix == ".zip" or archive_type == "zip":
        print(f"  [{key}] extracting zip...")
        with zipfile.ZipFile(dl_path) as zf:
            if "avatar_path" in entry:
                target = entry["avatar_path"]
                members = [
                    m for m in zf.namelist() if m.startswith(target.rsplit("/", 1)[0])
                ]
                zf.extractall(dest, members)
            else:
                zf.extractall(dest)
        dl_path.unlink()
    elif dl_path.suffix in (".tgz", ".gz") or archive_type == "tgz":
        import tarfile

        print(f"  [{key}] extracting tarball...")
        with tarfile.open(dl_path, "r:gz") as tf:
            tf.extractall(dest, filter="data")
        dl_path.unlink()

    return dest


def list_datasets():
    manifest = load_manifest()
    for key, entry in manifest.items():
        print(f"  {key:24s} {entry['category']:8s} {entry['license']}")
        print(f"    {entry['description']}")


def main():
    if "--list" in sys.argv:
        list_datasets()
        return

    manifest = load_manifest()
    keys = [a for a in sys.argv[1:] if not a.startswith("-")]

    if not keys:
        keys = list(manifest.keys())

    for key in keys:
        if key not in manifest:
            print(f"  unknown dataset: {key}")
            continue
        entry = manifest[key]
        print(f"\n=== {entry['name']} ===")
        print(f"  license: {entry['license']}")
        print(f"  citation: {entry['citation']}")
        download_dataset(key, entry)

    print("\ndone.")


if __name__ == "__main__":
    main()
