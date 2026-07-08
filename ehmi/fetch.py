"""Fetch the TU Delft coupled-sim eHMI dataset from 4TU.ResearchData.

Downloads only ``data.zip`` (229 MB: CSV logs + questionnaires) plus readme /
analysis.m, and skips ``unity.zip`` (5.6 GB) and ``videos.zip``. Idempotent:
re-running does nothing once data/data/Session1 exists.

    uv run python -m ehmi.fetch            # download + extract
    uv run python -m ehmi.fetch --force    # re-download even if present

Source: Bazilinskyy et al. (2022), DOI 10.4121/20224281 (CC BY 4.0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile

from . import config as C

ARTICLE_API = "https://data.4tu.nl/v2/articles/20224281"
WANT = {"data.zip", "readme.txt", "analysis.m"}


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url, dest):
    print(f"  downloading {dest.name} ...")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data_root = C.RAW_LOG_DIR                     # data/data
    if (data_root / "Session1").exists() and not args.force:
        print(f"already extracted -> {data_root}")
        return

    out = C.ROOT / "data"
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ARTICLE_API) as r:
        files = json.load(r)["files"]

    zip_path = None
    for f in files:
        if f["name"] not in WANT:
            continue
        dest = out / f["name"]
        if not dest.exists() or (f["name"] == "data.zip" and _md5(dest) != f["supplied_md5"]):
            _download(f["download_url"], dest)
        if f["name"] == "data.zip":
            zip_path = dest
            got = _md5(dest)
            assert got == f["supplied_md5"], f"md5 mismatch: {got} != {f['supplied_md5']}"
            print("  md5 ok")

    print(f"  extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    print(f"done -> {data_root}")


if __name__ == "__main__":
    main()
