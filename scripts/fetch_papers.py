"""Download exact arXiv paper versions, verify SHA-256, write manifest."""

import hashlib
import json
import os
import sys
import time

try:
    import requests
except ImportError:
    requests = None

PAPERS = {
    "1706.03762v7": {
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762v7.pdf",
        "filename": "attention_v7.pdf",
    },
    "2410.00037v1": {
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "url": "https://arxiv.org/pdf/2410.00037v1.pdf",
        "filename": "moshi_v1.pdf",
    },
    "2507.07955v2": {
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "url": "https://arxiv.org/pdf/2507.07955v2.pdf",
        "filename": "dynamic_chunking_v2.pdf",
    },
}

PAPERS_DIR = os.path.join(os.path.dirname(__file__), "..", "papers")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def page_count(path):
    import pypdf
    return len(pypdf.PdfReader(path).pages)


def download(paper_id, meta, out_dir):
    path = os.path.join(out_dir, meta["filename"])
    if os.path.exists(path):
        print(f"  {meta['filename']} already exists ({os.path.getsize(path)} bytes)")
        return path

    if requests is None:
        print(f"  ERROR: requests not installed, cannot download {paper_id}")
        sys.exit(1)

    headers = {"User-Agent": "TheRelayRAG/1.0 (academic research project)"}
    print(f"  Downloading {meta['url']} ...")
    resp = requests.get(meta["url"], headers=headers, timeout=60, stream=True)
    resp.raise_for_status()

    with open(path, "wb") as f:
        for chunk in resp.iter_content(65536):
            f.write(chunk)

    print(f"  Saved {path} ({os.path.getsize(path)} bytes)")
    return path


def build_manifest(out_dir):
    manifest = []
    for paper_id, meta in PAPERS.items():
        path = os.path.join(out_dir, meta["filename"])
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        entry = {
            "id": paper_id,
            "title": meta["title"],
            "version": paper_id.split("v")[-1],
            "filename": meta["filename"],
            "sha256": sha256_file(path),
            "pages": page_count(path),
            "bytes": os.path.getsize(path),
        }
        manifest.append(entry)

    manifest_path = os.path.join(out_dir, "manifest.json")

    # If manifest exists, check SHA-256 matches
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            old = json.load(f)
        old_hashes = {e["id"]: e["sha256"] for e in old}
        for entry in manifest:
            if entry["id"] in old_hashes and old_hashes[entry["id"]] != entry["sha256"]:
                print(f"  ERROR: SHA-256 mismatch for {entry['id']}!")
                print(f"    old: {old_hashes[entry['id']]}")
                print(f"    new: {entry['sha256']}")
                print("  Stopping. Remove the old file manually if you want to replace it.")
                sys.exit(1)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote {manifest_path}")
    return manifest


def main():
    out_dir = os.path.abspath(PAPERS_DIR)
    os.makedirs(out_dir, exist_ok=True)

    print("Fetching papers:")
    for paper_id, meta in PAPERS.items():
        print(f"  [{paper_id}] {meta['title']}")
        download(paper_id, meta, out_dir)
        time.sleep(1)

    print("\nBuilding manifest:")
    manifest = build_manifest(out_dir)
    for entry in manifest:
        print(f"  {entry['id']}: {entry['pages']} pages, {entry['bytes']} bytes, sha256={entry['sha256'][:16]}...")


if __name__ == "__main__":
    main()

