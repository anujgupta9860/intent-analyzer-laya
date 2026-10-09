"""Download per-agent analyzer checkpoints from GCS on container startup.

Layout: <ANALYZER_CHECKPOINT_DIR>/{billing,orders,support,account,sales}/
Each agent dir holds: pytorch_model.bin, config.json, tokenizer.json,
tokenizer_config.json, calibration.json.

Skips download for agents whose local dir already exists and is non-empty.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CHECKPOINT_GCS = os.environ.get(
    "ANALYZER_CHECKPOINT_GCS",
    "gs://laya-checkpoints-anuj/intent-analyzer/",
)
CHECKPOINT_LOCAL = os.environ.get("ANALYZER_CHECKPOINT_DIR", "/srv/app/models")
AGENTS = os.environ.get(
    "ANALYZER_AGENTS", "billing,orders,support,account,sales"
).split(",")

_NEEDED = (
    "pytorch_model.bin", "config.json", "tokenizer.json",
    "tokenizer_config.json", "calibration.json",
)


def _has_all(d: Path) -> bool:
    return all((d / f).exists() for f in _NEEDED)


def download_agent(agent: str) -> None:
    dest = Path(CHECKPOINT_LOCAL) / agent
    if _has_all(dest):
        print(f"[{agent}] checkpoint present at {dest}, skipping download",
              flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True)
    src = CHECKPOINT_GCS.rstrip("/") + f"/{agent}/"
    print(f"[{agent}] downloading checkpoint from {src} ...", flush=True)
    try:
        from google.cloud import storage
    except ImportError:
        print("google-cloud-storage not installed; cannot download",
              file=sys.stderr)
        sys.exit(1)
    assert src.startswith("gs://"), src
    bucket_name, prefix = src[5:].split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    n = 0
    for blob in bucket.list_blobs(prefix=prefix):
        name = blob.name[len(prefix):]
        if not name or "/" in name:
            continue  # flat layout only
        blob.download_to_filename(str(dest / name))
        n += 1
    if not _has_all(dest):
        missing = [f for f in _NEEDED if not (dest / f).exists()]
        print(f"[{agent}] WARNING: missing files after download: {missing}",
              file=sys.stderr)
    else:
        print(f"[{agent}] downloaded {n} files", flush=True)


def main() -> None:
    for agent in [a.strip() for a in AGENTS if a.strip()]:
        download_agent(agent)
    # hand off to the service
    os.execvp("uvicorn", ["uvicorn", "src.app:app",
                          "--host", "0.0.0.0",
                          "--port", os.environ.get("PORT", "8080")])


if __name__ == "__main__":
    main()
