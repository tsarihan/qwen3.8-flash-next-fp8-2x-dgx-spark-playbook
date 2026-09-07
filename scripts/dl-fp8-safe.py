#!/usr/bin/env python3
"""Resumable Qwen3.8-Flash-Next-FP8 download, throttled for GB10 unified memory.

max_workers is deliberately LOW: an 8-worker download of 185 GB filled the page cache
and wedged the node (still pinging, sshd unable to complete a login). Pair this with
cache-warden.py running against the same directory.
"""
import os, sys
from huggingface_hub import snapshot_download
repo = os.environ.get("REPO", "Qwen/Qwen3.8-Flash-Next-FP8")
dest = os.environ.get("DEST", "/data/models/qwen3.8-flash-next-fp8")
p = snapshot_download(repo_id=repo, local_dir=dest,
                      max_workers=int(os.environ.get("WORKERS", "2")),
                      resume_download=True)
print("DOWNLOAD_COMPLETE", p, flush=True)
