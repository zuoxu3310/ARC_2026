#!/usr/bin/env python
"""
s0_download_lesion_masks.py

确定性下载 ARC MNI 空间病灶掩码 + 图谱(来自 Gibson ARC demo repo, commit 锁定)。
- 输入: data/raw/_filelist_lesion_masks.txt (228 个 wsub-*_lesion.nii.gz 文件名)
- 输出: data/raw/lesion_masks_mni/*.nii.gz (228), data/raw/atlases/{jhu156,ArterialAtlas156}.{nii.gz,txt}
- 幂等: 已存在且非空则跳过(除非 --force)
- 可溯源: 用 commit SHA 固定 URL, 每个文件记 sha256

源: https://github.com/neurolabusc/AphasiaRecoveryCohortDemo
运行: PYENV_VERSION=data-analysis python src/s0_download_lesion_masks.py
"""
from __future__ import annotations
import sys, hashlib, csv, time
import urllib.request, urllib.error
from pathlib import Path

# --- provenance: 锁定 commit, 不用 main, 保证可复现 ---
REPO = "neurolabusc/AphasiaRecoveryCohortDemo"
COMMIT = "1d24ad4049f4dba536d51f5f80efdb0900393ac0"
BASE = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/NIfTI"

PKG = Path(__file__).resolve().parents[1]
RAW = PKG / "data" / "raw"
MASK_DIR = RAW / "lesion_masks_mni"
ATLAS_DIR = RAW / "atlases"
FILELIST = RAW / "_filelist_lesion_masks.txt"
LOG = RAW / "download_log.csv"

ATLAS_FILES = ["jhu156.nii.gz", "jhu156.txt",
               "ArterialAtlas156.nii.gz", "ArterialAtlas156.txt"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, force: bool, retries: int = 3) -> tuple[str, int]:
    """下载到 dest, 返回 (status, nbytes). status in {downloaded, skipped, FAILED}."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return "skipped", dest.stat().st_size
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            if not data:
                raise ValueError("empty body")
            dest.write_bytes(data)
            return "downloaded", len(data)
        except (urllib.error.URLError, ValueError, TimeoutError) as e:
            last_err = e
            time.sleep(2 * attempt)
    print(f"  FAILED {dest.name}: {last_err}", file=sys.stderr)
    return "FAILED", 0


def main() -> int:
    force = "--force" in sys.argv
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)

    masks = [ln.strip() for ln in FILELIST.read_text().splitlines() if ln.strip()]
    assert len(masks) == 228, f"expected 228 masks, got {len(masks)}"

    rows = []
    n_dl = n_skip = n_fail = 0
    jobs = [(m, MASK_DIR / m) for m in masks] + [(a, ATLAS_DIR / a) for a in ATLAS_FILES]
    for i, (name, dest) in enumerate(jobs, 1):
        status, nbytes = fetch(f"{BASE}/{name}", dest, force)
        digest = sha256(dest) if dest.exists() and dest.stat().st_size > 0 else ""
        rows.append({"file": name, "dest": str(dest.relative_to(PKG)),
                     "status": status, "bytes": nbytes or (dest.stat().st_size if dest.exists() else 0),
                     "sha256": digest})
        n_dl += status == "downloaded"; n_skip += status == "skipped"; n_fail += status == "FAILED"
        if i % 50 == 0 or i == len(jobs):
            print(f"  [{i}/{len(jobs)}] dl={n_dl} skip={n_skip} fail={n_fail}")

    with open(LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "dest", "status", "bytes", "sha256"])
        w.writeheader(); w.writerows(rows)

    print(f"\nsource: {REPO}@{COMMIT[:10]}")
    print(f"masks dir: {MASK_DIR}")
    print(f"downloaded={n_dl} skipped={n_skip} FAILED={n_fail}; log -> {LOG.relative_to(PKG)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
