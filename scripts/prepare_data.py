"""Fetch versioned public inputs and build the manuscript's feature tables."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "arc_lesion_image_benchmark"
DATA = PKG / "data"
DS = ROOT / "datasets/arc_ds004884"
DEMO = "neurolabusc/AphasiaRecoveryCohortDemo"
DEMO_COMMIT = "1d24ad4049f4dba536d51f5f80efdb0900393ac0"
OPENNEURO_COMMIT = "0885e5939abc8f909a175dd782369b7afc3fdd08"  # ds004884 v1.0.2
NEUROPARC_COMMIT = "5a5e7469671e65cb58087c47b69d0edb71dc2966"


def fetch(url, path, sha256=None, git_sha=None):
    path.parent.mkdir(parents=True, exist_ok=True)

    def valid(content):
        if sha256 and hashlib.sha256(content).hexdigest() != sha256:
            return False
        if git_sha:
            blob = f"blob {len(content)}\0".encode() + content
            return hashlib.sha1(blob).hexdigest() == git_sha
        return True

    if path.exists() and valid(path.read_bytes()):
        return path
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "ARC-2026-reproduction"})
            with urlopen(request, timeout=90) as response:
                content = response.read()
            if not valid(content):
                raise ValueError(f"Checksum mismatch: {path.name}")
            temporary = path.with_name(path.name + ".part")
            temporary.write_bytes(content)
            temporary.replace(path)
            return path
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def tree(repository, commit, name):
    path = fetch(f"https://api.github.com/repos/{repository}/git/trees/{commit}?recursive=1",
                 DS / "source_metadata" / name)
    result = json.loads(path.read_text())
    if result.get("truncated") or result.get("sha") != commit:
        raise ValueError(f"Incomplete or unexpected source tree: {repository}")
    return result["tree"]


def run(script, *args):
    subprocess.run([sys.executable, str(PKG / "src" / script), *args], check=True)


def prepare(extra_atlases=False):
    demo = tree(DEMO, DEMO_COMMIT, "arc_demo.json")
    masks = [x for x in demo if x["path"].startswith("NIfTI/wsub-")
             and x["path"].endswith("_lesion.nii.gz")]
    assert len(masks) == 228
    atlas_names = {"jhu156.nii.gz", "jhu156.txt", "ArterialAtlas156.nii.gz", "ArterialAtlas156.txt"}
    atlases = [x for x in demo if x["path"].startswith("NIfTI/")
               and Path(x["path"]).name in atlas_names]
    assert len(atlases) == 4

    def download(entry):
        folder = "lesion_masks_mni" if entry in masks else "atlases"
        return fetch(f"https://raw.githubusercontent.com/{DEMO}/{DEMO_COMMIT}/{entry['path']}",
                     DATA / "raw" / folder / Path(entry["path"]).name, git_sha=entry["sha"])

    print("Preparing 228 normalized lesion masks and two atlases...", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for index, _ in enumerate(pool.map(download, masks + atlases), 1):
            if index % 40 == 0:
                print(f"  {index}/{len(masks) + len(atlases)} source files", flush=True)

    fetch(f"https://raw.githubusercontent.com/{DEMO}/{DEMO_COMMIT}/Scripts/merged_artery_participants.tsv",
          DS / "derived_features_arc_demo/merged_artery_participants.tsv",
          sha256="794feaa81184ea2b9d94f2a29af4ee5c28703133aa84780543f654b95902b35a")
    fetch("https://raw.githubusercontent.com/OpenNeuroDatasets/ds004884/1.0.2/participants.tsv",
          DS / "participants.tsv",
          sha256="37a8e84577bbb5a8af8cb8a95b494473a939fed2fccbb35578262a620b1e184e")
    mat_path = fetch("https://ndownloader.figshare.com/files/41364972",
                     DS / "teghipco_figshare/DLAphasiaSeverityARCSubset.mat",
                     sha256="2d633a06a5e9329f1826fb8a2d226ed2698a2f259a9c252b17b89a6f60c17911")
    entries = tree("OpenNeuroDatasets/ds004884", OPENNEURO_COMMIT, "openneuro.json")
    subjects = sorted(x["path"] for x in entries if x["type"] == "tree"
                      and "/" not in x["path"] and x["path"].startswith("sub-"))
    rows = {sid: dict(participant_id=sid, anat=0, dwi=0, rest_fmri=0, task_fmri=0, flair=0)
            for sid in subjects}
    for entry in entries:
        path = entry["path"]
        sid = path.split("/")[0]
        if sid not in rows or not path.endswith(".nii.gz"):
            continue
        row = rows[sid]
        if "/anat/" in path:
            row["anat"] = 1
        if path.endswith("_dwi.nii.gz"):
            row["dwi"] = 1
        if "/func/" in path and path.endswith("_bold.nii.gz"):
            row["rest_fmri" if "task-rest" in path else "task_fmri"] = 1
        if path.endswith("_FLAIR.nii.gz"):
            row["flair"] = 1
    pd.DataFrame(rows.values()).to_csv(DATA / "arc_modality_availability.tsv", sep="\t", index=False)

    for atlas in ["jhu156", "ArterialAtlas156"]:
        run("s2_extract_atlas_features.py", "--atlas", atlas)
    features = pd.read_csv(DATA / "jhu156_features_merged.tsv", sep="\t")
    floor = features[["participant_id", "lesion_volume", "age_at_stroke", "wab_aq"]]
    for name in ["floor_volume_age.tsv", "floor_volage_features.tsv"]:
        floor.to_csv(DATA / name, sep="\t", index=False)
    mat = loadmat(mat_path, squeeze_me=True)
    subjects = [str(s) for s in np.asarray(mat["subsOnly"]).ravel()]
    labels = pd.DataFrame({"participant_id": [s if s.startswith("sub-") else "sub-" + s for s in subjects],
                           "subsOnly_raw": subjects, "wabClass2i": mat["wabClass2i"],
                           "wabClassi": mat["wabClassi"]})
    labels = labels.merge(floor, on="participant_id", how="inner")
    labels["our_severe"] = (labels.wab_aq < 50).astype(int)
    summary = PKG / "results/summary"
    summary.mkdir(parents=True, exist_ok=True)
    labels.to_csv(summary / "teghipco_label_crosscheck.csv", index=False)
    run("s17_freeze_pool_subsets.py")

    if extra_atlases:
        sources = [
            ("AAL", "89d3dc4ddccbd59483b89f3c47eaacc9353ad1b7e78e9f5514e08aaddcd425c9"),
            ("Brodmann", "ddf120fbd5bb1fd119c318bac5d8a825f18a55bb58c48229eb2f2d1c6c01cca9"),
            ("AICHAJoliot2015", "5573fcfead26bd6f744dfa930d38a8568cb76e1e8f6787279a7450642111a3fe")]
        for name, digest in sources:
            url = (f"https://raw.githubusercontent.com/neurodata/neuroparc/{NEUROPARC_COMMIT}/"
                   f"atlases/label/Human/{name}_space-MNI152NLin6_res-1x1x1.nii.gz")
            fetch(url, DATA / "raw/atlases" / f"src_{name}.nii.gz", sha256=digest)
        run("s23_resample_atlases.py")
        for atlas in ["aal116", "brodmann41", "aicha384"]:
            run("s2_extract_atlas_features.py", "--atlas", atlas)
    print("Prepared feature tables and 10 cohort definitions.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-atlases", action="store_true")
    prepare(parser.parse_args().extra_atlases)
