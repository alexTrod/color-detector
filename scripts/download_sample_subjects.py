#!/usr/bin/env python3
"""Download one-subject samples across configured datasets."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import openneuro
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data/raw_samples"
MANIFEST = ROOT / "data/manifests/sample_download_manifest.json"
REGISTRY_CSV = ROOT / "eeg_datasets_for_labram.csv"
OPENNEURO_GRAPHQL = "https://openneuro.org/crn/graphql"


def openneuro_first_subject(dataset_id: str) -> str:
    query = """
    query($id: ID!) {
      dataset(id: $id) {
        latestSnapshot {
          files {
            filename
            directory
          }
        }
      }
    }
    """
    r = requests.post(OPENNEURO_GRAPHQL, json={"query": query, "variables": {"id": dataset_id}}, timeout=30)
    r.raise_for_status()
    files = r.json()["data"]["dataset"]["latestSnapshot"]["files"]
    subs = sorted([f["filename"] for f in files if f.get("directory") and f["filename"].startswith("sub-")])
    if not subs:
        raise RuntimeError(f"No subject folders found in {dataset_id}")
    return subs[0]


def parse_openneuro_id(url: str) -> str | None:
    m = re.search(r"/datasets/(ds\d+)", url)
    return m.group(1) if m else None


def configured_openneuro_datasets() -> list[str]:
    if not REGISTRY_CSV.exists():
        return ["ds005815", "ds004621"]
    df = pd.read_csv(REGISTRY_CSV)
    ids: list[str] = []
    for link in df.get("data link", pd.Series(dtype=str)).fillna("").astype(str):
        dsid = parse_openneuro_id(link)
        if dsid:
            ids.append(dsid)
    if not ids:
        return ["ds005815", "ds004621"]
    return sorted(set(ids))


def download_openneuro_yoto_five(subjects: list[str]) -> dict:
    """Download ds005815 for given subjects, task-task and rest1 EEG."""
    target = RAW_ROOT / "ds005815"
    target.mkdir(parents=True, exist_ok=True)
    include = ["dataset_description.json", "CHANGES", "README"] # removed  "participants.tsv", "participants.json"
    for sub in subjects:
        include.append(f"{sub}/**/eeg/*task-task*")
        include.append(f"{sub}/**/eeg/*rest1*")
    openneuro.download(
        dataset="ds005815",
        target_dir=target,
        include=include,
        verify_hash=False,
        max_concurrent_downloads=4,
    )
    files = [p for p in target.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    return {
        "dataset": "ds005815",
        "subjects": subjects,
        "target_dir": str(target),
        "files_downloaded": len(files),
        "bytes": total,
    }


def openneuro_subject_list(dataset_id: str, max_subjects: int | None = None) -> list[str]:
    """Return list of subject folder names (sub-01, sub-02, ...) for dataset."""
    query = """
    query($id: ID!) {
      dataset(id: $id) {
        latestSnapshot {
          files {
            filename
            directory
          }
        }
      }
    }
    """
    r = requests.post(OPENNEURO_GRAPHQL, json={"query": query, "variables": {"id": dataset_id}}, timeout=30)
    r.raise_for_status()
    files = r.json()["data"]["dataset"]["latestSnapshot"]["files"]
    subs = sorted([f["filename"] for f in files if f.get("directory") and f["filename"].startswith("sub-")])
    if max_subjects is not None:
        subs = subs[:max_subjects]
    print(f"Found {len(subs)} subjects in {dataset_id}")
    print(subs)
    return subs


def download_openneuro_subject(
    dataset_id: str,
    subjects: list[str] | None = None,
    task_glob: str | None = None,
) -> dict:
    if subjects is None:
        subjects = [openneuro_first_subject(dataset_id)]
    use_sample_dir = len(subjects) == 1 and task_glob is None and dataset_id != "ds005815"
    target = RAW_ROOT / (f"{dataset_id}_sample" if use_sample_dir else dataset_id)
    target.mkdir(parents=True, exist_ok=True)

    run_glob = {
        "ds005815": "*rest1*",
        "ds004621": "*oddball*",
    }.get(dataset_id, "*")
    if task_glob is not None:
        run_glob = task_glob
    include = ["dataset_description.json", "CHANGES", "README", ".tsv"] # removed participants.tsv and participants.json
    for sub in subjects:
        include.append(f"{sub}/**/eeg/{run_glob}")
    openneuro.download(
        dataset=dataset_id,
        target_dir=target,
        include=include,
        verify_hash=False,
        max_concurrent_downloads=4,
    )
    files = [p for p in target.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    return {
        "dataset": dataset_id,
        "subject": subjects[0] if len(subjects) == 1 else None,
        "subjects": subjects,
        "include_glob": run_glob,
        "target_dir": str(target),
        "files_downloaded": len(files),
        "bytes": total,
    }


def download_openneuro_subjects(
    dataset_ids: list[str] | None = None,
    task_glob: str | None = None,
    max_subjects: int | None = None,
    include_hajonides: bool = True,
    ignore_errors: bool = True,
) -> list[dict]:
    if dataset_ids is None:
        dataset_ids = configured_openneuro_datasets()
    downloads: list[dict] = []
    for dataset_id in dataset_ids:
        try:
            subs = None
            if max_subjects is not None:
                subs = openneuro_subject_list(dataset_id, max_subjects=max_subjects)
            downloads.append(download_openneuro_subject(dataset_id, subjects=subs, task_glob=task_glob))
        except Exception as exc:  # noqa: BLE001
            if not ignore_errors:
                raise
            downloads.append({"dataset": dataset_id, "error": str(exc)})
    if include_hajonides:
        try:
            downloads.append(download_hajonides_subject())
        except Exception as exc:  # noqa: BLE001
            if not ignore_errors:
                raise
            downloads.append({"dataset": "hajonides_j289e", "error": str(exc)})
    return downloads


def download_yoto_five(
    max_subjects: int = 5,
    skip_other: bool = False,
    ignore_errors: bool = True,
) -> list[dict]:
    downloads: list[dict] = []
    subs = openneuro_subject_list("ds005815", max_subjects=max_subjects)
    try:
        downloads.append(download_openneuro_yoto_five(subs))
    except Exception as exc:  # noqa: BLE001
        if not ignore_errors:
            raise
        downloads.append({"dataset": "ds005815", "error": str(exc)})
    if not skip_other:
        other_ids = [d for d in configured_openneuro_datasets() if d != "ds005815"]
        downloads.extend(download_openneuro_subjects(dataset_ids=other_ids, ignore_errors=ignore_errors))
    return downloads


def parse_osf_info(url: str) -> tuple[str, str | None]:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    node = parts[0] if parts else None
    token = parse_qs(parsed.query).get("view_only", [None])[0]
    if not node:
        raise RuntimeError(f"Could not parse OSF node from {url}")
    return node, token


def osf_files(node: str, token: str | None) -> list[dict]:
    params = {"view_only": token} if token else None
    r = requests.get(f"https://api.osf.io/v2/nodes/{node}/files/", params=params, timeout=30)
    r.raise_for_status()
    providers = r.json().get("data", [])
    storage = next((p for p in providers if p.get("attributes", {}).get("name") == "osfstorage"), None)
    if not storage:
        return []
    url = storage["relationships"]["files"]["links"]["related"]["href"]
    if token and "view_only=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}view_only={token}"

    out, stack, visited = [], [url], set()
    while stack:
        page = stack.pop()
        while page:
            if page in visited:
                break
            visited.add(page)
            resp = requests.get(page, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            for item in payload.get("data", []):
                attrs = item.get("attributes", {})
                if attrs.get("kind") == "file":
                    out.append(item)
                elif attrs.get("kind") == "folder":
                    rel = item.get("relationships", {}).get("files", {}).get("links", {}).get("related", {}).get("href")
                    if rel and token and "view_only=" not in rel:
                        sep = "&" if "?" in rel else "?"
                        rel = f"{rel}{sep}view_only={token}"
                    if rel:
                        stack.append(rel)
            page = payload.get("links", {}).get("next")
    return out


def download_hajonides_subject() -> dict:
    osf_url = "https://osf.io/j289e/?view_only=b13407009b4245f7950960c34a5474a6"
    node, token = parse_osf_info(osf_url)
    files = osf_files(node, token)
    if not files:
        raise RuntimeError("No files discoverable in Hajonides OSF node")

    # Prefer explicit subject file; else first .mat file.
    candidate = None
    for f in files:
        path = f["attributes"].get("materialized_path", "")
        if re.search(r"sub[-_ ]?\d+", path, flags=re.IGNORECASE):
            candidate = f
            break
    if candidate is None:
        candidate = next((f for f in files if str(f["attributes"].get("name", "")).endswith(".mat")), files[0])

    download_url = candidate["links"]["download"]
    name = candidate["attributes"]["name"]
    target = RAW_ROOT / "hajonides_j289e"
    target.mkdir(parents=True, exist_ok=True)
    out_path = target / name

    with requests.get(download_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return {
        "dataset": "hajonides_j289e",
        "subject": "single_file_proxy",
        "source_file": candidate["attributes"].get("materialized_path"),
        "target_file": str(out_path),
        "bytes": out_path.stat().st_size,
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Download sample or YOTO 5-subject EEG data.")
    parser.add_argument("--yoto-five", action="store_true", help="Download ds005815 for 5 subjects with task-task (and rest1).")
    parser.add_argument("--skip-other", action="store_true", help="When using --yoto-five, skip other datasets (ds004621, Hajonides).")
    args = parser.parse_args()

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    downloads: list[dict] = []

    if args.yoto_five:
        downloads.extend(download_yoto_five(skip_other=args.skip_other))
    else:
        downloads.extend(download_openneuro_subjects())

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"downloads": downloads}
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
