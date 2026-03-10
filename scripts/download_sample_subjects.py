#!/usr/bin/env python3
"""Download one-subject samples across configured datasets."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import openneuro
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data/raw_samples"
MANIFEST = ROOT / "data/manifests/sample_download_manifest.json"
REGISTRY_CSV = ROOT / "eeg_datasets_for_labram.csv"
OPENNEURO_GRAPHQL = "https://openneuro.org/crn/graphql"
DEFAULT_USB_ROOT = Path("/Volumes/DISK_IMG")
KNOWN_YOTO_SUBJECTS = [
    "sub-01", "sub-02", "sub-05", "sub-07", "sub-08",
    "sub-09", "sub-10", "sub-11", "sub-12", "sub-13",
    "sub-14", "sub-16", "sub-18", "sub-19", "sub-21",
    "sub-22", "sub-23", "sub-24", "sub-25", "sub-26",
]


def _requests_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"Connection": "close"})
    return session


def _openneuro_graphql_files(dataset_id: str) -> list[dict]:
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
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with _requests_session() as session:
                r = session.post(
                    OPENNEURO_GRAPHQL,
                    json={"query": query, "variables": {"id": dataset_id}},
                    timeout=30,
                )
            r.raise_for_status()
            files = r.json()["data"]["dataset"]["latestSnapshot"]["files"]
            return files
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < 3:
                time.sleep(attempt)
    raise RuntimeError(f"OpenNeuro GraphQL lookup failed for {dataset_id}: {last_error}") from last_error


def openneuro_first_subject(dataset_id: str) -> str:
    files = _openneuro_graphql_files(dataset_id)
    subs = sorted([f["filename"] for f in files if f.get("directory") and f["filename"].startswith("sub-")])
    if not subs:
        raise RuntimeError(f"No subject folders found in {dataset_id}")
    return subs[0]


def yoto_subjects() -> list[str]:
    try:
        return openneuro_subject_list("ds005815", max_subjects=None)
    except Exception as exc:  # noqa: BLE001
        print(f"Falling back to built-in YOTO subject list: {exc}")
        return KNOWN_YOTO_SUBJECTS.copy()


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


def build_yoto_task_include(subjects: list[str] | None = None) -> list[str]:
    include = [
        "dataset_description.json",
        "CHANGES",
        "derivatives/**/task_event.mat",
    ]
    subject_patterns = subjects if subjects is not None else ["sub-*"]
    for sub in subject_patterns:
        include.append(f"{sub}/**/eeg/*task-task*")
    return include


def download_openneuro_yoto_subjects(subjects: list[str] | None, raw_root: Path = RAW_ROOT) -> dict:
    """Download ds005815 task-task raw EEG plus matching derivatives for given subjects."""
    target = raw_root / "ds005815"
    target.mkdir(parents=True, exist_ok=True)
    include = build_yoto_task_include(subjects)
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
        "include": include,
        "target_dir": str(target),
        "files_downloaded": len(files),
        "bytes": total,
    }


def openneuro_subject_list(dataset_id: str, max_subjects: int | None = None) -> list[str]:
    """Return list of subject folder names (sub-01, sub-02, ...) for dataset."""
    files = _openneuro_graphql_files(dataset_id)
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
    raw_root: Path = RAW_ROOT,
) -> dict:
    if subjects is None:
        subjects = [openneuro_first_subject(dataset_id)]
    use_sample_dir = len(subjects) == 1 and task_glob is None and dataset_id != "ds005815"
    target = raw_root / (f"{dataset_id}_sample" if use_sample_dir else dataset_id)
    target.mkdir(parents=True, exist_ok=True)

    run_glob = {
        "ds005815": "*rest1*",
        "ds004621": "*oddball*",
    }.get(dataset_id, "*")
    if task_glob is not None:
        run_glob = task_glob
    include: list[str] = []
    for sub in subjects:
        # Keep this broad for cross-dataset compatibility (some datasets are not laid out
        # under sub-*/**/eeg with predictable task naming).
        include.append(f"{sub}/**")
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


def download_hajonides_subject(raw_root: Path = RAW_ROOT) -> dict:
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
    target = raw_root / "hajonides_j289e"
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


def write_download_manifest(
    downloads: list[dict],
    manifest_path: Path = MANIFEST,
) -> dict:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"downloads": downloads}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    return {"n_downloads": len(downloads), "manifest": str(manifest_path)}


def download_yoto_five(
    max_subjects: int = 5,
    raw_root: Path = RAW_ROOT,
) -> dict:
    subs = yoto_subjects()[:max_subjects]
    return download_openneuro_yoto_subjects(subs, raw_root=raw_root)


def download_yoto_all(
    raw_root: Path = RAW_ROOT,
) -> dict:
    return download_openneuro_yoto_subjects(None, raw_root=raw_root)


def download_yoto_batch(
    batch_size: int = 5,
    batch_index: int = 0,
    raw_root: Path = RAW_ROOT,
) -> dict:
    subjects = yoto_subjects()
    start = batch_index * batch_size
    end = start + batch_size
    batch_subjects = subjects[start:end]
    if not batch_subjects:
        raise RuntimeError(
            f"No YOTO subjects found for batch_index={batch_index} with batch_size={batch_size}. "
            f"Available subjects: {len(subjects)}"
        )
    return download_openneuro_yoto_subjects(batch_subjects, raw_root=raw_root)


def download_openneuro_subjects(
    dataset_ids: list[str],
    max_subjects: int | None = 1,
    task_globs: dict[str, str] | None = None,
    raw_root: Path = RAW_ROOT,
) -> list[dict]:
    results = []
    task_globs = task_globs or {}
    for dataset_id in dataset_ids:
        subjects = None
        if max_subjects is not None:
            subjects = openneuro_subject_list(dataset_id, max_subjects=max_subjects)
        try:
            results.append(
                download_openneuro_subject(
                    dataset_id,
                    subjects=subjects,
                    task_glob=task_globs.get(dataset_id),
                    raw_root=raw_root,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"dataset": dataset_id, "error": str(exc)})
    return results


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download sample or YOTO EEG data.")
    parser.add_argument("--yoto-five", action="store_true", help="Download ds005815 for 5 subjects with task-task files only.")
    parser.add_argument("--yoto-all", action="store_true", help="Download ds005815 for all available subjects with task-task files only.")
    parser.add_argument("--yoto-batch", action="store_true", help="Download one YOTO batch of subjects.")
    parser.add_argument("--yoto-batch-size", type=int, default=5, help="Number of YOTO subjects per batch.")
    parser.add_argument("--yoto-batch-index", type=int, default=0, help="Zero-based YOTO batch index.")
    parser.add_argument("--skip-other", action="store_true", help="When using --yoto-five, skip other datasets (ds004621, Hajonides).")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_ROOT,
        help=f"Root directory for downloads, e.g. {DEFAULT_USB_ROOT}",
    )
    parser.add_argument("--dataset-id", type=str, default="", help="OpenNeuro dataset ID to download.")
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=1,
        help="Max subjects for --dataset-id (set to 0 for all).",
    )
    parser.add_argument(
        "--task-glob",
        type=str,
        default="",
        help="Glob to filter EEG files for --dataset-id (e.g. '*task-*').",
    )
    args = parser.parse_args()

    raw_root = args.raw_root.expanduser()
    raw_root.mkdir(parents=True, exist_ok=True)
    downloads: list[dict] = []

    if args.dataset_id:
        max_subjects = None if args.max_subjects == 0 else int(args.max_subjects)
        subjects = openneuro_subject_list(args.dataset_id, max_subjects=max_subjects)
        task_glob = args.task_glob if args.task_glob else None
        downloads.append(
            download_openneuro_subject(
                args.dataset_id,
                subjects=subjects,
                task_glob=task_glob,
                raw_root=raw_root,
            )
        )
        write_download_manifest(downloads)
        return 0

    if args.yoto_five or args.yoto_all or args.yoto_batch:
        try:
            if args.yoto_batch:
                downloads.append(
                    download_yoto_batch(
                        batch_size=args.yoto_batch_size,
                        batch_index=args.yoto_batch_index,
                        raw_root=raw_root,
                    )
                )
            elif args.yoto_all:
                downloads.append(download_yoto_all(raw_root=raw_root))
            else:
                downloads.append(download_yoto_five(raw_root=raw_root))
        except Exception as exc:  # noqa: BLE001
            downloads.append({"dataset": "ds005815", "error": str(exc)})
        if not args.skip_other:
            other_ids = [d for d in configured_openneuro_datasets() if d != "ds005815"]
            downloads.extend(download_openneuro_subjects(other_ids, raw_root=raw_root))
            try:
                downloads.append(download_hajonides_subject(raw_root=raw_root))
            except Exception as exc:  # noqa: BLE001
                downloads.append({"dataset": "hajonides_j289e", "error": str(exc)})
    else:
        downloads.extend(download_openneuro_subjects(configured_openneuro_datasets(), raw_root=raw_root))
        try:
            downloads.append(download_hajonides_subject(raw_root=raw_root))
        except Exception as exc:  # noqa: BLE001
            downloads.append({"dataset": "hajonides_j289e", "error": str(exc)})

    write_download_manifest(downloads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
