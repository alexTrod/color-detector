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
    include = ["dataset_description.json", "CHANGES", "README", ".tsv"] # removed participants.tsv and participants.json
    for sub in subjects:
        include.append(f"{sub}/**/eeg/*task-task*")
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


# Known OSF folder IDs for direct listing (faster than full tree walk)
HAJONIDES_PREPROCESSED_FOLDER_ID = "5f16f2b10596f60152798b83"
BAELUCK_EEG_FOLDER_ID = "660f5395bba39a1cc1729f1c"


def osf_list_folder(
    node: str,
    folder_id: str,
    token: str | None = None,
) -> list[dict]:
    """List files in a single OSF folder (no recursion). Handles pagination."""
    base = f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/{folder_id}/"
    if token:
        url = f"{base}?view_only={token}"
    else:
        url = base
    out = []
    while url:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        for item in payload.get("data", []):
            out.append(item)
        url = payload.get("links", {}).get("next")
        if url and token and "view_only=" not in url:
            url = f"{url}&view_only={token}" if "?" in url else f"{url}?view_only={token}"
    return out


def _download_osf_file(download_url: str, out_path: Path, token: str | None = None) -> None:
    url = download_url
    if token and "view_only=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}view_only={token}"
    with _requests_session() as session:
        with session.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)


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
    remote_size = candidate["attributes"].get("size")
    if out_path.exists() and (remote_size is None or out_path.stat().st_size == remote_size):
        return {
            "dataset": "hajonides_j289e",
            "subject": "single_file_proxy",
            "source_file": candidate["attributes"].get("materialized_path"),
            "target_file": str(out_path),
            "bytes": out_path.stat().st_size,
            "skipped": "already_downloaded",
        }
    _download_osf_file(download_url, out_path, token)

    return {
        "dataset": "hajonides_j289e",
        "subject": "single_file_proxy",
        "source_file": candidate["attributes"].get("materialized_path"),
        "target_file": str(out_path),
        "bytes": out_path.stat().st_size,
    }


def download_hajonides_color(
    max_subjects: int = 10,
    raw_root: Path = RAW_ROOT,
) -> dict:
    """Download up to max_subjects Hajonides (j289e) preprocessed .mat files (color/hue epochs)."""
    osf_url = "https://osf.io/j289e/?view_only=b13407009b4245f7950960c34a5474a6"
    node, token = parse_osf_info(osf_url)
    print("    Listing Hajonides OSF files...", flush=True)
    files = osf_list_folder(node, HAJONIDES_PREPROCESSED_FOLDER_ID, token)
    mat_files = [
        f for f in files
        if f.get("attributes", {}).get("kind") == "file"
        and str(f["attributes"].get("name", "")).endswith(".mat")
    ]
    mat_files.sort(key=lambda x: x["attributes"].get("name", ""))
    to_download = mat_files[:max_subjects]
    target = raw_root / "hajonides_j289e"
    target.mkdir(parents=True, exist_ok=True)
    downloaded = []
    total_bytes = 0
    for i, f in enumerate(to_download):
        name = f["attributes"]["name"]
        out_path = target / name
        remote_size = f["attributes"].get("size")
        if out_path.exists() and (remote_size is None or out_path.stat().st_size == remote_size):
            downloaded.append(str(out_path))
            total_bytes += out_path.stat().st_size
            continue
        print(f"    Downloading {i+1}/{len(to_download)} {name}...", flush=True)
        _download_osf_file(f["links"]["download"], out_path, token)
        downloaded.append(str(out_path))
        total_bytes += out_path.stat().st_size
    return {
        "dataset": "hajonides_j289e",
        "subjects": [re.search(r"S(\d+)", Path(p).name, re.I).group(0) if re.search(r"S(\d+)", Path(p).name, re.I) else Path(p).stem for p in downloaded],
        "files_downloaded": len(downloaded),
        "target_dir": str(target),
        "bytes": total_bytes,
    }


def download_baeluck_color(
    max_subjects: int = 10,
    raw_root: Path = RAW_ROOT,
) -> dict:
    """Download up to max_subjects Bae & Luck (jnwut) color EEG .mat files. One file per subject/condition."""
    node = "jnwut"
    token = None
    print("    Listing Bae & Luck OSF files...", flush=True)
    files = osf_list_folder(node, BAELUCK_EEG_FOLDER_ID, token)
    mat_files = [
        f for f in files
        if f.get("attributes", {}).get("kind") == "file"
        and str(f["attributes"].get("name", "")).endswith(".mat")
    ]
    # Extract subject id from name (e.g. Color_NoRotation_OSF_107.mat -> 107)
    def subj_id(attr):
        name = attr.get("name", "")
        m = re.search(r"_(\d+)\.mat$", name)
        return int(m.group(1)) if m else 0
    seen = set()
    selected = []
    for f in sorted(mat_files, key=lambda x: subj_id(x["attributes"])):
        sid = subj_id(f["attributes"])
        if sid and sid not in seen and len(seen) < max_subjects:
            seen.add(sid)
            selected.append(f)
    target = raw_root / "baeluck_jnwut"
    target.mkdir(parents=True, exist_ok=True)
    downloaded = []
    total_bytes = 0
    for i, f in enumerate(selected):
        name = f["attributes"]["name"]
        out_path = target / name
        remote_size = f["attributes"].get("size")
        if out_path.exists() and (remote_size is None or out_path.stat().st_size == remote_size):
            downloaded.append(str(out_path))
            total_bytes += out_path.stat().st_size
            continue
        print(f"    Downloading {i+1}/{len(selected)} {name}...", flush=True)
        _download_osf_file(f["links"]["download"], out_path, token)
        downloaded.append(str(out_path))
        total_bytes += out_path.stat().st_size
    return {
        "dataset": "baeluck_jnwut",
        "subjects": list(seen),
        "files_downloaded": len(downloaded),
        "target_dir": str(target),
        "bytes": total_bytes,
    }


def download_chauhan_color(
    max_subjects: int = 10,
    raw_root: Path = RAW_ROOT,
) -> dict:
    """Download Chauhan et al. (v9ewj) color hue EEG if available on OSF osfstorage."""
    node = "v9ewj"
    token = None
    files = osf_files(node, token)
    mat_or_eeg = [
        f for f in files
        if f.get("attributes", {}).get("kind") == "file"
        and (
            str(f["attributes"].get("name", "")).endswith(".mat")
            or str(f["attributes"].get("name", "")).endswith(".set")
            or str(f["attributes"].get("name", "")).endswith(".bdf")
        )
    ]
    mat_or_eeg = mat_or_eeg[:max_subjects]
    if not mat_or_eeg:
        return {"dataset": "chauhan_v9ewj", "error": "No EEG/mat files found in OSF osfstorage", "files_downloaded": 0}
    target = raw_root / "chauhan_v9ewj"
    target.mkdir(parents=True, exist_ok=True)
    downloaded = []
    total_bytes = 0
    for f in mat_or_eeg:
        name = f["attributes"]["name"]
        out_path = target / name
        remote_size = f["attributes"].get("size")
        if out_path.exists() and (remote_size is None or out_path.stat().st_size == remote_size):
            downloaded.append(str(out_path))
            total_bytes += out_path.stat().st_size
            continue
        _download_osf_file(f["links"]["download"], out_path, token)
        downloaded.append(str(out_path))
        total_bytes += out_path.stat().st_size
    return {
        "dataset": "chauhan_v9ewj",
        "files_downloaded": len(downloaded),
        "target_dir": str(target),
        "bytes": total_bytes,
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
    parser.add_argument("--color", action="store_true", help="Download color EEG datasets: Hajonides, Bae & Luck, Chauhan (max 10 subjects each).")
    parser.add_argument("--color-max-subjects", type=int, default=10, help="Max subjects per color dataset (default 10).")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_ROOT,
        help=f"Root directory for downloads, e.g. {DEFAULT_USB_ROOT}",
    )
    args = parser.parse_args()

    raw_root = args.raw_root.expanduser()
    raw_root.mkdir(parents=True, exist_ok=True)
    downloads: list[dict] = []

    if args.color:
        print(f"Downloading color datasets (max {args.color_max_subjects} subjects each)...", flush=True)
        for fn in (download_hajonides_color, download_baeluck_color, download_chauhan_color):
            name = fn.__name__.replace("download_", "").replace("_color", "")
            print(f"  {name}...", flush=True)
            try:
                res = fn(max_subjects=args.color_max_subjects, raw_root=raw_root)
                downloads.append(res)
                if res.get("error"):
                    print(f"Warning: {res.get('dataset', '')} {res.get('error', '')}", flush=True)
                else:
                    print(f"    -> {res.get('files_downloaded', 0)} files", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"    Error: {exc}", flush=True)
                downloads.append({"dataset": name, "error": str(exc)})
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
