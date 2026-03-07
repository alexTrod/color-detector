#!/usr/bin/env python3
"""Populate dataset total/per-subject size columns in eeg_datasets_for_labram.csv."""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

OPENNEURO_GRAPHQL = "https://openneuro.org/crn/graphql"
CSV_PATH = Path(__file__).resolve().parents[1] / "eeg_datasets_for_labram.csv"
REPORT_PATH = Path(__file__).resolve().parents[1] / "data/manifests/dataset_size_report.json"


def format_bytes(n: int | None) -> str:
    if n is None:
        return "N/A"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return "N/A"


def parse_openneuro_id(url: str) -> str | None:
    m = re.search(r"/datasets/(ds\d+)", url)
    return m.group(1) if m else None


def openneuro_query(dataset_id: str, tree: str | None = None) -> dict:
    query = """
    query DatasetSizes($id: ID!, $tree: String) {
      dataset(id: $id) {
        id
        latestSnapshot {
          tag
          size
          files(tree: $tree) {
            filename
            size
            directory
          }
        }
      }
    }
    """
    r = requests.post(
        OPENNEURO_GRAPHQL,
        json={"query": query, "variables": {"id": dataset_id, "tree": tree}},
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("data", {}).get("dataset", {})


def openneuro_sizes(dataset_id: str) -> tuple[int | None, int | None, dict]:
    ds = openneuro_query(dataset_id)
    if not ds or not ds.get("latestSnapshot"):
        return None, None, {"error": "dataset/snapshot missing"}

    snapshot = ds["latestSnapshot"]
    total = int(snapshot["size"]) if snapshot.get("size") is not None else None

    root_files = snapshot.get("files", [])
    subject_dirs = [f["filename"] for f in root_files if f.get("directory") and str(f.get("filename", "")).startswith("sub-")]
    per_subject = defaultdict(int)
    for sub in subject_dirs:
        subtree = openneuro_query(dataset_id, tree=sub).get("latestSnapshot", {}).get("files", [])
        for f in subtree:
            if f.get("directory"):
                continue
            size = int(f["size"]) if f.get("size") is not None else 0
            per_subject[sub] += size

    avg = int(sum(per_subject.values()) / len(per_subject)) if per_subject else None
    details = {
        "snapshot_tag": snapshot.get("tag"),
        "subjects_detected": len(per_subject),
        "largest_subject": max(per_subject.items(), key=lambda kv: kv[1])[0] if per_subject else None,
    }
    return total, avg, details


def parse_osf_info(url: str) -> tuple[str | None, str | None]:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    node = parts[0] if parts else None
    token = parse_qs(parsed.query).get("view_only", [None])[0]
    return node, token


def walk_osf_files(node: str, token: str | None) -> list[dict]:
    params = {"page[size]": 100}
    if token:
        params["view_only"] = token
    root = requests.get(f"https://api.osf.io/v2/nodes/{node}/files/", params=params, timeout=60)
    root.raise_for_status()
    providers = root.json().get("data", [])
    storage = next((p for p in providers if p.get("attributes", {}).get("name") == "osfstorage"), None)
    if not storage:
        return []

    out = []
    stack = []
    next_url = storage.get("relationships", {}).get("files", {}).get("links", {}).get("related", {}).get("href")
    if token and next_url:
        sep = "&" if "?" in next_url else "?"
        next_url = f"{next_url}{sep}view_only={token}"
    if next_url:
        stack.append(next_url)

    visited_pages = set()
    while stack:
        page_url = stack.pop()
        while page_url:
            if page_url in visited_pages:
                break
            visited_pages.add(page_url)
            r = requests.get(page_url, timeout=60)
            r.raise_for_status()
            payload = r.json()
            for item in payload.get("data", []):
                attrs = item.get("attributes", {})
                kind = attrs.get("kind")
                if kind == "file":
                    out.append(item)
                elif kind == "folder":
                    related = item.get("relationships", {}).get("files", {}).get("links", {}).get("related", {}).get("href")
                    if related and token and "view_only=" not in related:
                        sep = "&" if "?" in related else "?"
                        related = f"{related}{sep}view_only={token}"
                    if related:
                        stack.append(related)
            page_url = payload.get("links", {}).get("next")
    return out


def osf_sizes(url: str) -> tuple[int | None, int | None, dict]:
    node, token = parse_osf_info(url)
    if not node:
        return None, None, {"error": "invalid osf URL"}
    files = walk_osf_files(node, token)
    if not files:
        return None, None, {"error": "no files listed"}

    total = 0
    per_subject = defaultdict(int)
    for f in files:
        attrs = f.get("attributes", {})
        size = attrs.get("size") or 0
        total += int(size)
        path = attrs.get("materialized_path") or ""
        m = re.search(r"(sub-[^/]+)", path)
        if m:
            per_subject[m.group(1)] += int(size)

    avg = int(sum(per_subject.values()) / len(per_subject)) if per_subject else None
    return total, avg, {"subjects_detected": len(per_subject), "files_count": len(files)}


def compute_sizes(data_link: str) -> tuple[str, str, dict]:
    try:
        dsid = parse_openneuro_id(data_link)
        if dsid:
            total, avg, details = openneuro_sizes(dsid)
            return format_bytes(total), format_bytes(avg), {"source": "openneuro", "id": dsid, **details}
        if "osf.io" in data_link:
            total, avg, details = osf_sizes(data_link)
            return format_bytes(total), format_bytes(avg), {"source": "osf", **details}
    except Exception as exc:  # noqa: BLE001
        return "N/A", "N/A", {"error": str(exc)}
    return "N/A", "N/A", {"source": "manual"}


def parse_subject_count(value: str) -> int | None:
    m = re.search(r"\d+", value or "")
    return int(m.group(0)) if m else None


def main() -> int:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    report = {}
    for row in rows:
        total, per_subject, details = compute_sizes(row.get("data link", ""))
        # OpenNeuro only exposes snapshot total size directly; use average-per-subject estimate.
        if per_subject == "N/A" and details.get("source") == "openneuro":
            n_subjects = parse_subject_count(row.get("subjects number", ""))
            if n_subjects:
                total_bytes = None
                m = re.match(r"([0-9.]+) (B|KiB|MiB|GiB|TiB)", total)
                if m:
                    value = float(m.group(1))
                    units = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}
                    total_bytes = int(value * units[m.group(2)])
                if total_bytes:
                    per_subject = format_bytes(int(total_bytes / n_subjects))
                    details["per_subject_method"] = "estimated_total_div_subject_count"
        row["dataset total size"] = total
        row["per subject size"] = per_subject
        report[row.get("paper name", "unknown")] = details

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Updated sizes in {CSV_PATH}")
    print(f"Wrote details to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
