"""CT-RATE download, cleaning, de-duplication and split construction.

CT-RATE (Hamamci et al., 2024) ships one row per reconstructed *volume*; a single
CT scan usually has several reconstructions that share the same report. We
collapse to one row per scan, keep one scan per patient, and build:

  train.jsonl  — from CT-RATE train patients (LoRA training)
  valid.jsonl  — from CT-RATE train patients, disjoint from train.jsonl (loss monitoring)
  test.jsonl   — from CT-RATE *validation* patients (held-out evaluation; never seen)

Each JSONL row carries the raw fields (id, patient, report, labels[, evidence]) plus
`messages` in the OpenAI chat format that mlx_lm.lora consumes directly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .prompts import build_messages
from .schema import LABELS, target_json

REPO_ID = "ibrahimhamamci/CT-RATE"
FILES = {
    "train_reports": "dataset/radiology_text_reports/train_reports.csv",
    "valid_reports": "dataset/radiology_text_reports/validation_reports.csv",
    "train_labels": "dataset/multi_abnormality_labels/train_predicted_labels.csv",
    "valid_labels": "dataset/multi_abnormality_labels/valid_predicted_labels.csv",
}

_VOL_RE = re.compile(r"^(?P<patient>(?:train|valid)_\d+)_(?P<scan>[a-z]+)_(?P<recon>\d+)\.nii\.gz$")


def download(raw_dir: str | Path, token: str | None = None) -> dict[str, Path]:
    """Download the four CSVs (≈90 MB) into raw_dir. Requires accepted dataset terms + HF login."""
    from huggingface_hub import hf_hub_download

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, fname in FILES.items():
        p = hf_hub_download(
            repo_id=REPO_ID, filename=fname, repo_type="dataset", local_dir=raw_dir, token=token
        )
        out[key] = Path(p)
    return out


def parse_volume_name(name: str) -> tuple[str, str, int]:
    m = _VOL_RE.match(str(name).strip())
    if not m:
        raise ValueError(f"Unexpected VolumeName: {name!r}")
    patient = m["patient"]
    return patient, f"{patient}_{m['scan']}", int(m["recon"])


def _clean(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()


def compose_report(findings: str, impression: str) -> str:
    parts = []
    if findings:
        parts.append(f"FINDINGS: {findings}")
    if impression:
        parts.append(f"IMPRESSION: {impression}")
    return "\n".join(parts)


def load_split(reports_csv: str | Path, labels_csv: str | Path, min_chars: int = 40) -> pd.DataFrame:
    """Merge reports with labels, collapse reconstructions, one row per scan."""
    rep = pd.read_csv(reports_csv)
    lab = pd.read_csv(labels_csv)
    missing = [c for c in LABELS if c not in lab.columns]
    if missing:
        raise ValueError(f"Label CSV is missing columns: {missing}")
    df = rep.merge(lab[["VolumeName", *LABELS]], on="VolumeName", how="inner")

    parsed = df["VolumeName"].map(parse_volume_name)
    df["patient"] = [p[0] for p in parsed]
    df["scan_id"] = [p[1] for p in parsed]
    df["recon"] = [p[2] for p in parsed]
    df = df.sort_values(["scan_id", "recon"]).drop_duplicates("scan_id", keep="first")

    df["findings"] = df["Findings_EN"].map(_clean)
    df["impression"] = df["Impressions_EN"].map(_clean) if "Impressions_EN" in df else ""
    df["report"] = [compose_report(f, i) for f, i in zip(df["findings"], df["impression"])]
    df = df[df["findings"].str.len() >= min_chars].copy()
    df[LABELS] = df[LABELS].fillna(0).astype(int)
    return df.reset_index(drop=True)


def one_scan_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("scan_id").drop_duplicates("patient", keep="first").reset_index(drop=True)


def to_records(df: pd.DataFrame) -> list[dict]:
    recs = []
    for d in df.to_dict("records"):  # label names contain spaces -> no itertuples
        recs.append(
            {
                "id": d["scan_id"],
                "patient": d["patient"],
                "report": d["report"],
                "labels": {k: int(d[k]) for k in LABELS},
            }
        )
    return recs


def make_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    n_train: int,
    n_valid: int,
    n_test: int,
    seed: int = 42,
) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed)
    tr = one_scan_per_patient(train_df)
    te = one_scan_per_patient(valid_df)
    tr_idx = rng.permutation(len(tr))
    te_idx = rng.permutation(len(te))
    if n_train + n_valid > len(tr):
        raise ValueError(f"Requested {n_train}+{n_valid} but only {len(tr)} train patients")
    train = tr.iloc[tr_idx[:n_train]]
    valid = tr.iloc[tr_idx[n_train : n_train + n_valid]]
    test = te.iloc[te_idx[: min(n_test, len(te))]]
    return {"train": to_records(train), "valid": to_records(valid), "test": to_records(test)}


def attach_messages(records: list[dict], with_target: bool) -> None:
    """Add the chat `messages` field. Test rows get no assistant turn."""
    for rec in records:
        if with_target:
            tgt = target_json(rec["labels"], rec.get("evidence"))
            rec["target"] = tgt
            rec["messages"] = build_messages(rec["report"], target=tgt)
        else:
            rec["messages"] = build_messages(rec["report"])


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def label_prevalence(records: list[dict]) -> dict[str, float]:
    n = max(len(records), 1)
    return {k: round(sum(r["labels"].get(k, 0) for r in records) / n, 4) for k in LABELS}
