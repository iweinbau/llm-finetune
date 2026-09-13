"""Human audit of "supported" evidence.

The mechanical check proves a quote exists in the report; it cannot prove the
quote *means* the abnormality is present ("no pleural effusion" quoted for
Pleural effusion would pass). This module samples predicted-positive flags whose
quote passed the mechanical check and produces a CSV for a clinician to judge.

  llm-finetune audit make   -> results/audit_<arm>.csv   (fill the `verdict` column: yes / no / partial)
  llm-finetune audit score  -> adds human_supported_rate to results.json

Judging 50–100 rows takes under an hour and turns a proxy metric into a measured one.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from .data import read_jsonl
from .evaluate import load_predictions
from .metrics import evidence_status
from .parsing import parse_prediction
from .schema import LABELS


def make_audit_sheet(test_path: str | Path, pred_path: str | Path, out_csv: str | Path, n: int = 60, seed: int = 11) -> int:
    test = {r["id"]: r for r in read_jsonl(test_path)}
    arm, preds = load_predictions(pred_path)
    rows = []
    for rid, p in preds.items():
        rec = test.get(rid)
        if not rec:
            continue
        pred = parse_prediction(p["raw"])
        for k in LABELS:
            if pred.abnormalities.get(k):
                q = pred.evidence.get(k)
                if evidence_status(q, rec["report"]) == "supported":
                    rows.append({"id": rid, "arm": arm, "label": k, "gold": rec["labels"].get(k, 0), "quote": q, "report": rec["report"]})
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "arm", "label", "gold", "quote", "verdict", "note", "report"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "verdict": "", "note": ""})
    print(f"wrote {out_csv} with {len(rows)} rows — fill `verdict` with yes / no / partial")
    return len(rows)


def score_audit(audit_csv: str | Path, results_json: str | Path) -> dict:
    with Path(audit_csv).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    judged = [r for r in rows if r.get("verdict", "").strip()]
    if not judged:
        raise SystemExit("no verdicts filled in yet")
    arm = judged[0]["arm"]
    yes = sum(1 for r in judged if r["verdict"].strip().lower() == "yes")
    partial = sum(1 for r in judged if r["verdict"].strip().lower() == "partial")
    summary = {
        "n_judged": len(judged),
        "human_supported_rate": yes / len(judged),
        "human_partial_rate": partial / len(judged),
        "human_unsupported_rate": (len(judged) - yes - partial) / len(judged),
        "agreement_with_gold": sum(1 for r in judged if (r["verdict"].strip().lower() == "yes") == (str(r["gold"]) == "1")) / len(judged),
    }
    rp = Path(results_json)
    results = json.loads(rp.read_text(encoding="utf-8"))
    results.setdefault("arms", {}).setdefault(arm, {})["human_audit"] = summary
    rp.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({arm: summary}, indent=2))
    return summary
