"""Evaluation metrics.

Two families, deliberately kept separate:

1. Label agreement vs CT-RATE (per-label P/R/F1, micro/macro F1, exact match).
   Ground truth here is CT-RATE's released labels, which were themselves produced
   by a text classifier (RadBERT) and validated on a subset — so "agreement", not
   "accuracy".

2. Faithfulness of the model's own claims, which needs NO labels at all:
   every abnormality the model marks true must come with a quote that appears
   verbatim in the report. The headline number is

       unsupported_positive_rate = (# predicted-true flags without a verified quote)
                                   / (# predicted-true flags)

   i.e. how often the model asserts a finding it cannot point to. A quote is
   "verified" if it has >= MIN_EVIDENCE_WORDS words and occurs verbatim (after
   whitespace/case normalisation) in the report. This catches fabricated text,
   not misread text — a real sentence quoted for the wrong label passes. The
   human audit (llm-finetune audit) closes that gap on a sample.

Invalid / unparseable outputs are scored as "all false, no evidence" so a model
cannot improve its numbers by failing to answer. Validity rates are reported too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import LABELS, MIN_EVIDENCE_WORDS, Prediction, quote_in_report, word_count

EvidenceStatus = str  # "supported" | "missing" | "too_short" | "fabricated"


def evidence_status(quote: str | None, report: str) -> EvidenceStatus:
    if quote is None or not str(quote).strip():
        return "missing"
    if word_count(quote) < MIN_EVIDENCE_WORDS:
        return "too_short"
    return "supported" if quote_in_report(quote, report) else "fabricated"


@dataclass
class ItemScore:
    id: str
    gold: np.ndarray  # (18,) int
    pred: np.ndarray  # (18,) int
    parse_ok: bool
    schema_ok: bool
    statuses: dict[str, EvidenceStatus]  # for predicted-true labels
    stray_evidence: int  # evidence given for labels predicted false
    n_words: int


def score_item(rec: dict, pred: Prediction) -> ItemScore:
    gold = np.array([int(bool(rec["labels"].get(k, 0))) for k in LABELS])
    predv = np.array([int(pred.abnormalities.get(k, False)) for k in LABELS])
    statuses = {}
    for k in LABELS:
        if pred.abnormalities.get(k, False):
            statuses[k] = evidence_status(pred.evidence.get(k), rec["report"])
    stray = sum(1 for k, v in pred.evidence.items() if v and not pred.abnormalities.get(k, False))
    return ItemScore(
        id=rec["id"],
        gold=gold,
        pred=predv,
        parse_ok=pred.parse_ok,
        schema_ok=pred.schema_ok,
        statuses=statuses,
        stray_evidence=stray,
        n_words=len(rec["report"].split()),
    )


def _f1(tp: float, fp: float, fn: float) -> float:
    d = 2 * tp + fp + fn
    return float(2 * tp / d) if d else 0.0


def _safe_div(a: float, b: float) -> float | None:
    return float(a / b) if b else None


def _core(items: list[ItemScore]) -> dict:
    """Metrics for a fixed list of items (no CIs)."""
    if not items:
        return {}
    G = np.stack([it.gold for it in items])  # (n, 18)
    P = np.stack([it.pred for it in items])
    tp = ((G == 1) & (P == 1)).sum(0)
    fp = ((G == 0) & (P == 1)).sum(0)
    fn = ((G == 1) & (P == 0)).sum(0)
    tn = ((G == 0) & (P == 0)).sum(0)

    per_label = {}
    for i, k in enumerate(LABELS):
        per_label[k] = {
            "tp": int(tp[i]),
            "fp": int(fp[i]),
            "fn": int(fn[i]),
            "tn": int(tn[i]),
            "gold_pos": int(G[:, i].sum()),
            "pred_pos": int(P[:, i].sum()),
            "precision": _safe_div(tp[i], tp[i] + fp[i]),
            "recall": _safe_div(tp[i], tp[i] + fn[i]),
            "f1": _f1(tp[i], fp[i], fn[i]),
        }
    supported_labels = [k for i, k in enumerate(LABELS) if G[:, i].sum() > 0]
    macro = float(np.mean([per_label[k]["f1"] for k in supported_labels])) if supported_labels else 0.0

    n = len(items)
    n_pred_pos = int(P.sum())
    status_counts = {"supported": 0, "missing": 0, "too_short": 0, "fabricated": 0}
    for it in items:
        for s in it.statuses.values():
            status_counts[s] += 1
    unsupported = n_pred_pos - status_counts["supported"]

    normal_mask = G.sum(1) == 0
    n_normal = int(normal_mask.sum())
    normal_false_alarm = _safe_div((P[normal_mask].sum(1) > 0).sum(), n_normal)

    return {
        "n": n,
        "json_valid_rate": float(np.mean([it.parse_ok for it in items])),
        "schema_valid_rate": float(np.mean([it.schema_ok for it in items])),
        "micro_precision": _safe_div(tp.sum(), tp.sum() + fp.sum()),
        "micro_recall": _safe_div(tp.sum(), tp.sum() + fn.sum()),
        "micro_f1": _f1(tp.sum(), fp.sum(), fn.sum()),
        "macro_f1": macro,
        "macro_f1_labels": supported_labels,
        "exact_match_rate": float(np.mean((G == P).all(1))),
        "hamming_accuracy": float((G == P).mean()),
        "n_pred_positive": n_pred_pos,
        "n_gold_positive": int(G.sum()),
        "evidence_status_counts": status_counts,
        "unsupported_positive_rate": _safe_div(unsupported, n_pred_pos),
        "fabrication_rate": _safe_div(
            status_counts["fabricated"],
            status_counts["supported"] + status_counts["fabricated"] + status_counts["too_short"],
        ),
        "evidence_coverage": _safe_div(status_counts["supported"], n_pred_pos),
        "stray_evidence_total": int(sum(it.stray_evidence for it in items)),
        "n_normal_reports": n_normal,
        "normal_report_false_alarm_rate": normal_false_alarm,
        "per_label": per_label,
    }


def bootstrap_ci(
    items: list[ItemScore],
    keys: tuple[str, ...] = ("micro_f1", "macro_f1", "unsupported_positive_rate", "exact_match_rate"),
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict]:
    """Percentile bootstrap over reports (the unit of independence)."""
    rng = np.random.default_rng(seed)
    n = len(items)
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = _core([items[i] for i in idx])
        for k in keys:
            v = m.get(k)
            if v is not None:
                samples[k].append(v)
    out = {}
    for k in keys:
        arr = np.array(samples[k]) if samples[k] else np.array([np.nan])
        out[k] = {
            "lo": float(np.nanpercentile(arr, 100 * alpha / 2)),
            "hi": float(np.nanpercentile(arr, 100 * (1 - alpha / 2))),
        }
    return out


def by_length(items: list[ItemScore], n_bins: int = 3) -> list[dict]:
    """Micro-F1 and unsupported rate by report-length tertile."""
    if len(items) < n_bins * 5:
        return []
    lengths = np.array([it.n_words for it in items])
    edges = np.quantile(lengths, np.linspace(0, 1, n_bins + 1))
    out = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = [it for it in items if (lo <= it.n_words <= hi if b == n_bins - 1 else lo <= it.n_words < hi)]
        if not sel:
            continue
        m = _core(sel)
        out.append(
            {
                "bin": f"{int(lo)}–{int(hi)} words",
                "n": m["n"],
                "micro_f1": m["micro_f1"],
                "unsupported_positive_rate": m["unsupported_positive_rate"],
                "exact_match_rate": m["exact_match_rate"],
            }
        )
    return out


def evaluate_arm(items: list[ItemScore], n_boot: int = 1000, seed: int = 0) -> dict:
    m = _core(items)
    m["ci95"] = bootstrap_ci(items, n_boot=n_boot, seed=seed) if n_boot > 0 else {}
    m["by_length"] = by_length(items)
    valid = [it for it in items if it.schema_ok]
    if valid and len(valid) < len(items):
        v = _core(valid)
        m["valid_only"] = {
            "n": v["n"],
            "micro_f1": v["micro_f1"],
            "unsupported_positive_rate": v["unsupported_positive_rate"],
        }
    return m
