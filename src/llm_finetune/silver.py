"""Silver evidence for training targets.

CT-RATE gives us *which* abnormalities a report contains (18 binary labels) but
not *where* in the text they are stated. To teach the model to quote its
evidence we need a quote per positive label in the training set. Two methods:

  heuristic  — pick the report sentence matching a label-specific keyword pattern,
               preferring sentences without negation cues. Free, offline, noisy.
  claude     — ask a Claude model for the verbatim quote for each positive label.
               Costs a few dollars for ~3k reports on Haiku. Cleaner.

Either way, every quote is verified mechanically (must appear verbatim in the
report, >= MIN_EVIDENCE_WORDS words). Records where any positive label ends up
without a verified quote are DROPPED from training — we never teach the model
to assert a finding it cannot point to. Evidence is never generated for the test
split; the test set is scored against CT-RATE labels and the report text only.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .schema import (
    LABEL_PATTERNS,
    LABELS,
    MIN_EVIDENCE_WORDS,
    NEGATION_RE,
    quote_in_report,
    split_sentences,
    word_count,
)

_COMPILED = {k: re.compile(p, re.IGNORECASE) for k, p in LABEL_PATTERNS.items()}


def verify_quote(quote: str | None, report: str) -> bool:
    return (
        quote is not None
        and word_count(quote) >= MIN_EVIDENCE_WORDS
        and quote_in_report(quote, report)
    )


# --------------------------------------------------------------------------- heuristic
def heuristic_evidence(report: str, labels: dict[str, int | bool]) -> dict[str, str]:
    """Return {label: sentence} for positive labels where a plausible sentence exists."""
    sents = split_sentences(report)
    # Drop the section headers we add ourselves.
    sents = [s for s in sents if not re.fullmatch(r"(FINDINGS|IMPRESSION):?", s.strip(), re.IGNORECASE)]
    out: dict[str, str] = {}
    for label in LABELS:
        if not labels.get(label):
            continue
        pat = _COMPILED[label]
        cands = [s for s in sents if pat.search(s)]
        if not cands:
            continue
        positive = [s for s in cands if not NEGATION_RE.search(s)]
        pick = (positive or cands)[0]
        # Prefer a Findings-section sentence over the Impression when both match
        pick = re.sub(r"^(FINDINGS|IMPRESSION):\s*", "", pick, flags=re.IGNORECASE).strip()
        if verify_quote(pick, report):
            out[label] = pick
    return out


# --------------------------------------------------------------------------- claude
_SILVER_SYSTEM = (
    "You extract verbatim evidence from chest CT radiology reports. For each abnormality "
    "listed, return ONE quote copied exactly (character for character) from the report that "
    "states the abnormality is present. If the report does not clearly state it is present, "
    "return null for that abnormality. Respond with a JSON object mapping each listed "
    "abnormality to a string or null, and nothing else."
)


def _silver_user(report: str, positives: list[str]) -> str:
    return (
        "REPORT:\n"
        + report.strip()
        + "\n\nABNORMALITIES (labelled present by an upstream classifier):\n"
        + "\n".join(f"- {p}" for p in positives)
        + "\n\nReturn the JSON object."
    )


def claude_evidence_batch(
    records: list[dict],
    model: str = "claude-haiku-4-5-20251001",
    max_workers: int = 4,
    cache_path: str | Path | None = None,
) -> dict[str, dict[str, str]]:
    """Return {record_id: {label: verified_quote}} using the Anthropic API.

    Requires ANTHROPIC_API_KEY. Responses are cached to `cache_path` (jsonl) so a
    crashed run resumes for free.
    """
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pip install 'llm-finetune[frontier]' to use --method claude") from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic()
    from .parsing import extract_json

    cache: dict[str, dict] = {}
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                cache[d["id"]] = d["raw"]

    todo = [r for r in records if r["id"] not in cache and any(r["labels"].get(k) for k in LABELS)]

    def call(rec: dict) -> tuple[str, dict]:
        positives = [k for k in LABELS if rec["labels"].get(k)]
        for attempt in range(5):
            try:
                msg = client.messages.create(
                    model=model,
                    max_tokens=800,
                    system=_SILVER_SYSTEM,
                    messages=[{"role": "user", "content": _silver_user(rec["report"], positives)}],
                )
                text = "".join(getattr(b, "text", "") for b in msg.content)
                obj, _ = extract_json(text)
                return rec["id"], obj or {}
            except anthropic.RateLimitError:  # pragma: no cover
                time.sleep(2**attempt)
            except anthropic.APIStatusError as e:  # pragma: no cover
                if e.status_code >= 500:
                    time.sleep(2**attempt)
                else:
                    raise
        return rec["id"], {}

    if todo:
        with ThreadPoolExecutor(max_workers=max_workers) as ex, (
            cache_path.open("a", encoding="utf-8") if cache_path else _NullFile()
        ) as fh:
            futs = [ex.submit(call, r) for r in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                rid, obj = fut.result()
                cache[rid] = obj
                fh.write(json.dumps({"id": rid, "raw": obj}, ensure_ascii=False) + "\n")
                if i % 50 == 0:
                    print(f"  silver: {i}/{len(todo)}")

    out: dict[str, dict[str, str]] = {}
    for rec in records:
        raw = cache.get(rec["id"], {}) or {}
        ev: dict[str, str] = {}
        for label in LABELS:
            if not rec["labels"].get(label):
                continue
            q = raw.get(label)
            if isinstance(q, str) and verify_quote(q, rec["report"]):
                ev[label] = q.strip()
        out[rec["id"]] = ev
    return out


class _NullFile:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def write(self, *_):
        pass


# --------------------------------------------------------------------------- shared
def attach_evidence(records: list[dict], evidence_by_id: dict[str, dict[str, str]]) -> dict:
    """Attach evidence to records in place and return filtering statistics.

    A record is kept for training only if every positive label has verified evidence.
    Records with no positive labels (normal reports) are always kept.
    """
    kept, dropped = [], []
    per_label_missing = {k: 0 for k in LABELS}
    for rec in records:
        ev = evidence_by_id.get(rec["id"], {})
        rec["evidence"] = ev
        positives = [k for k in LABELS if rec["labels"].get(k)]
        missing = [k for k in positives if k not in ev]
        for k in missing:
            per_label_missing[k] += 1
        rec["train_ok"] = not missing
        (kept if not missing else dropped).append(rec)
    return {
        "n_records": len(records),
        "n_kept": len(kept),
        "n_dropped_missing_evidence": len(dropped),
        "missing_evidence_by_label": per_label_missing,
    }
