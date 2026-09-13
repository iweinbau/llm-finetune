"""Score one or more prediction files against the held-out test set.

Outputs (in --out-dir, default results/):
  results.json          all metrics for every arm + run metadata
  errors_<arm>.csv      every label-level disagreement, with the model's evidence and its status
  samples.json          a fixed sample of test reports with every arm's parsed output
                        (feeds the GitHub Pages gallery)
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
from pathlib import Path

from . import __version__
from .data import label_prevalence, read_jsonl
from .metrics import ItemScore, evaluate_arm, evidence_status, score_item
from .parsing import parse_prediction
from .schema import LABELS


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_predictions(path: str | Path) -> tuple[str, dict[str, dict]]:
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit(f"{path} is empty")
    arm = rows[0].get("arm") or Path(path).stem.replace("preds_", "")
    return arm, {r["id"]: r for r in rows}


def score_arm(test: list[dict], preds: dict[str, dict]) -> tuple[list[ItemScore], dict[str, dict]]:
    """Score the test reports that have a prediction. Reports the arm never ran on are
    NOT counted as wrong answers — a missing row is a pipeline gap (crash, --limit), not a
    model refusal; refusals show up as invalid JSON in `raw`. Coverage is reported separately."""
    items, parsed = [], {}
    for rec in test:
        p = preds.get(rec["id"])
        if p is None:
            continue
        raw = p.get("raw", "")
        pred = parse_prediction(raw)
        parsed[rec["id"]] = {
            "abnormalities": pred.abnormalities,
            "evidence": pred.evidence,
            "parse_ok": pred.parse_ok,
            "schema_ok": pred.schema_ok,
            "issues": pred.issues,
            "raw": raw,
        }
        items.append(score_item(rec, pred))
    return items, parsed


def write_errors_csv(path: Path, test: list[dict], parsed: dict[str, dict]) -> int:
    n = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "label", "gold", "pred", "evidence", "evidence_status", "issues", "report"])
        for rec in test:
            p = parsed.get(rec["id"])
            if p is None:
                continue
            for k in LABELS:
                g = int(bool(rec["labels"].get(k, 0)))
                pr = int(p["abnormalities"].get(k, False))
                ev = p["evidence"].get(k)
                st = evidence_status(ev, rec["report"]) if pr else ""
                if g != pr or (pr and st != "supported"):
                    w.writerow([rec["id"], k, g, pr, ev or "", st, ";".join(p["issues"]), rec["report"]])
                    n += 1
    return n


def build_samples(test: list[dict], parsed_by_arm: dict[str, dict], n: int = 40, seed: int = 3) -> list[dict]:
    import random

    rng = random.Random(seed)
    covered = [r for r in test if all(r["id"] in parsed for parsed in parsed_by_arm.values())]
    idx = list(range(len(covered)))
    rng.shuffle(idx)
    out = []
    for i in idx[:n]:
        rec = covered[i]
        entry = {"id": rec["id"], "report": rec["report"], "gold": rec["labels"], "arms": {}}
        for arm, parsed in parsed_by_arm.items():
            p = parsed[rec["id"]]
            entry["arms"][arm] = {
                "abnormalities": p["abnormalities"],
                "evidence": {
                    k: {"quote": v, "status": evidence_status(v, rec["report"])}
                    for k, v in p["evidence"].items()
                    if p["abnormalities"].get(k)
                },
                "parse_ok": p["parse_ok"],
                "schema_ok": p["schema_ok"],
                "issues": p["issues"],
            }
        out.append(entry)
    return out


def run_evaluate(
    test_path: str | Path,
    pred_paths: list[str | Path],
    out_dir: str | Path = "results",
    n_boot: int = 1000,
    train_stats_path: str | Path | None = None,
) -> dict:
    test = read_jsonl(test_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms: dict[str, dict] = {}
    parsed_by_arm: dict[str, dict] = {}
    for pp in pred_paths:
        arm, preds = load_predictions(pp)
        items, parsed = score_arm(test, preds)
        metrics = evaluate_arm(items, n_boot=n_boot)
        meta = next(iter(preds.values()))
        coverage = {"n_scored": len(items), "n_test": len(test), "fraction": len(items) / max(len(test), 1)}
        if coverage["fraction"] < 1:
            print(
                f"[{arm:>14}] NOTE: predictions cover {len(items)}/{len(test)} test reports "
                f"({100*coverage['fraction']:.0f}%) — metrics below are over those {len(items)} only. "
                + ("This looks like a --limit smoke test." if len(items) <= 50 else "Re-run predict to fill the gap.")
            )
        metrics["coverage"] = coverage
        metrics["meta"] = {
            "model": meta.get("model"),
            "adapter": meta.get("adapter"),
            "shots": meta.get("shots", 0),
            "thinking": meta.get("thinking", False),
            "n_predictions": len(preds),
            "source": str(pp),
        }
        n_err = write_errors_csv(out_dir / f"errors_{arm}.csv", test, parsed)
        metrics["n_error_rows"] = n_err
        arms[arm] = metrics
        parsed_by_arm[arm] = parsed
        print(
            f"[{arm:>14}] micro-F1 {metrics['micro_f1']:.3f}  macro-F1 {metrics['macro_f1']:.3f}  "
            f"exact {metrics['exact_match_rate']:.3f}  json-valid {metrics['json_valid_rate']:.3f}  "
            f"unsupported-pos {metrics['unsupported_positive_rate']}"
        )

    results = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "llm_finetune_version": __version__,
        "git_sha": _git_sha(),
        "test": {
            "path": str(test_path),
            "n": len(test),
            "label_prevalence": label_prevalence(test),
            "n_normal": sum(1 for r in test if not any(r["labels"].get(k) for k in LABELS)),
        },
        "labels": LABELS,
        "arms": arms,
    }
    if train_stats_path and Path(train_stats_path).exists():
        results["train_stats"] = json.loads(Path(train_stats_path).read_text(encoding="utf-8"))

    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    samples = build_samples(test, parsed_by_arm)
    (out_dir / "samples.json").write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_dir/'results.json'} and {out_dir/'samples.json'}")
    print()
    print(reading_guide(results))
    return results


# ----------------------------------------------------------------------------- reading guide
# Rough expectation bands for THIS task (CT-RATE reports, 18 labels, verbatim-quote contract).
# They are priors to orient a reader, not thresholds to pass; the real answer is always the
# comparison between arms on the same reports, with the confidence intervals.
BANDS = {
    "json_valid_rate": ("Valid JSON", "≥ 0.98 expected from any tuned model; < 0.90 means a format problem", True),
    "micro_f1": ("Micro-F1 vs labels", "0.80–0.90 typical for an un-tuned 4B; the ceiling is well below 1.0 because the labels are themselves classifier output", True),
    "unsupported_positive_rate": ("Unsupported positives", "< 0.02 very good, 0.02–0.05 acceptable, > 0.10 the model invents evidence", False),
    "exact_match_rate": ("Exact match", "harsh (all 18 flags right); 0.4–0.7 is normal — use it as a tie-breaker, not a headline", True),
    "normal_report_false_alarm_rate": ("False alarms on normal reports", "< 0.05 good; this is what a surgeon would notice first", False),
}


def _pct(v):
    return "–" if v is None else f"{100 * v:.1f}%"


def reading_guide(results: dict) -> str:
    """Plain-language interpretation of results.json, printed after every evaluate run."""
    arms = results.get("arms", {})
    if not arms:
        return ""
    lines = ["HOW TO READ THESE NUMBERS", "=" * 26]
    n_test = results.get("test", {}).get("n")
    for arm, m in arms.items():
        cov = m.get("coverage", {})
        n = cov.get("n_scored", m.get("n"))
        ci = m.get("ci95", {}).get("micro_f1")
        ci_txt = f" (95% CI {_pct(ci['lo'])}–{_pct(ci['hi'])})" if ci else ""
        lines.append(f"\n{arm}  — scored on {n} of {n_test} test reports")
        if n and n < 100:
            lines.append(f"  ! with n={n} every number here has a margin of several points; use it to check the pipeline, not to rank models")
        lines.append(f"  Valid JSON            {_pct(m.get('json_valid_rate'))}   {BANDS['json_valid_rate'][1]}")
        lines.append(f"  Micro-F1 vs labels    {_pct(m.get('micro_f1'))}{ci_txt}   {BANDS['micro_f1'][1]}")
        lines.append(f"    precision {_pct(m.get('micro_precision'))} = of the findings it asserted, this share agrees with the labels")
        lines.append(f"    recall    {_pct(m.get('micro_recall'))} = of the labelled findings, this share it found")
        esc = m.get("evidence_status_counts", {})
        lines.append(
            f"  Unsupported positives {_pct(m.get('unsupported_positive_rate'))}   of {m.get('n_pred_positive')} asserted findings: "
            f"{esc.get('supported', 0)} quoted correctly, {esc.get('fabricated', 0)} quote not in report, "
            f"{esc.get('missing', 0)} no quote, {esc.get('too_short', 0)} too short.  {BANDS['unsupported_positive_rate'][1]}"
        )
        lines.append(f"  Exact match           {_pct(m.get('exact_match_rate'))}   {BANDS['exact_match_rate'][1]}")
        lines.append(f"  Normal-report alarms  {_pct(m.get('normal_report_false_alarm_rate'))}   {BANDS['normal_report_false_alarm_rate'][1]}")
    base = arms.get("base_zeroshot")
    others = [a for a in arms if a != "base_zeroshot"]
    if base and others:
        lines.append("\nVERSUS THE BASE MODEL (the only comparison that answers 'did fine-tuning help?')")
        for a in others:
            m = arms[a]
            d_f1 = (m.get("micro_f1") or 0) - (base.get("micro_f1") or 0)
            bu, mu = base.get("unsupported_positive_rate"), m.get("unsupported_positive_rate")
            d_un = None if bu is None or mu is None else mu - bu
            d_ex = (m.get("exact_match_rate") or 0) - (base.get("exact_match_rate") or 0)
            ci_w = None
            if m.get("ci95", {}).get("micro_f1"):
                c = m["ci95"]["micro_f1"]
                ci_w = c["hi"] - c["lo"]
            noise = f"; the F1 confidence interval is ±{100*ci_w/2:.1f} pts wide, so treat differences smaller than that as noise" if ci_w else ""
            lines.append(
                f"  {a}: micro-F1 {d_f1:+.3f}, exact match {d_ex:+.3f}, unsupported positives "
                + ("n/a" if d_un is None else f"{d_un:+.3f}")
                + noise
            )
        lines.append("  A fine-tune has 'worked' if it is clearly better on unsupported positives AND not worse on F1.")
        lines.append("  Better F1 with MORE unsupported positives is the bad outcome — it learned to agree with the labels by asserting things it cannot show.")
    else:
        lines.append("\nOnly one arm so far. Absolute numbers cannot tell you whether a model is good; run base_fewshot and lora on the same reports and compare.")
    lines.append("\nNext look: results/errors_<arm>.csv lists every disagreement with the model's quote — read 20 rows before believing any metric.")
    return "\n".join(lines)
