"""Turn results/ into the static GitHub Pages site under docs/ and refresh the README.

docs/index.html is hand-written and static; this step refreshes docs/results.js (the data
the page renders) and rewrites the block between <!-- RESULTS:START --> and
<!-- RESULTS:END --> in README.md with a one-sentence verdict and a results table. Using a
.js file instead of fetch()ing JSON means the page also works when opened from disk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ARM_NAMES = {
    "base_zeroshot": "Before fine-tuning",
    "base_fewshot": "Before, with 3 examples in the prompt",
    "lora": "After fine-tuning (LoRA)",
    "claude": "Frontier model (Claude)",
}
ARM_ORDER = ["base_zeroshot", "base_fewshot", "lora", "claude"]

COLUMNS = [
    ("unsupported_positive_rate", "Claims without evidence ↓"),
    ("micro_f1", "Agreement with labels (F1) ↑"),
    ("exact_match_rate", "All 18 findings right ↑"),
    ("json_valid_rate", "Well-formed output ↑"),
    ("normal_report_false_alarm_rate", "False alarms on normal reports ↓"),
]


def _pct(v, ci=None, digits=1):
    if v is None:
        return "–"
    s = f"{100 * v:.{digits}f}%"
    if ci:
        s += f" ({100 * ci['lo']:.0f}–{100 * ci['hi']:.0f})"
    return s


def _sorted_arms(arms: dict) -> list[str]:
    return sorted(arms, key=lambda a: (ARM_ORDER.index(a) if a in ARM_ORDER else 99, a))


def verdict(results: dict) -> str:
    """One sentence a recruiter can read: what changed, on how many reports."""
    arms = results.get("arms", {})
    n = results.get("test", {}).get("n")
    base, tuned = arms.get("base_zeroshot"), arms.get("lora")

    def word(delta, up, down, same, tol=0.015):
        return same if abs(delta) < tol else (up if delta > 0 else down)

    if base and tuned and tuned.get("unsupported_positive_rate") is not None and base.get("unsupported_positive_rate") is not None:
        du = tuned["unsupported_positive_rate"] - base["unsupported_positive_rate"]
        df = (tuned.get("micro_f1") or 0) - (base.get("micro_f1") or 0)
        return (
            f"Fine-tuning **{word(du, 'raised', 'cut', 'left unchanged')}** the share of findings the model claimed "
            f"without evidence from **{_pct(base['unsupported_positive_rate'])}** to **{_pct(tuned['unsupported_positive_rate'])}**, "
            f"and **{word(df, 'raised', 'lowered', 'left unchanged')}** agreement with the reference labels from "
            f"{_pct(base.get('micro_f1'))} to {_pct(tuned.get('micro_f1'))}, on {n} reports it had never seen."
        )
    if arms:
        a = "base_zeroshot" if base else _sorted_arms(arms)[0]
        m = arms[a]
        return (
            f"Baseline so far ({ARM_NAMES.get(a, a)}): claims a finding without evidence "
            f"**{_pct(m.get('unsupported_positive_rate'))}** of the time and agrees with the reference labels at "
            f"**{_pct(m.get('micro_f1'))}** F1 on {n} reports. The fine-tuned version has not been evaluated yet."
        )
    return ""


def results_markdown(results: dict) -> str:
    arms = results.get("arms", {})
    head = "| Version | " + " | ".join(h for _, h in COLUMNS) + " |"
    sep = "|" + "---|" * (1 + len(COLUMNS))
    lines = [head, sep]
    for a in _sorted_arms(arms):
        m = arms[a]
        cells = [_pct(m.get(k), (m.get("ci95") or {}).get(k)) for k, _ in COLUMNS]
        lines.append(f"| {ARM_NAMES.get(a, a)} | " + " | ".join(cells) + " |")
    n = results.get("test", {}).get("n")
    covs = {(m.get("coverage") or {}).get("n_scored", m.get("n")) for m in arms.values()}
    note = f"_{n} held-out CT-RATE reports; parentheses are 95% bootstrap confidence intervals._"
    if covs and covs != {n}:
        note = f"_Partial run: scored on {'/'.join(str(c) for c in sorted(covs))} of {n} test reports (smoke test)._ " + note
    return "\n".join(lines) + "\n\n" + note


def update_readme(readme: Path, results: dict) -> bool:
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    block = verdict(results) + "\n\n" + results_markdown(results)
    new, n = re.subn(
        r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)",
        lambda m: f"{m.group(1)}\n{block}\n{m.group(2)}",
        text,
        flags=re.DOTALL,
    )
    if n:
        readme.write_text(new, encoding="utf-8")
    return bool(n)


def build_site(results_dir: str | Path = "results", docs_dir: str | Path = "docs", readme: str | Path | None = "README.md") -> Path:
    results_dir, docs_dir = Path(results_dir), Path(docs_dir)
    rj = results_dir / "results.json"
    if not rj.exists():
        raise SystemExit(f"{rj} not found — run `llm-finetune evaluate` first")
    results = json.loads(rj.read_text(encoding="utf-8"))
    samples_path = results_dir / "samples.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8")) if samples_path.exists() else []
    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / "results.js"
    out.write_text(
        "// Generated by `llm-finetune site` — do not edit by hand.\n"
        f"window.LLM_FINETUNE_RESULTS = {json.dumps(results, ensure_ascii=False)};\n"
        f"window.LLM_FINETUNE_SAMPLES = {json.dumps(samples, ensure_ascii=False)};\n",
        encoding="utf-8",
    )
    md = verdict(results) + "\n\n" + results_markdown(results)
    (results_dir / "results_table.md").write_text(md + "\n", encoding="utf-8")
    updated = update_readme(Path(readme), results) if readme else False
    print(f"wrote {out}" + ("; README results block updated" if updated else "") + f"\n\n{md}\n")
    return out
