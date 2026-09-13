"""`llm-finetune` command-line entry point.

  llm-finetune prepare   download CT-RATE CSVs, build train/valid/test JSONL (+ heuristic evidence)
  llm-finetune silver    replace heuristic evidence with Claude-generated, verified quotes
  llm-finetune predict   run an inference arm (local MLX base/LoRA, or Claude)
  llm-finetune evaluate  score prediction files -> results/results.json
  llm-finetune site      refresh docs/results.js for GitHub Pages
  llm-finetune audit     make / score the human evidence audit sheet

Training itself is plain `mlx_lm.lora -c configs/lora_qwen3_4b.yaml` (see Makefile).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .schema import LABELS


def cmd_prepare(a: argparse.Namespace) -> None:
    from . import data
    from .silver import attach_evidence, heuristic_evidence

    raw = Path(a.raw_dir)
    if a.download:
        print("downloading CT-RATE CSVs (needs `huggingface-cli login` + accepted terms)…")
        paths = data.download(raw, token=a.hf_token)
    else:
        paths = {k: raw / v for k, v in data.FILES.items()}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            sys.exit("missing raw files (run with --download):\n  " + "\n  ".join(missing))

    print("loading + de-duplicating…")
    train_df = data.load_split(paths["train_reports"], paths["train_labels"])
    valid_df = data.load_split(paths["valid_reports"], paths["valid_labels"])
    print(f"  train scans: {len(train_df)}  ({train_df['patient'].nunique()} patients)")
    print(f"  valid scans: {len(valid_df)}  ({valid_df['patient'].nunique()} patients)")

    splits = data.make_splits(train_df, valid_df, a.n_train, a.n_valid, a.n_test, seed=a.seed)

    stats = {"seed": a.seed, "splits": {}}
    for name in ("train", "valid"):
        recs = splits[name]
        ev = {r["id"]: heuristic_evidence(r["report"], r["labels"]) for r in recs}
        s = attach_evidence(recs, ev)
        s["evidence_method"] = "heuristic"
        s["label_prevalence"] = data.label_prevalence(recs)
        stats["splits"][name] = s
        data.attach_messages([r for r in recs if r["train_ok"]], with_target=True)
        print(f"  {name}: {s['n_kept']}/{s['n_records']} kept with complete evidence")
    stats["splits"]["test"] = {
        "n_records": len(splits["test"]),
        "label_prevalence": data.label_prevalence(splits["test"]),
    }

    out = Path(a.out_dir)
    mlx_dir = out / "mlx"
    for name in ("train", "valid", "test"):
        data.write_jsonl(splits[name], out / f"{name}.jsonl")
    for name in ("train", "valid"):
        data.write_jsonl(
            [{"messages": r["messages"]} for r in splits[name] if r.get("train_ok")],
            mlx_dir / f"{name}.jsonl",
        )
    (out / "prepare_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {out}/{{train,valid,test}}.jsonl and {mlx_dir}/{{train,valid}}.jsonl")


def cmd_silver(a: argparse.Namespace) -> None:
    from . import data
    from .silver import attach_evidence, claude_evidence_batch

    out = Path(a.out_dir)
    stats_path = out / "prepare_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {"splits": {}}
    for name in ("train", "valid"):
        recs = data.read_jsonl(out / f"{name}.jsonl")
        print(f"[{name}] requesting verified quotes from {a.model} for {len(recs)} reports…")
        ev = claude_evidence_batch(recs, model=a.model, max_workers=a.workers, cache_path=out / f"silver_cache_{name}.jsonl")
        s = attach_evidence(recs, ev)
        s["evidence_method"] = f"claude:{a.model}"
        s["label_prevalence"] = data.label_prevalence(recs)
        stats["splits"][name] = s
        data.attach_messages([r for r in recs if r["train_ok"]], with_target=True)
        data.write_jsonl(recs, out / f"{name}.jsonl")
        data.write_jsonl([{"messages": r["messages"]} for r in recs if r.get("train_ok")], out / "mlx" / f"{name}.jsonl")
        print(f"  {name}: {s['n_kept']}/{s['n_records']} kept with complete evidence")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def cmd_predict(a: argparse.Namespace) -> None:
    out = Path(a.out) if a.out else Path("results") / f"preds_{a.arm}.jsonl"
    if a.backend == "claude":
        from .frontier import run_claude

        run_claude(a.test, out, arm=a.arm, model=a.model, limit=a.limit, max_workers=a.workers)
    else:
        from .infer import run_mlx

        run_mlx(
            model_id=a.model,
            test_path=a.test,
            out_path=out,
            arm=a.arm,
            adapter_path=a.adapter,
            shots=a.shots,
            train_path=a.train,
            max_tokens=a.max_tokens,
            temperature=a.temperature,
            batch_size=a.batch_size,
            limit=a.limit,
            thinking=a.thinking,
            resume=not a.no_resume,
            sequential=a.sequential,
            retry_invalid=a.retry_invalid,
        )
    print(f"predictions -> {out}")


def cmd_evaluate(a: argparse.Namespace) -> None:
    from .evaluate import run_evaluate

    run_evaluate(a.test, a.preds, out_dir=a.out_dir, n_boot=a.n_boot, train_stats_path=a.train_stats)


def cmd_site(a: argparse.Namespace) -> None:
    from .site import build_site

    build_site(a.results_dir, a.docs_dir, readme=None if a.no_readme else a.readme)


def cmd_audit(a: argparse.Namespace) -> None:
    from .audit import make_audit_sheet, score_audit

    if a.audit_cmd == "make":
        make_audit_sheet(a.test, a.preds, a.out, n=a.n)
    else:
        score_audit(a.csv, a.results)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-finetune", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare", help="download + build splits")
    s.add_argument("--raw-dir", default="data/raw")
    s.add_argument("--out-dir", default="data/processed")
    s.add_argument("--download", action="store_true", help="fetch CSVs from Hugging Face")
    s.add_argument("--hf-token", default=None)
    s.add_argument("--n-train", type=int, default=3000)
    s.add_argument("--n-valid", type=int, default=200)
    s.add_argument("--n-test", type=int, default=500)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("silver", help="Claude-generated verified evidence for train/valid")
    s.add_argument("--out-dir", default="data/processed")
    s.add_argument("--model", default="claude-haiku-4-5-20251001")
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(func=cmd_silver)

    s = sub.add_parser("predict", help="run one inference arm")
    s.add_argument("--arm", required=True, help="name used in results, e.g. base_zeroshot / base_fewshot / lora / claude")
    s.add_argument("--backend", choices=["mlx", "claude"], default="mlx")
    s.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    s.add_argument("--adapter", default=None, help="path to LoRA adapters dir")
    s.add_argument("--shots", type=int, default=0)
    s.add_argument("--test", default="data/processed/test.jsonl")
    s.add_argument("--train", default="data/processed/train.jsonl", help="source of few-shot examples")
    s.add_argument("--out", default=None)
    s.add_argument("--max-tokens", type=int, default=1024, help="output budget; long reports with many quotes need ~700")
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--limit", type=int, default=None, help="only the first N test reports (smoke test)")
    s.add_argument("--thinking", action="store_true", help="let Qwen3 think before answering (slower)")
    s.add_argument("--no-resume", action="store_true")
    s.add_argument("--sequential", action="store_true", help="one prompt at a time; use for long (few-shot) prompts")
    s.add_argument("--retry-invalid", action="store_true", help="regenerate only rows whose output failed to parse")
    s.add_argument("--workers", type=int, default=4, help="claude backend concurrency")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("evaluate", help="score prediction files")
    s.add_argument("--test", default="data/processed/test.jsonl")
    s.add_argument("--preds", nargs="+", required=True)
    s.add_argument("--out-dir", default="results")
    s.add_argument("--n-boot", type=int, default=1000)
    s.add_argument("--train-stats", default="data/processed/prepare_stats.json")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("site", help="refresh docs/results.js")
    s.add_argument("--results-dir", default="results")
    s.add_argument("--docs-dir", default="docs")
    s.add_argument("--readme", default="README.md", help="README whose RESULTS block gets refreshed")
    s.add_argument("--no-readme", action="store_true")
    s.set_defaults(func=cmd_site)

    s = sub.add_parser("audit", help="human evidence audit")
    ss = s.add_subparsers(dest="audit_cmd", required=True)
    m = ss.add_parser("make")
    m.add_argument("--test", default="data/processed/test.jsonl")
    m.add_argument("--preds", required=True)
    m.add_argument("--out", default="results/audit.csv")
    m.add_argument("--n", type=int, default=60)
    sc = ss.add_parser("score")
    sc.add_argument("--csv", default="results/audit.csv")
    sc.add_argument("--results", default="results/results.json")
    s.set_defaults(func=cmd_audit)

    lab = sub.add_parser("labels", help="print the 18 labels")
    lab.set_defaults(func=lambda a: print("\n".join(LABELS)))
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
