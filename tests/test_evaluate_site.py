import json

from llm_finetune import data
from llm_finetune.audit import make_audit_sheet
from llm_finetune.cli import main
from llm_finetune.schema import LABELS
from llm_finetune.silver import heuristic_evidence


def _fake_predictions(test_records, arm, quality):
    """Simulate a model: `quality` in [0,1] = probability each label is copied from gold."""
    import random

    rng = random.Random(hash(arm) % 1000)
    rows = []
    for r in test_records:
        abn = {}
        for k in LABELS:
            g = bool(r["labels"][k])
            abn[k] = g if rng.random() < quality else (not g if rng.random() < 0.3 else g)
        ev = heuristic_evidence(r["report"], {k: int(v) for k, v in abn.items()})
        if rng.random() < 0.2:  # sometimes fabricate
            for k, flag in abn.items():
                if flag and k not in ev:
                    ev[k] = "Marked " + k.lower() + " is present throughout."
                    break
        raw = json.dumps({"abnormalities": abn, "evidence": ev})
        if rng.random() < 0.05:
            raw = "```json\n" + raw + "\n```"
        rows.append({"id": r["id"], "arm": arm, "model": "fake", "adapter": None, "shots": 0, "raw": raw})
    return rows


def test_evaluate_and_site_end_to_end(raw_dir, tmp_path):
    out = tmp_path / "processed"
    main(["prepare", "--raw-dir", str(raw_dir), "--out-dir", str(out), "--n-train", "20", "--n-valid", "5", "--n-test", "20"])
    test = data.read_jsonl(out / "test.jsonl")

    preds_dir = tmp_path / "results"
    preds_dir.mkdir()
    p_a = preds_dir / "preds_base_zeroshot.jsonl"
    p_b = preds_dir / "preds_lora.jsonl"
    data.write_jsonl(_fake_predictions(test, "base_zeroshot", 0.6), p_a)
    data.write_jsonl(_fake_predictions(test, "lora", 0.95), p_b)

    main(["evaluate", "--test", str(out / "test.jsonl"), "--preds", str(p_a), str(p_b), "--out-dir", str(preds_dir), "--n-boot", "50", "--train-stats", str(out / "prepare_stats.json")])
    results = json.loads((preds_dir / "results.json").read_text())
    assert set(results["arms"]) == {"base_zeroshot", "lora"}
    lora, base = results["arms"]["lora"], results["arms"]["base_zeroshot"]
    assert lora["micro_f1"] >= base["micro_f1"]
    assert "ci95" in lora and "per_label" in lora and len(lora["per_label"]) == 18
    assert (preds_dir / "errors_lora.csv").exists()
    assert results["train_stats"]["splits"]["train"]["n_records"] == 20
    samples = json.loads((preds_dir / "samples.json").read_text())
    assert samples and set(samples[0]["arms"]) == {"base_zeroshot", "lora"}

    docs = tmp_path / "docs"
    readme = tmp_path / "README.md"
    readme.write_text("# x\n\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\n\nrest\n")
    main(["site", "--results-dir", str(preds_dir), "--docs-dir", str(docs), "--readme", str(readme)])
    js = (docs / "results.js").read_text()
    assert js.startswith("// Generated") and "window.LLM_FINETUNE_RESULTS" in js and "window.LLM_FINETUNE_SAMPLES" in js
    assert (preds_dir / "results_table.md").exists()
    r = readme.read_text()
    assert "old" not in r and "Fine-tuning **" in r and "| After fine-tuning (LoRA) |" in r and r.endswith("rest\n")

    n = make_audit_sheet(out / "test.jsonl", p_b, preds_dir / "audit.csv", n=10)
    assert 0 < n <= 10


def test_partial_predictions_are_scored_on_covered_reports_only(raw_dir, tmp_path):
    """A --limit smoke test must not count un-predicted reports as wrong answers."""
    out = tmp_path / "processed"
    main(["prepare", "--raw-dir", str(raw_dir), "--out-dir", str(out), "--n-train", "20", "--n-valid", "5", "--n-test", "20"])
    test = data.read_jsonl(out / "test.jsonl")
    preds_dir = tmp_path / "results"
    preds_dir.mkdir()
    p = preds_dir / "preds_base_zeroshot.jsonl"
    data.write_jsonl(_fake_predictions(test[:5], "base_zeroshot", 1.0), p)  # perfect on 5 of 20
    main(["evaluate", "--test", str(out / "test.jsonl"), "--preds", str(p), "--out-dir", str(preds_dir), "--n-boot", "0"])
    m = json.loads((preds_dir / "results.json").read_text())["arms"]["base_zeroshot"]
    assert m["coverage"] == {"n_scored": 5, "n_test": 20, "fraction": 0.25}
    assert m["n"] == 5
    assert m["json_valid_rate"] >= 0.8  # not diluted by the 15 un-predicted reports
    assert m["micro_recall"] is None or m["micro_recall"] > 0.5
    samples = json.loads((preds_dir / "samples.json").read_text())
    assert len(samples) == 5 and all("no_prediction" not in s["arms"]["base_zeroshot"]["issues"] for s in samples)


def test_reading_guide_mentions_comparison(raw_dir, tmp_path):
    from llm_finetune.evaluate import reading_guide

    results = {
        "test": {"n": 10},
        "arms": {
            "base_zeroshot": {"micro_f1": 0.8, "unsupported_positive_rate": 0.1, "exact_match_rate": 0.4, "coverage": {"n_scored": 10}, "n_pred_positive": 10, "evidence_status_counts": {}},
            "lora": {"micro_f1": 0.9, "unsupported_positive_rate": 0.01, "exact_match_rate": 0.6, "coverage": {"n_scored": 10}, "n_pred_positive": 10, "evidence_status_counts": {}, "ci95": {"micro_f1": {"lo": 0.85, "hi": 0.95}}},
        },
    }
    g = reading_guide(results)
    assert "VERSUS THE BASE MODEL" in g and "+0.100" in g and "-0.090" in g
