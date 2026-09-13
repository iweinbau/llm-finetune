import numpy as np

from llm_finetune.metrics import evaluate_arm, evidence_status, score_item
from llm_finetune.schema import LABELS, Prediction

REPORT = (
    "FINDINGS: Trachea is patent. Paraseptal emphysema is observed in the upper lobes. "
    "A 6 mm solid nodule is seen in the right lower lobe. No pleural effusion.\n"
    "IMPRESSION: Emphysema, nodule."
)


def _rec(positives, rid="r1"):
    return {"id": rid, "report": REPORT, "labels": {k: int(k in positives) for k in LABELS}}


def _pred(positives, evidence=None, **kw):
    return Prediction(
        abnormalities={k: (k in positives) for k in LABELS},
        evidence=evidence or {},
        **kw,
    )


def test_evidence_status_categories():
    assert evidence_status("Paraseptal emphysema is observed in the upper lobes.", REPORT) == "supported"
    assert evidence_status("  paraseptal   EMPHYSEMA is observed in the upper lobes. ", REPORT) == "supported"
    assert evidence_status("Severe centrilobular emphysema everywhere.", REPORT) == "fabricated"
    assert evidence_status("emphysema", REPORT) == "too_short"
    assert evidence_status(None, REPORT) == "missing"
    assert evidence_status("", REPORT) == "missing"


def test_perfect_prediction():
    rec = _rec(["Emphysema", "Lung nodule"])
    pred = _pred(
        ["Emphysema", "Lung nodule"],
        {
            "Emphysema": "Paraseptal emphysema is observed in the upper lobes.",
            "Lung nodule": "A 6 mm solid nodule is seen in the right lower lobe.",
        },
    )
    m = evaluate_arm([score_item(rec, pred)], n_boot=0)
    assert m["micro_f1"] == 1.0
    assert m["exact_match_rate"] == 1.0
    assert m["unsupported_positive_rate"] == 0.0
    assert m["evidence_coverage"] == 1.0
    assert m["json_valid_rate"] == 1.0


def test_unsupported_positive_counted():
    rec = _rec(["Emphysema"])
    # Correct label but fabricated quote, plus a false positive with no quote at all.
    pred = _pred(["Emphysema", "Pleural effusion"], {"Emphysema": "Diffuse emphysema throughout both lungs."})
    m = evaluate_arm([score_item(rec, pred)], n_boot=0)
    assert m["n_pred_positive"] == 2
    assert m["evidence_status_counts"] == {"supported": 0, "missing": 1, "too_short": 0, "fabricated": 1}
    assert m["unsupported_positive_rate"] == 1.0
    assert m["fabrication_rate"] == 1.0
    assert m["micro_precision"] == 0.5 and m["micro_recall"] == 1.0


def test_invalid_output_scored_as_all_false():
    rec = _rec(["Emphysema"])
    pred = Prediction.empty(raw="garbage")
    m = evaluate_arm([score_item(rec, pred)], n_boot=0)
    assert m["json_valid_rate"] == 0.0
    assert m["micro_recall"] == 0.0
    assert m["n_pred_positive"] == 0
    assert m["unsupported_positive_rate"] is None  # nothing asserted, nothing to support


def test_normal_report_false_alarm():
    recs = [_rec([], "n1"), _rec([], "n2"), _rec(["Emphysema"], "a1")]
    preds = [_pred(["Lung nodule"]), _pred([]), _pred(["Emphysema"], {"Emphysema": "Paraseptal emphysema is observed in the upper lobes."})]
    m = evaluate_arm([score_item(r, p) for r, p in zip(recs, preds)], n_boot=0)
    assert m["n_normal_reports"] == 2
    assert m["normal_report_false_alarm_rate"] == 0.5


def test_macro_f1_only_over_labels_present_in_gold():
    recs = [_rec(["Emphysema"], "a"), _rec(["Emphysema"], "b")]
    preds = [_pred(["Emphysema"]), _pred(["Emphysema", "Cardiomegaly"])]
    m = evaluate_arm([score_item(r, p) for r, p in zip(recs, preds)], n_boot=0)
    assert m["macro_f1_labels"] == ["Emphysema"]
    assert m["macro_f1"] == 1.0
    assert m["per_label"]["Cardiomegaly"]["fp"] == 1


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    recs, preds = [], []
    for i in range(40):
        pos = ["Emphysema"] if rng.random() < 0.5 else []
        recs.append(_rec(pos, f"r{i}"))
        # 80% correct predictions
        ppos = pos if rng.random() < 0.8 else (["Emphysema"] if not pos else [])
        preds.append(_pred(ppos))
    m = evaluate_arm([score_item(r, p) for r, p in zip(recs, preds)], n_boot=200)
    ci = m["ci95"]["micro_f1"]
    assert ci["lo"] <= m["micro_f1"] <= ci["hi"]
    assert 0 <= ci["lo"] < ci["hi"] <= 1
