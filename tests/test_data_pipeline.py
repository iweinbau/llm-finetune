import json

from llm_finetune import data
from llm_finetune.cli import main
from llm_finetune.prompts import SYSTEM_PROMPT, build_messages
from llm_finetune.schema import LABELS
from llm_finetune.silver import attach_evidence, heuristic_evidence


def test_parse_volume_name():
    assert data.parse_volume_name("train_3758_a_1.nii.gz") == ("train_3758", "train_3758_a", 1)
    assert data.parse_volume_name("valid_12_b_3.nii.gz") == ("valid_12", "valid_12_b", 3)


def test_load_split_collapses_reconstructions(raw_dir):
    paths = {k: raw_dir / v for k, v in data.FILES.items()}
    df = data.load_split(paths["train_reports"], paths["train_labels"])
    assert df["scan_id"].is_unique
    assert set(LABELS) <= set(df.columns)
    assert df["report"].str.startswith("FINDINGS:").all()
    assert (df[LABELS].isin([0, 1])).all().all()


def test_splits_are_patient_disjoint(raw_dir):
    paths = {k: raw_dir / v for k, v in data.FILES.items()}
    tr = data.load_split(paths["train_reports"], paths["train_labels"])
    va = data.load_split(paths["valid_reports"], paths["valid_labels"])
    s = data.make_splits(tr, va, n_train=30, n_valid=10, n_test=15, seed=0)
    p_train = {r["patient"] for r in s["train"]}
    p_valid = {r["patient"] for r in s["valid"]}
    p_test = {r["patient"] for r in s["test"]}
    assert len(s["train"]) == 30 and len(s["valid"]) == 10 and len(s["test"]) == 15
    assert not (p_train & p_valid) and not (p_train & p_test) and not (p_valid & p_test)
    assert all(r["patient"].startswith("valid_") for r in s["test"])


def test_heuristic_evidence_prefers_positive_sentence():
    report = (
        "FINDINGS: No pleural effusion on the right. There is a small amount of pleural effusion on the left. "
        "Paraseptal emphysema is observed.\nIMPRESSION: Effusion."
    )
    ev = heuristic_evidence(report, {"Pleural effusion": 1, "Emphysema": 1, "Cardiomegaly": 0})
    assert ev["Pleural effusion"].startswith("There is a small amount")
    assert ev["Emphysema"].startswith("Paraseptal")
    assert "Cardiomegaly" not in ev


def test_attach_evidence_drops_incomplete():
    recs = [
        {"id": "a", "report": "FINDINGS: Paraseptal emphysema is observed.", "labels": {"Emphysema": 1}},
        {"id": "b", "report": "FINDINGS: Lungs are clear.", "labels": {"Emphysema": 1}},  # no evidence possible
        {"id": "c", "report": "FINDINGS: Lungs are clear.", "labels": {}},  # normal, always kept
    ]
    ev = {r["id"]: heuristic_evidence(r["report"], r["labels"]) for r in recs}
    stats = attach_evidence(recs, ev)
    assert stats["n_kept"] == 2 and stats["n_dropped_missing_evidence"] == 1
    assert [r["train_ok"] for r in recs] == [True, False, True]
    assert stats["missing_evidence_by_label"]["Emphysema"] == 1


def test_build_messages_training_and_inference():
    tgt = {"abnormalities": {k: False for k in LABELS}, "evidence": {}}
    train_msgs = build_messages("FINDINGS: normal.", target=tgt)
    assert [m["role"] for m in train_msgs] == ["system", "user", "assistant"]
    assert train_msgs[0]["content"] == SYSTEM_PROMPT
    assert json.loads(train_msgs[-1]["content"]) == tgt
    infer_msgs = build_messages("FINDINGS: normal.", shots=[{"report": "R", "target": tgt}])
    assert [m["role"] for m in infer_msgs] == ["system", "user", "assistant", "user"]


def test_prepare_cli_end_to_end(raw_dir, tmp_path):
    out = tmp_path / "processed"
    main(
        [
            "prepare",
            "--raw-dir", str(raw_dir),
            "--out-dir", str(out),
            "--n-train", "30", "--n-valid", "8", "--n-test", "12", "--seed", "1",
        ]
    )
    for name in ("train", "valid", "test"):
        assert (out / f"{name}.jsonl").exists()
    mlx_train = data.read_jsonl(out / "mlx" / "train.jsonl")
    assert mlx_train and set(mlx_train[0]) == {"messages"}
    assert mlx_train[0]["messages"][-1]["role"] == "assistant"
    # every training target's evidence is verbatim in its report
    for r in data.read_jsonl(out / "train.jsonl"):
        if r.get("train_ok"):
            for k, q in r["target"]["evidence"].items():
                assert q.lower() in r["report"].lower()
                assert r["labels"][k] == 1
    test = data.read_jsonl(out / "test.jsonl")
    assert "messages" not in test[0] and "evidence" not in test[0]
    stats = json.loads((out / "prepare_stats.json").read_text())
    assert stats["splits"]["train"]["n_records"] == 30
