import json

from llm_finetune.parsing import extract_json, parse_prediction
from llm_finetune.schema import LABELS


def _full(positives=(), evidence=None):
    return {
        "abnormalities": {k: (k in positives) for k in LABELS},
        "evidence": evidence or {},
    }


def test_clean_json_parses():
    obj = _full(["Emphysema"], {"Emphysema": "Paraseptal emphysema is observed."})
    p = parse_prediction(json.dumps(obj))
    assert p.parse_ok and p.schema_ok
    assert p.positives() == ["Emphysema"]
    assert p.evidence["Emphysema"].startswith("Paraseptal")
    assert p.issues == []


def test_strips_think_block_and_fences():
    obj = _full(["Lung nodule"])
    text = "<think>\nsome reasoning\n</think>\n\n```json\n" + json.dumps(obj) + "\n```"
    p = parse_prediction(text)
    assert p.parse_ok and p.schema_ok
    assert p.positives() == ["Lung nodule"]


def test_prose_around_object():
    obj = _full()
    p = parse_prediction("Sure! Here is the JSON:\n" + json.dumps(obj) + "\nLet me know if you need more.")
    assert p.parse_ok and p.schema_ok and p.positives() == []


def test_missing_labels_flagged_but_scored():
    obj = {"abnormalities": {"Emphysema": True}, "evidence": {}}
    p = parse_prediction(json.dumps(obj))
    assert p.parse_ok and not p.schema_ok
    assert any(i.startswith("missing_labels") for i in p.issues)
    assert p.abnormalities["Emphysema"] is True
    assert p.abnormalities["Cardiomegaly"] is False


def test_coerces_ints_and_strings():
    abn = {k: 0 for k in LABELS}
    abn["Atelectasis"] = 1
    abn["Consolidation"] = "true"
    p = parse_prediction(json.dumps({"abnormalities": abn, "evidence": {}}))
    assert p.schema_ok
    assert set(p.positives()) == {"Atelectasis", "Consolidation"}


def test_unparseable_is_empty_prediction():
    p = parse_prediction("I cannot help with that.")
    assert not p.parse_ok and not p.schema_ok
    assert p.positives() == []
    assert "no_object" in p.issues


def test_trailing_comma_repair():
    text = '{"abnormalities": {' + ", ".join(f'"{k}": false' for k in LABELS) + ',}, "evidence": {},}'
    obj, err = extract_json(text)
    assert err is None and isinstance(obj, dict)


def test_unknown_label_recorded():
    obj = _full()
    obj["abnormalities"]["Pneumothorax"] = True
    p = parse_prediction(json.dumps(obj))
    assert p.schema_ok  # all 18 present
    assert any(i.startswith("unknown_label") for i in p.issues)


def test_truncated_output_is_labelled():
    text = '{"abnormalities": {"Emphysema": true, "Cardiomegaly": false}, "evidence": {"Emphysema": "Paraseptal emph'
    obj, err = extract_json(text)
    assert obj is None and err == "truncated"
    p = parse_prediction(text)
    assert not p.parse_ok and "truncated" in p.issues


def test_corrupted_string_is_labelled():
    text = '{"abnormalities": {"Emphysema": true, "Interlobular septal thick\'n\': false}, "evidence": {}}'
    obj, err = extract_json(text)
    assert obj is None and err == "corrupt_string"
