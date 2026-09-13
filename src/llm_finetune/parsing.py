"""Lenient-but-honest parsing of model output into a `Prediction`.

Lenient: we tolerate markdown fences, a leading <think> block, stray prose
before/after the object, 0/1 or "true"/"false" instead of booleans.

Honest: every deviation from the contract is recorded in `issues`, and the
validity rates are reported as first-class metrics, so leniency here never
hides a model that cannot follow the format.
"""

from __future__ import annotations

import json
import re

from .schema import LABELS, Prediction

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_TRUE = {"true", "1", "yes", "present", "y"}
_FALSE = {"false", "0", "no", "absent", "n", "none", "null"}


def _first_json_object(text: str) -> str | None:
    """Return the substring of the first balanced {...} object, or None."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> tuple[dict | None, str | None]:
    """Extract and load the first JSON object from free text. Returns (obj, error)."""
    if text is None:
        return None, "empty"
    t = _THINK_RE.sub("", text).strip()
    m = _FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()
    candidate = _first_json_object(t)
    if candidate is None:
        if "{" not in t:
            return None, "no_object"
        # An opening brace with no balanced close. If the text still ends with a closing brace
        # (or restarts a second object) the model finished but corrupted a string mid-way;
        # otherwise the generation was cut off by the token budget.
        finished = t.rstrip().endswith("}") or t.count('"abnormalities"') > 1
        return None, "corrupt_string" if finished else "truncated"
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as e:
        # Common failure: trailing commas. One repair attempt, recorded as an issue upstream.
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            obj = json.loads(repaired)
        except json.JSONDecodeError:
            return None, f"json_error:{e.msg}"
    if not isinstance(obj, dict):
        return None, "not_an_object"
    return obj, None


def _coerce_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
    return None


def _norm_key(k: str) -> str:
    return re.sub(r"\s+", " ", str(k)).strip().lower()


_LABEL_BY_NORM = {_norm_key(k): k for k in LABELS}


def parse_prediction(text: str) -> Prediction:
    obj, err = extract_json(text)
    if obj is None:
        return Prediction.empty(raw=text or "", issue=err or "unparseable")

    issues: list[str] = []
    abn_raw = obj.get("abnormalities")
    if not isinstance(abn_raw, dict):
        # Some models flatten the object; accept top-level label keys as a fallback.
        abn_raw = {k: v for k, v in obj.items() if _norm_key(k) in _LABEL_BY_NORM}
        if abn_raw:
            issues.append("flattened_abnormalities")
        else:
            issues.append("missing_abnormalities")
            abn_raw = {}

    abnormalities: dict[str, bool] = {}
    seen: set[str] = set()
    for k, v in abn_raw.items():
        canon = _LABEL_BY_NORM.get(_norm_key(k))
        if canon is None:
            issues.append(f"unknown_label:{k}")
            continue
        b = _coerce_bool(v)
        if b is None:
            issues.append(f"bad_value:{canon}")
            b = False
        abnormalities[canon] = b
        seen.add(canon)
    missing = [k for k in LABELS if k not in seen]
    if missing:
        issues.append(f"missing_labels:{len(missing)}")
        for k in missing:
            abnormalities[k] = False

    ev_raw = obj.get("evidence")
    evidence: dict[str, str | None] = {}
    if ev_raw is None:
        issues.append("missing_evidence")
    elif isinstance(ev_raw, dict):
        for k, v in ev_raw.items():
            canon = _LABEL_BY_NORM.get(_norm_key(k))
            if canon is None:
                issues.append(f"unknown_evidence_label:{k}")
                continue
            if isinstance(v, list):
                v = " ".join(str(x) for x in v)
            evidence[canon] = None if v is None else str(v)
    else:
        issues.append("evidence_not_object")

    schema_ok = not any(
        i.startswith(("missing_abnormalities", "missing_labels", "bad_value", "flattened"))
        for i in issues
    )
    return Prediction(
        abnormalities={k: abnormalities[k] for k in LABELS},
        evidence=evidence,
        parse_ok=True,
        schema_ok=schema_ok,
        issues=issues,
        raw=text,
    )
