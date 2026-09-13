"""Task schema: the 18 CT-RATE abnormality labels, the output JSON contract,
and the text-normalisation used everywhere a quote is checked against a report.

Keeping all of this in one module means the prompt, the silver-label builder,
the parser and the metrics can never drift apart on what counts as "the same
label" or "the same text".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# The 18 multi-abnormality labels released with CT-RATE
# (dataset/multi_abnormality_labels/*_predicted_labels.csv), in the CSV column order.
LABELS: list[str] = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]

LABEL_SET = frozenset(LABELS)

# Minimum length of an evidence quote before it counts as "supported".
# One- or two-word quotes ("nodule") trivially appear in almost any report and
# would let a model game the faithfulness metric.
MIN_EVIDENCE_WORDS = 3

# Keyword patterns used ONLY by the heuristic silver-evidence builder to pick a
# candidate sentence for a label that CT-RATE marks positive. They are not used
# at evaluation time, so a model cannot "learn the metric".
LABEL_PATTERNS: dict[str, str] = {
    "Medical material": r"catheter|stent|pacemaker|\bport\b|prosthe|sternotomy|\bwire|\bclip|implant|"
    r"\btube\b|valve replacement|\bICD\b|electrode|surgical material|medical material|"
    r"foreign body|\bgraft|drain\b|suture",
    "Arterial wall calcification": r"(aort|arter)\w*[^.]{0,40}calcif|calcif\w*[^.]{0,40}(aort|arter)|"
    r"atheroscleros|atheromatous|calcific plaque",
    "Cardiomegaly": r"cardiomegaly|enlarged heart|cardiac enlargement|"
    r"heart (size|contour|silhouette)[^.]{0,40}(enlarg|increas|large)",
    "Pericardial effusion": r"pericardial (effusion|fluid)",
    "Coronary artery wall calcification": r"coronary[^.]{0,60}calcif|calcif[^.]{0,60}coronary",
    "Hiatal hernia": r"hiat(al|us) hernia",
    "Lymphadenopathy": r"lymphadenopathy|(enlarg|increas|pathologic)\w*[^.]{0,40}lymph node|"
    r"lymph node\w*[^.]{0,60}(enlarg|increas|pathologic|short axis|\d+\s?mm)",
    "Emphysema": r"emphysema|emphysematous",
    "Atelectasis": r"atelecta",
    "Lung nodule": r"nodul",
    "Lung opacity": r"opacit|ground[- ]glass|densit|infiltrat",
    "Pulmonary fibrotic sequela": r"fibro|sequela|\bscar|\bband\b|parenchymal distortion",
    "Pleural effusion": r"pleural (effusion|fluid)|effusion[^.]{0,30}pleura",
    "Mosaic attenuation pattern": r"mosaic",
    "Peribronchial thickening": r"peribronchial|bronchial wall thick",
    "Consolidation": r"consolidat",
    "Bronchiectasis": r"bronchiectasi|bronchiectatic",
    "Interlobular septal thickening": r"septal thick|interlobular",
}

# Simple negation cues; a candidate sentence containing one of these is deprioritised.
NEGATION_RE = re.compile(
    r"\b(no|not|without|absent|free of|negative for|unremarkable|normal|non[- ]?\w*|"
    r"rule[sd]? out|ruled out|excluded)\b",
    re.IGNORECASE,
)

_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def normalize_text(text: str) -> str:
    """Whitespace-collapse, lowercase, strip surrounding quotes. Used for quote matching."""
    if text is None:
        return ""
    t = str(text).replace("\u00a0", " ")  # non-breaking spaces -> plain spaces
    t = _WS_RE.sub(" ", t).strip()
    t = t.strip("\"'“”‘’ ")
    return t.lower()


def split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT_RE.split(text or "")
    return [p.strip() for p in parts if p and p.strip()]


def quote_in_report(quote: str, report: str) -> bool:
    """True if `quote` appears verbatim (after normalisation) inside `report`."""
    q = normalize_text(quote)
    return bool(q) and q in normalize_text(report)


def word_count(text: str) -> int:
    return len(normalize_text(text).split())


@dataclass
class Prediction:
    """A parsed model output. `abnormalities` always has all 18 keys (missing -> False)."""

    abnormalities: dict[str, bool]
    evidence: dict[str, str | None]
    parse_ok: bool = True  # JSON could be extracted and loaded
    schema_ok: bool = True  # all 18 keys present with boolean-coercible values
    issues: list[str] = field(default_factory=list)
    raw: str = ""

    @classmethod
    def empty(cls, raw: str = "", issue: str = "unparseable") -> Prediction:
        return cls(
            abnormalities={k: False for k in LABELS},
            evidence={},
            parse_ok=False,
            schema_ok=False,
            issues=[issue],
            raw=raw,
        )

    def positives(self) -> list[str]:
        return [k for k in LABELS if self.abnormalities.get(k, False)]


def target_json(labels: dict[str, int | bool], evidence: dict[str, str] | None) -> dict:
    """Build the canonical training target object (label order fixed, compact)."""
    abn = {k: bool(labels.get(k, 0)) for k in LABELS}
    ev = {}
    if evidence:
        for k in LABELS:
            if abn[k] and evidence.get(k):
                ev[k] = evidence[k]
    return {"abnormalities": abn, "evidence": ev}
