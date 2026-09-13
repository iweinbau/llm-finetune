"""Prompt construction shared by training-data generation and every inference arm.

The same system prompt is used for the zero-shot base model, the few-shot base
model, the LoRA model and the frontier model, so differences in results come
from the model/adapter — not from prompt engineering per arm.
"""

from __future__ import annotations

import json

from .schema import LABELS

SYSTEM_PROMPT = (
    "You are a radiology report structuring assistant. You read one chest CT radiology "
    "report and return a single JSON object recording which abnormalities from a fixed list "
    "the report states are PRESENT, each with verbatim supporting evidence.\n\n"
    "Rules:\n"
    "1. Output ONLY the JSON object. No prose, no markdown fences, no explanations.\n"
    "2. \"abnormalities\" must contain exactly these 18 keys, each true or false:\n"
    + "\n".join(f"   - {k}" for k in LABELS)
    + "\n"
    "3. Mark an abnormality true only if the report states it is present. Findings that are "
    "explicitly negated (e.g. \"No pleural effusion\") or described as normal are false.\n"
    "4. \"evidence\" maps each TRUE abnormality to ONE short quote copied exactly, character "
    "for character, from the report (a sentence or fragment) that supports it. Never "
    "paraphrase or invent text. Do not include keys for false abnormalities.\n"
    "5. If you cannot point to a verbatim quote supporting an abnormality, it must be false.\n"
    "6. Output format: {\"abnormalities\": {...}, \"evidence\": {...}}"
)

USER_TEMPLATE = "REPORT:\n{report}\n\nReturn the JSON object."


def user_prompt(report: str) -> str:
    return USER_TEMPLATE.format(report=report.strip())


def target_text(target: dict) -> str:
    """Canonical serialisation of a training target (single line, stable key order)."""
    return json.dumps(target, ensure_ascii=False, separators=(", ", ": "))


def build_messages(
    report: str,
    target: dict | None = None,
    shots: list[dict] | None = None,
) -> list[dict]:
    """Chat messages for one example.

    shots: optional list of {"report": str, "target": dict} used as in-context examples
           (few-shot arm). They are inserted as prior user/assistant turns.
    target: when given, appended as the assistant turn (training format).
    """
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for s in shots or []:
        msgs.append({"role": "user", "content": user_prompt(s["report"])})
        msgs.append({"role": "assistant", "content": target_text(s["target"])})
    msgs.append({"role": "user", "content": user_prompt(report)})
    if target is not None:
        msgs.append({"role": "assistant", "content": target_text(target)})
    return msgs
