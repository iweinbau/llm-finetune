"""Optional frontier-model arm (Claude via the Anthropic API).

Same system prompt, same user prompt, same parser and metrics as the local arms,
so the comparison is like-for-like. Responses are appended to the output file as
they arrive, so an interrupted run resumes.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .data import read_jsonl
from .prompts import SYSTEM_PROMPT, user_prompt


def run_claude(
    test_path: str | Path,
    out_path: str | Path,
    arm: str = "claude",
    model: str = "claude-sonnet-5",
    max_tokens: int = 700,
    max_workers: int = 4,
    limit: int | None = None,
) -> Path:
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pip install 'llm-finetune[frontier]' for the Claude arm") from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic()
    test = read_jsonl(test_path)
    if limit:
        test = test[:limit]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for r in test if r["id"] not in done]
    print(f"[{arm}] {len(todo)} reports to run with {model} ({len(done)} cached)")

    def call(rec: dict) -> dict:
        for attempt in range(6):
            try:
                msg = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt(rec["report"])}],
                )
                text = "".join(getattr(b, "text", "") for b in msg.content)
                return {
                    "id": rec["id"],
                    "arm": arm,
                    "model": model,
                    "adapter": None,
                    "shots": 0,
                    "raw": text,
                    "prompt_tokens": msg.usage.input_tokens,
                    "gen_tokens": msg.usage.output_tokens,
                }
            except anthropic.RateLimitError:  # pragma: no cover
                time.sleep(2**attempt)
            except anthropic.APIStatusError as e:  # pragma: no cover
                if e.status_code >= 500:
                    time.sleep(2**attempt)
                else:
                    raise
        return {"id": rec["id"], "arm": arm, "model": model, "raw": "", "error": "gave_up"}

    with ThreadPoolExecutor(max_workers=max_workers) as ex, out_path.open("a", encoding="utf-8") as fh:
        futs = [ex.submit(call, r) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            fh.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"[{arm}] {i}/{len(todo)}")
    return out_path
