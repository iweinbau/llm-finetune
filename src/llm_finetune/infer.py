"""Local inference with MLX (Apple Silicon) for the base and LoRA arms.

Writes one JSONL row per test report:
  {"id", "arm", "model", "adapter", "shots", "raw", "prompt_tokens", "gen_tokens"}

`raw` is the untouched model output; parsing happens at evaluation time so the
same predictions can be re-scored if the parser changes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .data import read_jsonl
from .prompts import build_messages
from .schema import LABELS


def pick_shots(train_records: list[dict], k: int, seed: int = 7) -> list[dict]:
    """Deterministic few-shot examples: mix of normal and abnormal training reports
    that have complete evidence (train_ok)."""
    import random

    rng = random.Random(seed)
    ok = [r for r in train_records if r.get("train_ok", True) and "target" in r]
    normal = [r for r in ok if not any(r["labels"].get(l) for l in LABELS)]
    abnormal = [r for r in ok if any(r["labels"].get(l) for l in LABELS)]
    rng.shuffle(normal)
    rng.shuffle(abnormal)
    n_abn = max(1, k - k // 3)
    chosen = abnormal[:n_abn] + normal[: k - n_abn]
    return [{"report": r["report"], "target": r["target"]} for r in chosen[:k]]


def run_mlx(
    model_id: str,
    test_path: str | Path,
    out_path: str | Path,
    arm: str,
    adapter_path: str | None = None,
    shots: int = 0,
    train_path: str | Path | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    batch_size: int = 8,
    limit: int | None = None,
    thinking: bool = False,
    resume: bool = True,
    sequential: bool = False,
    retry_invalid: bool = False,
) -> Path:
    """Generate one output per test report.

    sequential:    use mlx_lm.generate one prompt at a time instead of batch_generate. Slower,
                   but avoids a batched-generation glitch seen with prompts > ~2000 tokens
                   (few-shot arm), where output tokens get corrupted mid-JSON.
    retry_invalid: keep rows whose output parses, drop the rest, and regenerate only those.
                   Use after raising --max-tokens (truncation) or switching to --sequential.
    """
    try:
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as e:  # pragma: no cover
        raise SystemExit("mlx-lm is required: pip install 'llm-finetune[mlx]' (Apple Silicon only)") from e

    test = read_jsonl(test_path)
    if limit:
        test = test[:limit]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if resume and out_path.exists():
        rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if retry_invalid:
            from .parsing import parse_prediction

            keep = [r for r in rows if parse_prediction(r.get("raw", "")).parse_ok]
            n_bad = len(rows) - len(keep)
            out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
            print(f"[{arm}] retry-invalid: keeping {len(keep)} parseable rows, regenerating {n_bad}")
            rows = keep
        done = {r["id"] for r in rows}
        if not retry_invalid:
            print(f"[{arm}] resuming: {len(done)} already done")
    todo = [r for r in test if r["id"] not in done]
    if not todo:
        print(f"[{arm}] nothing to do")
        return out_path

    shot_examples: list[dict] = []
    if shots > 0:
        if not train_path:
            raise SystemExit("--shots requires --train-path for in-context examples")
        shot_examples = pick_shots(read_jsonl(train_path), shots)

    print(f"[{arm}] loading {model_id}" + (f" + adapter {adapter_path}" if adapter_path else ""))
    model, tokenizer = load(model_id, adapter_path=adapter_path)
    sampler = make_sampler(temp=temperature)

    def encode(rec: dict) -> list[int]:
        msgs = build_messages(rec["report"], shots=shot_examples)
        # enable_thinking is honoured by Qwen3-style templates and ignored by others.
        return tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, enable_thinking=thinking
        )

    from mlx_lm import generate

    batch_generate = None
    if not sequential:
        try:
            from mlx_lm import batch_generate  # mlx-lm >= 0.28
        except ImportError:  # pragma: no cover
            batch_generate = None
    if batch_generate is None:
        batch_size = 1
        print(f"[{arm}] sequential generation")

    from .parsing import parse_prediction

    t0 = time.time()
    n_suspect = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for start in range(0, len(todo), batch_size):
            chunk = todo[start : start + batch_size]
            prompts = [encode(r) for r in chunk]
            if batch_generate is not None:
                # Prefill the whole prompt in one step: chunked prefill (default 2048) is where the
                # long-prompt corruption was observed.
                step = max(2048, ((max(len(p) for p in prompts) + 255) // 256) * 256)
                resp = batch_generate(
                    model,
                    tokenizer,
                    prompts,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    verbose=False,
                    completion_batch_size=batch_size,
                    prefill_step_size=step,
                )
                texts = resp.texts
            else:
                texts = [
                    generate(model, tokenizer, prompt=p, max_tokens=max_tokens, sampler=sampler)
                    for p in prompts
                ]
            for text in texts:
                if not parse_prediction(text).parse_ok:
                    n_suspect += 1
            for rec, p, text in zip(chunk, prompts, texts):
                fh.write(
                    json.dumps(
                        {
                            "id": rec["id"],
                            "arm": arm,
                            "model": model_id,
                            "adapter": adapter_path,
                            "shots": shots,
                            "thinking": thinking,
                            "raw": text,
                            "prompt_tokens": len(p),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            fh.flush()
            done_n = start + len(chunk)
            el = time.time() - t0
            if n_suspect and done_n % (batch_size * 5) == 0:
                print(f"[{arm}]   {n_suspect} unparseable outputs so far — if many, try --sequential and/or --max-tokens 1024, then --retry-invalid")
            print(f"[{arm}] {done_n}/{len(todo)}  {el/60:.1f} min  ({el/done_n:.1f}s/report)")
    return out_path
