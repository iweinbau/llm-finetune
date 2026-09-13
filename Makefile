# One command per pipeline stage. Run them top to bottom.
# Everything is idempotent / resumable: re-running a stage picks up where it stopped.

# Pick the newest suitable interpreter on PATH; override with `make setup PY=/path/to/python3.12`
PY        ?= $(shell command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
MODEL     ?= mlx-community/Qwen3-4B-4bit
CONFIG    ?= configs/lora_qwen3_4b.yaml
ADAPTER   ?= adapters/qwen3-4b-lora
N_TEST    ?= 500
LIMIT     ?=            # e.g. `make predict-all LIMIT=20` for a smoke test
LIM_FLAG   = $(if $(LIMIT),--limit $(LIMIT),)

.PHONY: setup data silver train retry-invalid predict-base predict-fewshot predict-lora predict-claude predict-all evaluate site serve audit test lint clean

setup:              ## create venv + install (Apple Silicon: includes mlx-lm)
	@echo "using $(PY) ($$($(PY) --version 2>&1))"
	@$(PY) -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10 (brew install python@3.12, or: make setup PY=python3.12)"'
	rm -rf .venv && $(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e ".[mlx,dev]"
	@echo "Now: huggingface-cli login   (accept the CT-RATE terms on the dataset page first)"

data:               ## download CT-RATE CSVs and build splits with heuristic evidence
	llm-finetune prepare --download --n-test $(N_TEST)

silver:             ## optional: replace heuristic evidence with Claude-verified quotes (needs ANTHROPIC_API_KEY)
	llm-finetune silver

train:              ## LoRA fine-tune on Apple Silicon
	mlx_lm.lora -c $(CONFIG)

predict-base:       ## arm A — base model, zero-shot
	llm-finetune predict --arm base_zeroshot --model $(MODEL) $(LIM_FLAG)

predict-fewshot:    ## arm B — base model, 3 in-context examples (long prompts -> no batching)
	llm-finetune predict --arm base_fewshot --model $(MODEL) --shots 3 --sequential $(LIM_FLAG)

predict-lora:       ## arm C — fine-tuned adapter, zero-shot
	llm-finetune predict --arm lora --model $(MODEL) --adapter $(ADAPTER) $(LIM_FLAG)

predict-claude:     ## arm D (optional) — frontier model, zero-shot
	llm-finetune predict --arm claude --backend claude --model claude-sonnet-5 $(LIM_FLAG)

predict-all: predict-base predict-fewshot predict-lora

retry-invalid:      ## regenerate only the rows that failed to parse, for every arm that has predictions
	@[ -f results/preds_base_zeroshot.jsonl ] && llm-finetune predict --arm base_zeroshot --model $(MODEL) --retry-invalid || true
	@[ -f results/preds_base_fewshot.jsonl ] && llm-finetune predict --arm base_fewshot --model $(MODEL) --shots 1 --sequential --retry-invalid --max-tokens 50000 || true
	@[ -f results/preds_lora.jsonl ] && llm-finetune predict --arm lora --model $(MODEL) --adapter $(ADAPTER) --retry-invalid || true

evaluate:           ## score every results/preds_*.jsonl that exists
	llm-finetune evaluate --preds $(wildcard results/preds_*.jsonl)

site: evaluate      ## refresh docs/results.js for GitHub Pages
	llm-finetune site

serve:              ## preview the site locally
	cd docs && $(PY) -m http.server 8000

audit:              ## make the human audit sheet for the LoRA arm
	llm-finetune audit make --preds results/preds_lora.jsonl --out results/audit_lora.csv --n 60

test:
	pytest

lint:
	ruff check src tests

clean:
	rm -rf results/preds_*.jsonl results/results.json results/samples.json results/errors_*.csv
