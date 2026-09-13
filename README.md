# Teaching a small language model to read CT reports

I fine-tuned a 4-billion-parameter open model (Qwen3-4B, LoRA, on a MacBook) so that it turns a radiologist's free-text chest CT report into a structured list of 18 findings **and backs every finding with a quote from the report**. Then I measured how often each model version claimed a finding it could not back up, before and after fine-tuning, against the same model with examples in the prompt, and against a frontier model.

**Live results page:** [https://iweinbau.github.io/llm-finetune/]([https://iweinbau.github.io/llm-finetune/)

<!-- RESULTS:START -->
Fine-tuning **cut** the share of findings the model claimed without evidence from **8.4%** to **1.2%**, and **raised** agreement with the reference labels from 87.2% to 95.2%, on 500 reports it had never seen.

| Version | Claims without evidence ↓ | Agreement with labels (F1) ↑ | All 18 findings right ↑ | Well-formed output ↑ | False alarms on normal reports ↓ |
|---|---|---|---|---|---|
| Before fine-tuning | 8.4% (7–11) | 87.2% (86–88) | 51.2% (47–55) | 99.8% | 2.7% |
| Before, with 3 examples in the prompt | 9.3% (8–11) | 84.6% (83–86) | 46.6% (42–51) | 90.2% | 0.0% |
| After fine-tuning (LoRA) | 1.2% (1–2) | 95.2% (94–96) | 75.4% (71–79) | 100.0% | 0.0% |

_500 held-out CT-RATE reports; parentheses are 95% bootstrap confidence intervals._
<!-- RESULTS:END -->

**The approach**

1. **Data.** Public chest-CT reports with 18 abnormality labels ([CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE), Hamamci et al.; see [Dataset credit](#dataset-credit-and-terms)). For training I attached a
   verbatim quote from the report to every positive finding, and dropped reports where that was not
   possible rather than teach the model to assert without evidence.
2. **Fine-tuning.** Rank-8 LoRA adapters on a 4-bit Qwen3-4B, trained with `mlx-lm` on Apple Silicon
   in about an hour. Compared against the same model un-tuned (zero-shot and 3-shot) and, optionally,
   Claude with identical instructions.
3. **Evaluation.** 500 reports from patients never seen in training. Every finding a model marks
   present is checked for a verbatim quote in the report; label agreement, false alarms on normal
   reports and output validity are scored alongside, with bootstrap confidence intervals. A clinician
   audit of a sample of quotes checks that "quoted" also means "supported".

---

## Why this task

Structured extraction from radiology reports is what a surgeon or planning engineer
actually needs before a case: which findings are present, where, how big. It is a
language task with public data and real ground truth, which the 3D-imaging side of
surgical planning does not have.

It also has a failure mode that matters: a model that *invents* a finding is worse than
one that misses it, because the invented finding looks exactly as confident as a real
one. So the output contract requires a verbatim quote for every positive claim, and the
evaluation checks every quote mechanically against the report. The headline metric is
not accuracy; it is the **unsupported-positive rate** — the share of positive claims the
model could not back with text that is actually in the report.

## What the model does

Input: one report (Findings + Impression). Output: one JSON object.

```json
{
  "abnormalities": {"Emphysema": true, "Lung nodule": true, "Pleural effusion": false, "...": false},
  "evidence": {
    "Emphysema":   "Paraseptal emphysema is observed in the upper lobes of both lungs.",
    "Lung nodule": "A 6 mm solid nodule is seen in the right lower lobe."
  }
}
```

The 18 keys are CT-RATE's released abnormality labels (`llm-finetune labels` prints them).

## The four models

| Model | Model description |
|---|---|
| `base_zeroshot` | Qwen3-4B-4bit, system prompt only |
| `base_fewshot` | same model, 3 in-context examples |
| `lora` | same model + rank-8 LoRA finetuned |
| `claude` (optional) | Claude via API, identical prompt |

## Quickstart (Apple Silicon)

```bash
# 0. Accept the CT-RATE terms once at https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
git clone https://github.com/<you>/llm-finetune && cd llm-finetune
make setup && source .venv/bin/activate
huggingface-cli login

# 1. Data (downloads ~90 MB of CSVs; no image volumes)
make data                       # 3000 train / 200 valid / 500 test reports, heuristic evidence

# 3. Train
make train

# 4. Run the models on the full test set, score, publish
make predict-all                # base zero-shot, base 3-shot, LoRA
make predict-claude             # optional; needs ANTHROPIC_API_KEY
make retry-invalid              # regenerate only outputs that were cut off or corrupted (minutes)
make site                       # evaluate + refresh docs/results.js and 
```

Machines with 8–16 GB: use `make train CONFIG=configs/lora_qwen3_1p7b.yaml` and
`MODEL=mlx-community/Qwen3-1.7B-4bit ADAPTER=adapters/qwen3-1.7b-lora` for the predict targets.

## Evaluation protocol

*Reading the results.* No single number says whether a model is good; the design is a comparison
on identical reports, so a difference between versions is a difference between models. The number
that matters most is **claims without evidence** (findings marked present without a verbatim quote
from the report, lower is better), read together with **agreement with the reference labels**
(micro-F1, higher is better). A fine-tune has worked if it clearly lowers the first without
lowering the second. `llm-finetune evaluate` prints a plain-language summary of your numbers and the
deltas against the un-tuned model after every run.


*Test set.* 500 reports from CT-RATE's **validation** patients (never seen in training;
one scan per patient). Training and loss-monitoring data come from CT-RATE's training
patients only.

*Label agreement.* Per-label precision/recall/F1, micro-F1 (primary), macro-F1 over
labels with at least one gold positive, exact-match rate over all 18 labels, and the
false-alarm rate on reports with no reference abnormality. 95% bootstrap CIs over reports.

*Faithfulness.* For every `true` the model emits, its quote is classified as
`supported` (≥3 words, occurs verbatim in the report after whitespace/case
normalisation), `too_short`, `missing`, or `fabricated`. **Unsupported-positive rate** =
1 − supported / all positives. This needs no labels at all.

*Validity.* JSON-parse rate and schema-valid rate. Unparseable or schema-invalid outputs
are scored as *all false, no evidence*, so a model gains nothing by refusing to answer.

*Human audit.* The verbatim check proves a quote exists, not that it means what the
label says ("no pleural effusion" quoted for *Pleural effusion* would pass).
`make audit` samples 60 "supported" quotes from the LoRA arm into a CSV; a clinician
marks each `yes / no / partial`; `llm-finetune audit score` adds the human-verified support
rate to the results. This is the step where medical background turns a proxy metric into
a measured one.

## Dataset credit and terms

All reports and abnormality labels come from **CT-RATE**, created and released by Ibrahim Ethem
Hamamci, Sezgin Er, Bjoern Menze and colleagues (University of Zurich / TUM). This project would
not exist without their work of collecting, translating and labelling the reports and making them
public: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE

If you use this repository or its results, please cite the dataset authors. BibTeX for the five
papers they ask to be cited is in [`CITATION.bib`](CITATION.bib):

- Hamamci et al., *Generalist foundation models from a multimodal dataset for 3D computed tomography*, Nature Biomedical Engineering, 2026.
- Hamamci et al., *GenerateCT: Text-conditional generation of 3D chest CT volumes*, ECCV 2024.
- Hamamci, Er, Menze, *CT2Rep: Automated radiology report generation for 3D medical imaging*, MICCAI 2024.
- Hamamci et al., *Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging*, NeurIPS 2025.
- Hamamci et al., *CRG Score: A Distribution-Aware Clinical Metric for Radiology Report Generation*, MIDL 2025 (short papers).

CT-RATE is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0
license (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). In practice for
this repository:

- **Attribution** — the credit above and in the page footer, plus `CITATION.bib`.
- **NonCommercial** — this is a non-commercial portfolio/research project; do not reuse the data,
  the processed splits or the trained adapters commercially.
- **ShareAlike** — everything derived from the dataset is shared under the same CC BY-NC-SA 4.0
  terms: the processed splits in `data/`, the silver evidence quotes, the report excerpts in
  `docs/results.js`, and any LoRA adapters you publish. The code itself is separate (MIT, below).

The dataset is gated: accept the terms on the dataset page before `make data`. Raw CSVs,
processed splits, predictions and `results/` are git-ignored and should stay out of a public repo;
the ~40 sample report excerpts that `make site` places in `docs/results.js` are redistributed under
CC BY-NC-SA 4.0 with attribution, which the license permits.
