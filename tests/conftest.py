"""Synthetic CT-RATE-shaped fixtures. No real patient data is used or committed."""

from __future__ import annotations

import random

import pandas as pd
import pytest

from llm_finetune.schema import LABELS

NORMAL_FINDINGS = (
    "Trachea and both main bronchi are patent. No mass or infiltration is detected in the lung "
    "parenchyma. Mediastinal structures are within normal limits. Heart size is normal. "
    "No pleural or pericardial effusion. No pathologically enlarged lymph nodes."
)

ABNORMAL_SENTENCES = {
    "Emphysema": "Paraseptal emphysema is observed in the upper lobes of both lungs.",
    "Lung nodule": "A 6 mm solid nodule is seen in the right lower lobe.",
    "Pleural effusion": "There is a small amount of pleural effusion on the left.",
    "Atelectasis": "Linear atelectasis is present in the bilateral lower lobes.",
    "Cardiomegaly": "Heart size is increased, consistent with cardiomegaly.",
    "Coronary artery wall calcification": "Calcific plaques are noted in the coronary artery walls.",
    "Hiatal hernia": "A hiatal hernia is present.",
    "Lymphadenopathy": "Mediastinal lymph nodes are enlarged, the largest measuring 14 mm in short axis.",
    "Consolidation": "Consolidation is seen in the right middle lobe.",
    "Bronchiectasis": "Bronchiectatic changes are noted in the lower lobes.",
}


def make_report(positives: list[str]) -> tuple[str, str]:
    sents = [ABNORMAL_SENTENCES[p] for p in positives]
    findings = ("Trachea and both main bronchi are patent. " + " ".join(sents)).strip()
    if not positives:
        findings = NORMAL_FINDINGS
    impression = ", ".join(positives) + "." if positives else "Findings within normal limits."
    return findings, impression


def synth_frames(n_patients: int, prefix: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    rep_rows, lab_rows = [], []
    keys = list(ABNORMAL_SENTENCES)
    for i in range(1, n_patients + 1):
        n_scans = rng.choice([1, 1, 2])
        for scan in "ab"[:n_scans]:
            k = rng.choice([0, 0, 1, 2, 3])
            positives = rng.sample(keys, k)
            findings, impression = make_report(positives)
            for recon in (1, 2):  # two reconstructions share one report
                vol = f"{prefix}_{i}_{scan}_{recon}.nii.gz"
                rep_rows.append(
                    {
                        "VolumeName": vol,
                        "ClinicalInformation_EN": "Cough.",
                        "Technique_EN": "Axial MDCT.",
                        "Findings_EN": findings,
                        "Impressions_EN": impression,
                    }
                )
                lab_rows.append({"VolumeName": vol, **{l: int(l in positives) for l in LABELS}})
    return pd.DataFrame(rep_rows), pd.DataFrame(lab_rows)


@pytest.fixture
def raw_dir(tmp_path):
    from llm_finetune.data import FILES

    tr_rep, tr_lab = synth_frames(60, "train", 1)
    va_rep, va_lab = synth_frames(25, "valid", 2)
    paths = {k: tmp_path / v for k, v in FILES.items()}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    tr_rep.to_csv(paths["train_reports"], index=False)
    tr_lab.to_csv(paths["train_labels"], index=False)
    va_rep.to_csv(paths["valid_reports"], index=False)
    va_lab.to_csv(paths["valid_labels"], index=False)
    return tmp_path
