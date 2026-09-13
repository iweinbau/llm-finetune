"""llm-finetune — structured, evidence-grounded findings from chest CT reports.

Pipeline:  prepare -> (silver) -> train (mlx_lm.lora) -> predict -> evaluate -> site
"""

__version__ = "0.1.0"
