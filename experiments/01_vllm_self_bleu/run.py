"""Evaluate OLMo-3 checkpoints with vLLM and compute self-BLEU on GSM8K + CSQA."""

from pathlib import Path

from allegro.vllm_self_bleu import EvalConfig, MODELS, run


if __name__ == "__main__":
    run(
        EvalConfig(
            output_dir=Path(__file__).resolve().parent / "results",
            models=MODELS,
            tasks=["gsm8k", "commonsense_qa"],
            num_samples=100,
            n_responses=5,
        )
    )
