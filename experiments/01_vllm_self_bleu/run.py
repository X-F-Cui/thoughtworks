"""Evaluate OLMo-3 checkpoints with vLLM and compute self-BLEU on GSM8K + CSQA."""

from __future__ import annotations

import argparse
from pathlib import Path

from allegro.vllm_self_bleu import EvalConfig, MODELS, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory where per-model/task JSON outputs are written.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Space-separated HF model names to evaluate.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["gsm8k", "commonsense_qa"],
        choices=["gsm8k", "commonsense_qa"],
        help="Tasks to run.",
    )
    parser.add_argument("--num-samples", type=int, default=100, help="Examples per task split to evaluate.")
    parser.add_argument("--n-responses", type=int, default=5, help="Generations sampled per question.")
    parser.add_argument("--max-model-len", type=int, default=4096, help="vLLM max_model_len.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor_parallel_size.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to allow tokenizer/model remote code in HF repos (recommended for OLMo).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        EvalConfig(
            output_dir=args.output_dir,
            models=args.models,
            tasks=args.tasks,
            num_samples=args.num_samples,
            n_responses=args.n_responses,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=args.trust_remote_code,
        )
    )
