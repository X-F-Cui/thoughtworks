"""Run vLLM generations for QA tasks and compute self-BLEU."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


MODELS = [
    "allenai/Olmo-3-1025-7B",
    "allenai/Olmo-3-7B-Instruct-SFT",
    "allenai/Olmo-3-7B-Instruct-DPO",
    "allenai/Olmo-3-7B-Instruct",
]


@dataclass
class EvalConfig:
    output_dir: Path
    models: list[str]
    tasks: list[str]
    num_samples: int = 5
    n_responses: int = 5
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = True


def get_prompt_and_stop(task: str, doc: dict[str, Any]) -> tuple[str, list[str]]:
    if task == "gsm8k":
        question = doc["question"].strip()
        prompt = (
            "Question: "
            f"{question}\n"
            "Answer:"
        )
        # LM Eval uses generate_until-style stopping for gsm8k.
        return prompt, ["\nQuestion:"]

    if task == "commonsense_qa":
        choices = "\n".join(
            f"{label}. {text}"
            for label, text in zip(doc["choices"]["label"], doc["choices"]["text"], strict=True)
        )
        prompt = (
            "You are solving a multiple-choice commonsense reasoning problem. "
            "Think step by step and end with 'Final Answer: <option letter>'.\n\n"
            f"Question: {doc['question'].strip()}\n"
            f"Choices:\n{choices}\n"
            "Reasoning:"
        )
        # Convert CSQA to generate_until by adding a stop sequence like lm-eval generation tasks.
        return prompt, ["\nQuestion:"]

    raise ValueError(f"Unsupported task: {task}")


def load_task_docs(task: str, num_samples: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    if task == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
    elif task == "commonsense_qa":
        ds = load_dataset("tau/commonsense_qa", split="validation")
    else:
        raise ValueError(f"Unsupported task: {task}")

    return [ds[i] for i in range(min(num_samples, len(ds)))]


def compute_self_bleu(responses: list[str]) -> float:
    if len(responses) <= 1:
        return 0.0

    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    smoothie = SmoothingFunction().method1
    tokenized = [r.split() for r in responses]
    scores: list[float] = []

    for idx, hypothesis in enumerate(tokenized):
        references = [cand for j, cand in enumerate(tokenized) if j != idx]
        if not references or not hypothesis:
            scores.append(0.0)
            continue
        score = sentence_bleu(references, hypothesis, smoothing_function=smoothie)
        scores.append(float(score))

    return sum(scores) / len(scores)


def run_model_on_task(model_name: str, task: str, config: EvalConfig) -> dict[str, Any]:
    from tqdm import tqdm
    from vllm import LLM, SamplingParams

    docs = load_task_docs(task, config.num_samples)
    try:
        llm = LLM(
            model=model_name,
            tokenizer=model_name,
            trust_remote_code=config.trust_remote_code,
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
        )
    except AttributeError as exc:
        if "all_special_tokens_extended" in str(exc):
            raise RuntimeError(
                "Tokenizer initialization failed. This model likely requires "
                "remote-code tokenizer loading. Try `--trust-remote-code` "
                "(enabled by default here) and ensure compatible `vllm`/`transformers` versions."
            ) from exc
        raise

    rows: list[dict[str, Any]] = []
    doc_scores: list[float] = []

    for idx, doc in enumerate(tqdm(docs, desc=f"{model_name} :: {task}")):
        prompt, stop = get_prompt_and_stop(task, doc)
        sampling_params = SamplingParams(
            n=config.n_responses,
            temperature=1.0,
            top_p=1.0,
            max_tokens=512,
            stop=stop,
        )
        output = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)[0]
        responses = [candidate.text.strip() for candidate in output.outputs]
        self_bleu = compute_self_bleu(responses)

        rows.append(
            {
                "task": task,
                "doc_index": idx,
                "prompt": prompt,
                "responses": responses,
                "self_bleu": self_bleu,
            }
        )
        doc_scores.append(self_bleu)

    mean_self_bleu = sum(doc_scores) / len(doc_scores) if doc_scores else 0.0
    return {
        "model": model_name,
        "task": task,
        "num_docs": len(docs),
        "n_responses": config.n_responses,
        "mean_self_bleu": mean_self_bleu,
        "rows": rows,
    }


def run(config: EvalConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate: list[dict[str, Any]] = []

    for model_name in config.models:
        for task in config.tasks:
            result = run_model_on_task(model_name, task, config)
            out_path = config.output_dir / f"{model_name.replace('/', '__')}__{task}.json"
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            aggregate.append(
                {
                    "model": model_name,
                    "task": task,
                    "mean_self_bleu": result["mean_self_bleu"],
                    "num_docs": result["num_docs"],
                    "n_responses": result["n_responses"],
                }
            )

    summary_path = config.output_dir / "summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")


__all__ = ["EvalConfig", "MODELS", "compute_self_bleu", "run"]
