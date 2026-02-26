# 01_vllm_self_bleu

Runs vLLM generation for:
- `allenai/Olmo-3-1025-7B`
- `allenai/Olmo-3-7B-Instruct-SFT`
- `allenai/Olmo-3-7B-Instruct-DPO`
- `allenai/Olmo-3-7B-Instruct`

on:
- `gsm8k` (generate-until style)
- `commonsense_qa` converted to generate-until with CoT prompting

## What it does

For each model/task pair:
1. Load `num_samples` examples.
2. Generate **5** responses per question with vLLM sampling defaults (`temperature=1.0`, `top_p=1.0`) and task stop sequences.
3. Compute self-BLEU over the 5 responses for each question.
4. Average per-question self-BLEU into a final model/task score.

## Run

```bash
uv run python experiments/01_vllm_self_bleu/run.py
```

## Output

- `results/<model>__<task>.json`: per-question generations + self-BLEU.
- `results/summary.json`: average self-BLEU per model/task.
