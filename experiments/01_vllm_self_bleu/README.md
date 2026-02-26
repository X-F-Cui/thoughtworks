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

Default run:

```bash
uv run python experiments/01_vllm_self_bleu/run.py
```

Run only **5** examples per task:

```bash
uv run python experiments/01_vllm_self_bleu/run.py --num-samples 5
```

Run a single model + single task:

```bash
uv run python experiments/01_vllm_self_bleu/run.py \
  --models allenai/Olmo-3-7B-Instruct \
  --tasks commonsense_qa \
  --num-samples 5
```

## CLI arguments

- `--output-dir` output directory for JSON files.
- `--models` one or more HF model IDs.
- `--tasks` one or both of: `gsm8k`, `commonsense_qa`.
- `--num-samples` number of examples loaded per task split.
- `--n-responses` generations sampled per question (default: 5).
- `--max-model-len`, `--tensor-parallel-size`, `--gpu-memory-utilization` vLLM runtime settings.
- `--trust-remote-code/--no-trust-remote-code` controls HF remote tokenizer/model code (default: enabled).

## Output

- `results/<model>__<task>.json`: per-question generations + self-BLEU.
- `results/summary.json`: average self-BLEU per model/task.

## Troubleshooting

If you see an error like `GPT2Tokenizer has no attribute all_special_tokens_extended`, it usually means tokenizer loading fell back to an incompatible tokenizer class. For OLMo checkpoints, keep `--trust-remote-code` enabled (default), and use compatible `vllm` + `transformers` versions.
