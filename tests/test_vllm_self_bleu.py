from datasets import Dataset

from allegro import vllm_self_bleu
from allegro.vllm_self_bleu import compute_self_bleu, get_prompt_and_stop


def test_compute_self_bleu_non_negative():
    responses = [
        "The answer is 12.",
        "I think the answer is 12.",
        "Let's solve it: 12.",
        "Final answer: 12",
        "12",
    ]
    score = compute_self_bleu(responses)
    assert 0.0 <= score <= 1.0


def test_commonsense_qa_prompt_includes_cot_and_choices():
    doc = {
        "question": "Where would you find a pillow?",
        "choices": {"label": ["A", "B"], "text": ["bed", "freezer"]},
    }
    prompt, stop = get_prompt_and_stop("commonsense_qa", doc)
    assert "Think step by step" in prompt
    assert "A. bed" in prompt
    assert stop == ["\nQuestion:"]


def test_load_task_docs_is_seeded_and_reproducible(monkeypatch):
    docs = [{"id": i} for i in range(10)]
    ds = Dataset.from_list(docs)

    def fake_load_dataset(*args, **kwargs):
        return ds

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    sample_a = vllm_self_bleu.load_task_docs("gsm8k", num_samples=5, random_seed=42)
    sample_b = vllm_self_bleu.load_task_docs("gsm8k", num_samples=5, random_seed=42)
    sample_c = vllm_self_bleu.load_task_docs("gsm8k", num_samples=5, random_seed=7)

    ids_a = [row["id"] for row in sample_a]
    ids_b = [row["id"] for row in sample_b]
    ids_c = [row["id"] for row in sample_c]

    assert ids_a == ids_b
    assert ids_a != ids_c


def test_load_task_docs_defaults_to_all_examples(monkeypatch):
    docs = [{"id": i} for i in range(7)]
    ds = Dataset.from_list(docs)

    def fake_load_dataset(*args, **kwargs):
        return ds

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    sample = vllm_self_bleu.load_task_docs("commonsense_qa", num_samples=None, random_seed=42)

    assert len(sample) == len(docs)
