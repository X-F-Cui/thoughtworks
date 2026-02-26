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
