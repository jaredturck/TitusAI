import torch

from generate import (
    apply_top_k,
    apply_top_p,
    extract_response,
    sample_next_token,
)


def test_greedy_sampling():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    token = sample_next_token(logits, 0.0, 0, 1.0)
    assert token.item() == 1


def test_top_k_filter():
    logits = torch.tensor([[1.0, 5.0, 2.0, 4.0]])
    filtered = apply_top_k(logits, 2)
    assert torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[0, 2])
    assert filtered[0, 1].item() == 5.0
    assert filtered[0, 3].item() == 4.0


def test_top_p_keeps_crossing_token():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    filtered = apply_top_p(logits, 0.8)
    assert not torch.isneginf(filtered[0, 0])
    assert not torch.isneginf(filtered[0, 1])


def test_extract_response():
    result = extract_response(
        '<|think|>Work it out.<|/think|><|final|>42<|/final|>'
    )
    assert result['thinking'] == 'Work it out.'
    assert result['final'] == '42'


def test_repetition_penalty():
    from generate import apply_repetition_penalty

    logits = torch.tensor([[2.0, -2.0, 1.0]])
    filtered = apply_repetition_penalty(logits, [0, 1], 2.0)
    assert filtered[0, 0].item() == 1.0
    assert filtered[0, 1].item() == -4.0
    assert filtered[0, 2].item() == 1.0


def test_reasoning_budget_forces_closing_tokens():
    from generate import generate

    class BudgetTokenizer:
        eos_token_id = 4

        def __call__(self, prompt, return_tensors, add_special_tokens):
            return {'input_ids': torch.tensor([[5]])}

        def convert_tokens_to_ids(self, token):
            return {
                '<|think|>': 1,
                '<|/think|>': 2,
                '<|final|>': 3,
            }[token]

        def decode(self, token_ids, skip_special_tokens, clean_up_tokenization_spaces):
            values = {
                1: '<|think|>',
                2: '<|/think|>',
                3: '<|final|>',
                4: '<|endoftext|>',
                6: 'x',
            }
            return ''.join(values[token_id] for token_id in token_ids)

    class BudgetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.max_seq_len = 32
            self.calls = 0

        def forward(self, input_ids, kv_cache=None, start_pos=0, use_cache=False, return_logits=True):
            logits = torch.full((1, input_ids.size(1), 8), -1000.0)
            if self.calls == 0:
                next_id = 1
            elif self.calls < 5:
                next_id = 6
            else:
                next_id = 4
            logits[:, -1, next_id] = 1000.0
            self.calls += 1
            return {
                'logits': logits,
                'kv_cache': [],
            }

    result = generate(
        BudgetModel(),
        BudgetTokenizer(),
        'prompt',
        max_new_tokens=8,
        temperature=0.0,
        stop_token_ids=[4],
        reasoning_token_budget=2,
    )

    assert '<|/think|><|final|>' in result['raw']
    assert result['thinking'] == 'xx'
