import torch

from model import TitusModel


def test_forward_and_backward(tiny_config):
    model = TitusModel(tiny_config)
    input_ids = torch.randint(0, tiny_config['vocab_size'], (2, 16))
    labels = torch.randint(0, tiny_config['vocab_size'], (2, 16))
    labels[:, :3] = -100

    output = model(
        input_ids,
        labels=labels,
        return_logits=True,
    )

    assert output['logits'].shape == (2, 16, tiny_config['vocab_size'])
    assert output['loss_tokens'].item() == 26
    output['loss'].backward()
    assert model.token_embedding.weight.grad is not None


def test_tied_output_projection(tiny_config):
    model = TitusModel(tiny_config)
    input_ids = torch.randint(0, tiny_config['vocab_size'], (1, 8))
    output = model(input_ids, return_logits=True)
    hidden_output = model(input_ids, return_logits=False)

    assert output['logits'] is not None
    assert hidden_output['logits'] is None
    assert not hasattr(model, 'lm_head')


def test_causal_attention(tiny_config):
    torch.manual_seed(10)
    model = TitusModel(tiny_config).eval()
    first = torch.randint(0, tiny_config['vocab_size'], (1, 12))
    second = first.clone()
    second[:, 7:] = torch.randint(0, tiny_config['vocab_size'], (1, 5))

    first_logits = model(first, return_logits=True)['logits']
    second_logits = model(second, return_logits=True)['logits']

    assert torch.allclose(first_logits[:, :7], second_logits[:, :7], atol=1e-6)


def test_kv_cache_matches_full_forward(tiny_config):
    torch.manual_seed(11)
    model = TitusModel(tiny_config).eval()
    input_ids = torch.randint(0, tiny_config['vocab_size'], (1, 14))
    full_logits = model(input_ids, return_logits=True)['logits']

    cache = None
    cached_logits = []
    for position in range(input_ids.size(1)):
        output = model(
            input_ids[:, position:position + 1],
            kv_cache=cache,
            start_pos=position,
            use_cache=True,
            return_logits=True,
        )
        cache = output['kv_cache']
        cached_logits.append(output['logits'])

    cached_logits = torch.cat(cached_logits, dim=1)
    assert torch.allclose(full_logits, cached_logits, atol=1e-5)


def test_segment_mask_isolates_documents(tiny_config):
    torch.manual_seed(12)
    model = TitusModel(tiny_config).eval()
    first = torch.randint(0, tiny_config['vocab_size'], (1, 12))
    second = first.clone()
    second[:, :6] = torch.randint(0, tiny_config['vocab_size'], (1, 6))
    segment_ids = torch.tensor([[0] * 6 + [1] * 6])

    first_logits = model(
        first,
        segment_ids=segment_ids,
        return_logits=True,
    )['logits']
    second_logits = model(
        second,
        segment_ids=segment_ids,
        return_logits=True,
    )['logits']

    assert torch.allclose(first_logits[:, 6:], second_logits[:, 6:], atol=1e-6)


def test_gradient_checkpointing(tiny_config):
    model = TitusModel(tiny_config)
    model.enable_gradient_checkpointing(True)
    input_ids = torch.randint(0, tiny_config['vocab_size'], (2, 12))
    labels = torch.randint(0, tiny_config['vocab_size'], (2, 12))
    output = model(input_ids, labels=labels, return_logits=False)
    output['loss'].backward()
    assert model.token_embedding.weight.grad is not None
