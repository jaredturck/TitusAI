import re

import torch


THINK_PATTERN = re.compile(r'<\|think\|>.*?<\|/think\|>', re.DOTALL)
FINAL_PATTERN = re.compile(r'<\|final\|>(.*?)<\|/final\|>', re.DOTALL)


def apply_repetition_penalty(logits, generated_ids, penalty):
    if penalty <= 1.0 or not generated_ids:
        return logits

    logits = logits.clone()
    unique_ids = torch.tensor(
        sorted(set(generated_ids)),
        device=logits.device,
        dtype=torch.long,
    )
    selected = logits.index_select(-1, unique_ids)
    selected = torch.where(
        selected < 0,
        selected * penalty,
        selected / penalty,
    )
    scatter_ids = unique_ids.unsqueeze(0).expand(logits.size(0), -1)
    logits.scatter_(-1, scatter_ids, selected)
    return logits


def banned_ngram_tokens(token_ids, ngram_size):
    if ngram_size is None or ngram_size <= 1 or len(token_ids) < ngram_size - 1:
        return set()

    prefix = tuple(token_ids[-(ngram_size - 1):])
    banned = set()

    for index in range(len(token_ids) - ngram_size + 1):
        candidate_prefix = tuple(token_ids[index:index + ngram_size - 1])
        if candidate_prefix == prefix:
            banned.add(token_ids[index + ngram_size - 1])

    return banned


def apply_no_repeat_ngram(logits, token_ids, ngram_size):
    banned = banned_ngram_tokens(token_ids, ngram_size)
    if not banned:
        return logits

    logits = logits.clone()
    logits[..., list(banned)] = float('-inf')
    return logits


def apply_top_k(logits, top_k):
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits

    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float('-inf'))


def apply_top_p(logits, top_p):
    if top_p <= 0.0 or top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative_probabilities = sorted_probabilities.cumsum(dim=-1)
    remove_mask = cumulative_probabilities > top_p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove_mask, float('-inf'))

    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits.scatter_(-1, sorted_indices, sorted_logits)
    return filtered_logits


def sample_next_token(logits, temperature, top_k, top_p):
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def extract_response(raw_text):
    final_match = FINAL_PATTERN.search(raw_text)
    if final_match:
        final_text = final_match.group(1).strip()
    else:
        final_text = THINK_PATTERN.sub('', raw_text)
        final_text = final_text.replace('<|direct|>', '')
        final_text = final_text.replace('<|reason|>', '')
        final_text = final_text.replace('<|final|>', '')
        final_text = final_text.replace('<|/final|>', '')
        final_text = final_text.replace('<|think|>', '')
        final_text = final_text.replace('<|/think|>', '')
        final_text = final_text.replace('<|im_start|>', '')
        final_text = final_text.replace('<|im_end|>', '')
        final_text = final_text.replace('<|endoftext|>', '')
        final_text = final_text.strip()

    thinking_match = THINK_PATTERN.search(raw_text)
    thinking_text = ''
    if thinking_match:
        thinking_text = thinking_match.group(0)
        thinking_text = thinking_text.replace('<|think|>', '')
        thinking_text = thinking_text.replace('<|/think|>', '')
        thinking_text = thinking_text.strip()

    return {
        'raw': raw_text.strip(),
        'final': final_text,
        'thinking': thinking_text,
    }


def generate(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.08, no_repeat_ngram_size=4, stop_token_ids=None, reasoning_token_budget=None):
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
    input_ids = encoded['input_ids'].to(device)
    max_new_tokens = min(max_new_tokens, model.max_seq_len - 1)
    maximum_prompt_length = max(1, model.max_seq_len - max_new_tokens)

    if input_ids.size(1) > maximum_prompt_length:
        input_ids = input_ids[:, -maximum_prompt_length:]

    if stop_token_ids is None:
        stop_token_ids = [tokenizer.eos_token_id]

    stop_token_ids = {token_id for token_id in stop_token_ids if token_id is not None}
    generated_ids = []
    kv_cache = None
    current_input = input_ids
    start_pos = 0
    forced_token_ids = []
    inside_thinking = False
    thinking_tokens = 0
    forcing_thinking_end = False
    think_open_id = tokenizer.convert_tokens_to_ids('<|think|>')
    think_close_id = tokenizer.convert_tokens_to_ids('<|/think|>')
    final_open_id = tokenizer.convert_tokens_to_ids('<|final|>')

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            output = model(
                current_input,
                kv_cache=kv_cache,
                start_pos=start_pos,
                use_cache=True,
                return_logits=True,
            )
            kv_cache = output['kv_cache']
            logits = output['logits'][:, -1, :]
            logits = apply_repetition_penalty(
                logits,
                generated_ids,
                repetition_penalty,
            )
            logits = apply_no_repeat_ngram(
                logits,
                generated_ids,
                no_repeat_ngram_size,
            )
            if forced_token_ids:
                token_id = forced_token_ids.pop(0)
                next_token = torch.tensor(
                    [[token_id]],
                    device=device,
                    dtype=torch.long,
                )
            else:
                next_token = sample_next_token(
                    logits,
                    temperature,
                    top_k,
                    top_p,
                )
                token_id = int(next_token.item())

            generated_ids.append(token_id)

            if token_id == think_open_id:
                inside_thinking = True
                thinking_tokens = 0
            elif token_id == think_close_id:
                inside_thinking = False
                forcing_thinking_end = False
            elif inside_thinking:
                thinking_tokens += 1

            budget_reached = (
                reasoning_token_budget is not None
                and reasoning_token_budget > 0
                and inside_thinking
                and thinking_tokens >= reasoning_token_budget
                and not forcing_thinking_end
            )
            if budget_reached:
                forced_token_ids.append(think_close_id)
                forced_token_ids.append(final_open_id)
                forcing_thinking_end = True

            if token_id in stop_token_ids:
                break

            start_pos += current_input.size(1)
            current_input = next_token

            if start_pos + 1 >= model.max_seq_len:
                break

    raw_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return extract_response(raw_text)
