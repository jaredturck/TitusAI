import torch

from checkpoint import find_latest_snapshot
from config import (
    GENERATION_CONFIG,
    INFERENCE_CONFIG,
    SNAPSHOT_PATH,
)
from generate import generate
from model import TitusModel
from tokenizer import format_conversation_prompt, load_tokenizer


def token_id(tokenizer, token):
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) == 1:
        return token_ids[0]
    return None


def load_latest_model():
    snapshot_path = find_latest_snapshot(SNAPSHOT_PATH)
    assert snapshot_path is not None, 'No inference snapshots were found'

    snapshot_data = torch.load(
        snapshot_path,
        map_location='cpu',
        weights_only=False,
    )
    model = TitusModel(snapshot_data['model_config'])
    model.load_state_dict(snapshot_data['model'])
    model.eval()
    model.float()

    return model, snapshot_data, snapshot_path


def print_snapshot_info(snapshot, snapshot_path):
    print(f'[+] Loaded: {snapshot_path.relative_to(SNAPSHOT_PATH)}')
    print(f'[+] Step: {snapshot.get("global_step", 0):,}')
    print(f'[+] Tokens seen: {snapshot.get("tokens_seen", 0):,}')
    print(f'[+] Validation loss: {snapshot.get("validation_loss")}')
    print(f'[+] Saved: {snapshot.get("saved_at")}')


def print_help():
    print('/reload      Load the newest snapshot')
    print('/info        Show snapshot information')
    print('/clear       Clear conversation history')
    print('/help        Show commands')
    print('/exit        Exit')


def main():
    torch.set_num_threads(INFERENCE_CONFIG['threads'])
    torch.set_num_interop_threads(1)

    tokenizer = load_tokenizer(local_only=True)
    model, snapshot, snapshot_path = load_latest_model()
    messages = []
    newline_token_id = token_id(tokenizer, '\n')
    stop_token_ids = [
        tokenizer.eos_token_id,
        newline_token_id,
    ]

    print_snapshot_info(snapshot, snapshot_path)
    print_help()

    while True:
        user_text = input('[chat] You: ').strip()
        if not user_text:
            continue

        if user_text == '/exit':
            break

        if user_text == '/help':
            print_help()
            continue

        if user_text == '/info':
            print_snapshot_info(snapshot, snapshot_path)
            continue

        if user_text == '/clear':
            messages = []
            print('[+] Conversation cleared')
            continue

        if user_text == '/reload':
            newest_path = find_latest_snapshot(SNAPSHOT_PATH)
            if newest_path == snapshot_path:
                print('[+] Already using the newest snapshot')
                continue
            model, snapshot, snapshot_path = load_latest_model()
            print_snapshot_info(snapshot, snapshot_path)
            continue

        messages.append(user_text)
        prompt = format_conversation_prompt(messages)
        result = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=GENERATION_CONFIG['max_new_tokens'],
            temperature=GENERATION_CONFIG['temperature'],
            top_k=GENERATION_CONFIG['top_k'],
            top_p=GENERATION_CONFIG['top_p'],
            repetition_penalty=GENERATION_CONFIG['repetition_penalty'],
            no_repeat_ngram_size=GENERATION_CONFIG['no_repeat_ngram_size'],
            stop_token_ids=stop_token_ids,
        )

        response = result['final'].split('\n', 1)[0].strip()
        print(f'Titus: {response}')
        messages.append(response)


if __name__ == '__main__':
    main()
