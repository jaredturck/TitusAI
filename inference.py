''' Generate continuations from pretrained or post-trained checkpoints. '''

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model import LanguageModel

WEIGHTS_PATH = Path('weights')
TOKENIZER_NAME = 'gpt2'
PRETRAIN_CONTEXT_LENGTH = 256
POSTTRAIN_CONTEXT_LENGTH = 1024
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.8
TOP_K = 40
REPETITION_PENALTY = 1.15
DEVICE = 'cuda'

class Inference:
    ''' Generate continuations from the selected training stage. '''

    def __init__(self, stage):
        ''' Load the tokenizer and latest checkpoint for the selected stage. '''
        self.stage = stage
        self.context_length = PRETRAIN_CONTEXT_LENGTH if stage == 'pretrain' else POSTTRAIN_CONTEXT_LENGTH
        pattern = 'model_*.pt' if stage == 'pretrain' else 'posttrain_*.pt'
        checkpoints = sorted(WEIGHTS_PATH.glob(pattern), key=lambda path: path.stat().st_mtime)
        self.model_path = checkpoints[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.model = LanguageModel().to(DEVICE)
        self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
        self.model.eval()

    def sample_posttrain(self, logits, generated):
        ''' Sample a post-trained token while discouraging repetition. '''
        logits = logits.clone()

        if generated:
            previous = torch.tensor(list(set(generated)), device=logits.device)
            scores = logits[:, previous]
            logits[:, previous] = torch.where(scores < 0, scores * REPETITION_PENALTY, scores / REPETITION_PENALTY)

        logits = logits / TEMPERATURE
        values, indexes = torch.topk(logits, TOP_K, dim=-1)
        probabilities = torch.softmax(values, dim=-1)
        sample = torch.multinomial(probabilities, 1)
        return indexes.gather(-1, sample)

    def generate(self, prompt):
        ''' Generate one continuation. '''
        if self.stage == 'posttrain':
            prompt = f'User:\n{prompt}\nAssistant:\n'

        tokens = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        generated = []

        with torch.inference_mode():
            for _ in range(MAX_NEW_TOKENS):
                logits = self.model(tokens[:, -self.context_length:])[:, -1]

                if self.stage == 'posttrain':
                    next_token = self.sample_posttrain(logits, generated)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                token_id = next_token.item()
                generated.append(token_id)
                tokens = torch.cat((tokens, next_token), dim=1)

                if token_id == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def run(self):
        ''' Run the interactive inference loop. '''
        print(f'Loaded {self.model_path}', file=sys.stderr)

        while True:
            print('prompt> ', end='', file=sys.stderr, flush=True)
            prompt = input()
            sys.stdout.write(self.generate(prompt))
            sys.stdout.flush()
            print(file=sys.stderr)

if __name__ == '__main__':
    print('1. pretrain')
    print('2. posttrain')
    stage = {'1': 'pretrain', '2': 'posttrain'}[input('> ').strip()]
    Inference(stage).run()
