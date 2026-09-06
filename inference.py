''' Generate raw continuations from the pretrained language model. '''

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model import LanguageModel

WEIGHTS_PATH = Path('weights')
CONTEXT_LENGTH = 256
TOKENIZER_NAME = 'gpt2'
MAX_NEW_TOKENS = 256
DEVICE = 'cuda'

class Inference:
    ''' Generate raw continuations from the latest pretraining checkpoint. '''

    def __init__(self):
        ''' Load the tokenizer and latest pretraining checkpoint. '''
        checkpoints = sorted(WEIGHTS_PATH.glob('model_*.pt'), key=lambda path: path.stat().st_mtime)
        self.model_path = checkpoints[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.model = LanguageModel().to(DEVICE)
        self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
        self.model.eval()

    def generate(self, prompt):
        ''' Generate the model's greedy continuation without altering its logits. '''
        tokens = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        generated = []

        with torch.inference_mode():
            for _ in range(MAX_NEW_TOKENS):
                logits = self.model(tokens[:, -CONTEXT_LENGTH:])
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
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
    Inference().run()
