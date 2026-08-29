''' Generate responses from the trained language model. '''

from pathlib import Path

import torch
from transformers import AutoTokenizer, LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

from model import LanguageModel

WEIGHTS_PATH = Path('weights')
CONTEXT_LENGTH = 1024
TOKENIZER_NAME = 'gpt2'
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.8
TOP_K = 40
TOP_P = 0.9
DEVICE = 'cuda'

class Inference:
    ''' Generate responses from the trained language model. '''

    def __init__(self):
        ''' Load the tokenizer, checkpoint, and sampling configuration. '''
        checkpoints = sorted(WEIGHTS_PATH.glob('posttrain_*.pt'), key=lambda path: path.stat().st_mtime)

        if not checkpoints:
            checkpoints = sorted(WEIGHTS_PATH.glob('model_*.pt'), key=lambda path: path.stat().st_mtime)

        self.model_path = checkpoints[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.model = LanguageModel().to(DEVICE)
        self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
        self.model.eval()
        self.sampling = LogitsProcessorList([
            TemperatureLogitsWarper(TEMPERATURE),
            TopKLogitsWarper(TOP_K),
            TopPLogitsWarper(TOP_P),
        ])

    def generate(self, prompt):
        ''' Generate one response. '''
        text = f'User:\n{prompt}\nAssistant:\n<think>\n'
        tokens = self.tokenizer(text, return_tensors='pt')['input_ids'].to(DEVICE)
        prompt_length = tokens.shape[1]

        with torch.inference_mode():
            for _ in range(MAX_NEW_TOKENS):
                logits = self.model(tokens[:, -CONTEXT_LENGTH:])
                scores = self.sampling(tokens, logits[:, -1])
                probabilities = torch.softmax(scores, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
                tokens = torch.cat((tokens, next_token), dim=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        response = self.tokenizer.decode(tokens[0, prompt_length:], skip_special_tokens=True)

        if '</think>' in response:
            response = response.split('</think>', 1)[1]

        return response.strip()

    def run(self):
        ''' Run the interactive inference loop. '''
        print(f'Loaded {self.model_path}')

        while True:
            prompt = input('user> ').strip()

            if prompt.lower() in ('exit', 'quit'):
                break

            if prompt:
                print(self.generate(prompt))

if __name__ == '__main__':
    inference = Inference()
    inference.run()
