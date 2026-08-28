''' Generate text from the trained language model. '''

import torch
from transformers import AutoTokenizer

from model import CONTEXT_LENGTH, LanguageModel

DEVICE = 'cuda'
MODEL_PATH = 'weights/model.pt'
TOKENIZER_NAME = 'gpt2'
PROMPT = 'The meaning of life is'
MAX_NEW_TOKENS = 100

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = LanguageModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

tokens = tokenizer(PROMPT, return_tensors='pt')['input_ids'].to(DEVICE)

with torch.no_grad():
    for _ in range(MAX_NEW_TOKENS):
        logits = model(tokens[:, -CONTEXT_LENGTH:])
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=1)

print(tokenizer.decode(tokens[0]))
