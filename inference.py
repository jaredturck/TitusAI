import torch
from transformers import AutoTokenizer

from model import LanguageModel, context_length


device = 'cuda'
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = LanguageModel().to(device)
model.load_state_dict(torch.load('weights/model.pt', map_location=device))
model.eval()

prompt = 'The meaning of life is'
tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)

with torch.no_grad():
    for _ in range(100):
        logits = model(tokens[:, -context_length:])
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=1)

print(tokenizer.decode(tokens[0]))
