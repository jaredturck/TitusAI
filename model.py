''' Define the language model architecture. '''

import torch
from torch import nn

VOCAB_SIZE = 50257
CONTEXT_LENGTH = 256
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 4
FFN_DIM = 1024

class LanguageModel(nn.Module):
    ''' Predict the next token with a small Transformer language model. '''

    def __init__(self):
        ''' Build the model layers. '''
        super().__init__()

        self.token_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH, D_MODEL)

        layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=NUM_HEADS, dim_feedforward=FFN_DIM, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS)
        self.output = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, input_ids):
        ''' Return vocabulary logits for each input token. '''
        sequence_length = input_ids.shape[1]
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)

        mask = torch.triu(torch.ones(sequence_length, sequence_length, device=input_ids.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer(hidden, mask=mask)
        return self.output(hidden)
