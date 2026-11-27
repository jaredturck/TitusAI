import torch
from torch import nn

class TitusAI(nn.Module):
    '''Predict the next token with a causal Transformer.'''

    def __init__(self):
        '''Initialize the model layers and causal mask.'''
        super().__init__()

        vocab_size = 8000
        self.token_embedding = nn.Embedding(vocab_size, 384)
        self.position_embedding = nn.Embedding(512, 384)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=384, nhead=6, dim_feedforward=3264, dropout=0.0, activation="gelu", batch_first=True),
            num_layers=6,
        )

        self.output = nn.Linear(384, vocab_size)
        self.register_buffer("causal_mask", nn.Transformer.generate_square_subsequent_mask(512), persistent=False)
        self.register_buffer("position_ids", torch.arange(512), persistent=False)

    def forward(self, input_ids):
        '''Return vocabulary logits at each position in the input batch.'''
        sequence_length = input_ids.size(1)
        positions = self.position_ids[:sequence_length]
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.transformer(
            hidden,
            mask=self.causal_mask[:sequence_length, :sequence_length],
            is_causal=True,
        )
        return self.output(hidden)
