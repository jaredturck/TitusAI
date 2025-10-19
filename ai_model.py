from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import csv, torch, math
from transformers import AutoTokenizer

DEVICE = 'cuda'
BATCH_SIZE = 12

class ShakespeareDataset(Dataset):
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        return self.training_data[idx]
    
    def collate_fn(self, batch):
        src, trg = zip(*batch)
        src_pad = pad_sequence(src, batch_first=True, padding_value=0)
        trg_pad = pad_sequence(trg, batch_first=True, padding_value=0)
        return src_pad, trg_pad
    
    def read_data(self):
        ''' Reads training data from CSV file '''
        self.training_data = []
        with open('datasets/training_data.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                src_ids = self.tokenizer(row[0], truncation=True, max_length=self.max_length, return_tensors='pt')
                target_ids = self.tokenizer(row[1], truncation=True, max_length=self.max_length, return_tensors='pt')
                self.training_data.append((src_ids['input_ids'].squeeze(0), target_ids['input_ids'].squeeze(0)))

class ShakespearModel(Module):
    def __init__(self):
        super().__init__()
        self.dataset = ShakespeareDataset()
        self.d_model = 128
        self.nhead = self.d_model // 64
        self.dim_feedforward = self.d_model * 4
        self.dropout = 0.1
        self.embedding_size = 50257
        self.max_length = 512
        self.max_epochs = 10000

        self.embeddings = nn.Embedding(num_embeddings=self.embedding_size, embedding_dim=self.d_model)
        self.pos_emb = nn.Embedding(self.max_length, self.d_model)
        self.em_dropout = nn.Dropout(self.dropout)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, batch_first=True),
            num_layers=6
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, batch_first=True),
            num_layers=6
        )

        self.out_proj = nn.Linear(self.d_model, self.embedding_size)
    
    def forward(self, src, trg):

        sqrt_dmodel = math.sqrt(self.d_model)
        
        src_emb = self.em_dropout(
            self.embeddings(src) * sqrt_dmodel + self.pos_emb(
                torch.arange(src.size(1), device=DEVICE).unsqueeze(0).expand(*src.size())
            )
        )

        trg_emb = self.em_dropout(
            self.embeddings(trg) * sqrt_dmodel + self.pos_emb(
                torch.arange(trg.size(1), device=DEVICE).unsqueeze(0).expand(*trg.size())
            )
        )

        mask = torch.triu(torch.ones(trg_emb.size(1), trg_emb.size(1), device=DEVICE, dtype=torch.bool), diagonal=1)

        memory = self.encoder(src_emb, src_key_padding_mask=(src == 0))
        output = self.decoder(trg_emb, memory, tgt_mask=mask, tgt_key_padding_mask=(trg == 0), memory_key_padding_mask=(src == 0))
        logits = self.out_proj(output)

        return logits

    def train(self):
        self.dataset.read_data()
        self.dataloader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=self.dataset.collate_fn)

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        loss_func = nn.CrossEntropyLoss(ignore_index=0)

        print('[+] Starting training')
        for epoch in range(self.max_epochs):
            total_loss = 0.0
            for src, trg in self.dataloader:
                src = src.to(DEVICE)
                trg = trg.to(DEVICE)

                optimizer.zero_grad()
                output = self.forward(src, trg[:, :-1])
                loss = loss_func(output.reshape(-1, output.size(-1)), trg[:, 1:].reshape(-1))

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f'[+] Epoch {epoch+1} of {self.max_epochs}, avg loss: {total_loss/len(self.dataloader):.4f}')

if __name__ == "__main__":
    dataset = ShakespearModel().to(DEVICE)
    dataset.train()
