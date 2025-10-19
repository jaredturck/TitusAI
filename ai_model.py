from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import csv, torch, math, time, sys, os
from transformers import AutoTokenizer

DEVICE = 'cuda'
BATCH_SIZE = 10
WEIGHTS_FILE = 'weights/shakespeare_model.pth'
MAX_SAMPLES = 1000
TARGET_LOSS = 0.01
USE_ALL_SAMPLES = True

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

                if self.tokenizer.eos_token_id is not None:
                    ids = target_ids['input_ids'].squeeze(0)
                    if ids.size(0) < self.max_length:
                        ids = torch.cat([ids, torch.tensor([self.tokenizer.eos_token_id])], dim=0)
                    else:
                        ids[-1] = self.tokenizer.eos_token_id
                    target_tensor = ids
                else:
                    target_tensor = target_ids['input_ids'].squeeze(0)

                self.training_data.append((src_ids['input_ids'].squeeze(0), target_tensor))

                if not USE_ALL_SAMPLES and len(self.training_data) >= MAX_SAMPLES:
                    break
        
        print(f'[+] Loaded {len(self.training_data)} training samples')

class TitusModel(Module):
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
        self.context = torch.empty(1, 0, dtype=torch.long, device=DEVICE)
        self.sqrt_dmodel = math.sqrt(self.d_model)

        self.register_buffer('pos_arange', torch.arange(self.max_length, device=DEVICE))
        self.register_buffer('full_causal_mask', torch.triu(torch.ones(self.max_length, self.max_length, dtype=torch.bool, device=DEVICE), diagonal=1))

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

        src_B, src_T = src.size()
        trg_B, trg_T = trg.size()

        src_pos = self.pos_arange[:src_T].unsqueeze(0).expand(src_B, src_T)
        trg_pos = self.pos_arange[:trg_T].unsqueeze(0).expand(trg_B, trg_T)
        mask = self.full_causal_mask[:trg_T, :trg_T]

        src_emb = self.em_dropout(
            self.embeddings(src) * self.sqrt_dmodel + self.pos_emb(src_pos)
        )

        trg_emb = self.em_dropout(
            self.embeddings(trg) * self.sqrt_dmodel + self.pos_emb(trg_pos)
        )

        memory = self.encoder(src_emb, src_key_padding_mask=(src == 0))
        output = self.decoder(trg_emb, memory, tgt_mask=mask, tgt_key_padding_mask=(trg == 0), memory_key_padding_mask=(src == 0))
        logits = self.out_proj(output)

        return logits
    
    def save_weights(self):
        torch.save(self.state_dict(), WEIGHTS_FILE)
        print('[+] Model weights saved')
    
    def load_weights(self):
        if os.path.isfile(WEIGHTS_FILE):
            self.load_state_dict(torch.load(WEIGHTS_FILE))
            print('[+] Model weights loaded')

    def train(self):
        self.dataset.read_data()
        self.dataloader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=self.dataset.collate_fn)

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        loss_func = nn.CrossEntropyLoss(ignore_index=0)

        print('[+] Starting training')
        for epoch in range(self.max_epochs):
            total_loss = 0.0
            start = time.time()
            for src, trg in self.dataloader:
                src = src.to(DEVICE)
                trg = trg.to(DEVICE)

                optimizer.zero_grad()
                output = self.forward(src, trg[:, :-1])
                loss = loss_func(output.reshape(-1, output.size(-1)), trg[:, 1:].reshape(-1))

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            print(f'[+] Epoch {epoch+1} of {self.max_epochs}, avg loss: {avg_loss:.4f}, time: {time.time()-start:.2f}s')

            if avg_loss < TARGET_LOSS:
                print('[+] Target loss reached, stopping training')
                self.save_weights()
                return
    
    @torch.no_grad()
    def predict(self, text):

        eos_id = self.dataset.tokenizer.eos_token_id
        src = self.dataset.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_length)['input_ids'].to(DEVICE)

        # Update Context
        if self.context.size(1) > 0 and eos_id is not None:
            self.context = torch.cat([self.context, torch.tensor([[eos_id]], device=DEVICE)], dim=1)

        self.context = torch.cat([self.context, src], dim=1)
        if self.context.size(1) > self.max_length:
            self.context = self.context[:, -self.max_length:]

        # Pass through encoder
        src_emb = self.em_dropout(
            self.embeddings(self.context) * math.sqrt(self.d_model) + self.pos_emb(
                torch.arange(self.context.size(1), device=DEVICE).unsqueeze(0).expand(*self.context.size())
            )
        )

        memory = self.encoder(src_emb, src_key_padding_mask=None)
        target = torch.tensor([[self.dataset.tokenizer.bos_token_id]], device=DEVICE)

        # Pass through decoder
        for i in range(self.max_length):
            trg_pos = torch.arange(target.size(1), device=DEVICE).unsqueeze(0).expand(*target.size())
            trg_emb = self.em_dropout(
                self.embeddings(target) * math.sqrt(self.d_model) + self.pos_emb(trg_pos)
            )
            trg_mask = torch.triu(torch.ones(trg_emb.size(1), trg_emb.size(1), device=DEVICE, dtype=torch.bool), diagonal=1)
            output = self.decoder(trg_emb, memory, tgt_mask=trg_mask, tgt_key_padding_mask=(target == 0), memory_key_padding_mask=None)
            logits = self.out_proj(output)

            next_token = logits[:, -1, :].argmax(-1).unsqueeze(1)
            target = torch.cat((target, next_token), dim=1)

            if eos_id is not None and next_token.item() == eos_id:
                break
        
        ids = target.squeeze(0).tolist()
        text_output = self.dataset.tokenizer.decode(ids, skip_special_tokens=True)
        print(text_output)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        try:
            model = TitusModel().to(DEVICE)
            model.load_weights()
            model.train()
            
        except KeyboardInterrupt:
            model.save_weights()
    else:
        model = TitusModel().to(DEVICE)
        model.load_weights()
        while True:
            text = input('> ')
            model.predict(text)
