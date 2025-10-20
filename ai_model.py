from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
import sentencepiece as spm
import torch.nn as nn
import torch, math, time, sys, os, platform
from transformers import T5Tokenizer

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configuration
DEVICE = 'cuda'
TARGET_LOSS = 1.3
EMBEDDING_SIZE = 2000

if platform.node() == 'Jared-PC':
    BATCH_SIZE = 30
    MAX_SAMPLES = 100_000
    WEIGHTS_FILE = 'weights/shakespeare_model.pth'
    TOKENIZER_FILE = 'weights/spu_tokenizer'
    TRAINING_DATA = ['datasets/training_data.txt', 'datasets/romantic_novels.txt']
    USE_ALL_SAMPLES = False
else:
    BATCH_SIZE = 285
    MAX_SAMPLES = 10_000_000
    WEIGHTS_FILE = '/home/jared/TitusAI/weights/shakespeare_model.pth'
    TOKENIZER_FILE = '/home/jared/TitusAI/weights/spu_tokenizer'
    TRAINING_DATA = ['/home/jared/TitusAI/datasets/training_data.txt', '/home/jared/TitusAI/datasets/romantic_novels.txt']
    USE_ALL_SAMPLES = True

class ShakespeareDataset(Dataset):
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.embedding_size = EMBEDDING_SIZE
        self.load_tokenizer()

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        return self.training_data[idx]
    
    def read_data(self):
        ''' Reads training data from TXT file '''
        self.training_data = []

        raw_text = ''
        for file_path in TRAINING_DATA:
            with open(file_path, 'r', encoding='utf-8') as file:
                raw_text += file.read()
        
        ids = self.tokenizer(raw_text, return_tensors='pt').input_ids.squeeze(0)
        max_start = ids.size(0) - (self.max_length + 1)

        for start in range(0, max_start + 1):
            x = ids[start : start + self.max_length]
            y = ids[start + 1 : start + self.max_length + 1]
            self.training_data.append((x, y))

            if not USE_ALL_SAMPLES and start >= MAX_SAMPLES:
                break

        print(f'[+] Loaded {len(self.training_data)} training samples')
    
    def load_tokenizer(self):
        ''' Loads an existing tokenizer from file '''

        if not os.path.isfile(os.path.join(TOKENIZER_FILE, 'spu_tokenizer.model')):
            input('[error] Failed to load existing tokenizer, please train a new one. Press Enter to continue...')
            self.train_tokenizer()

        # Load tokenizer
        self.tokenizer = T5Tokenizer(
            vocab_file=os.path.join(TOKENIZER_FILE, 'spu_tokenizer.model'),
            bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', extra_ids=0
        )
        print('[+] Loaded existing tokenizer')

    def train_tokenizer(self):
        ''' Trains a SentencePiece tokenizer on the training data '''

        if not os.path.isdir(TOKENIZER_FILE):
            os.makedirs(TOKENIZER_FILE)

        spm.SentencePieceTrainer.Train(
            input=','.join(TRAINING_DATA),
            model_prefix=os.path.join(TOKENIZER_FILE, 'spu_tokenizer'),
            model_type='unigram',
            vocab_size=EMBEDDING_SIZE - 256,
            byte_fallback=True,
            user_defined_symbols=['<eod>'],
            pad_id=0, pad_piece='<pad>',
            unk_id=1, unk_piece='<unk>',
            bos_id=2, bos_piece='<s>',
            eos_id=3,  eos_piece='</s>',
            normalization_rule_name='nfkc'
        )

class TitusModel(Module):
    def __init__(self):
        super().__init__()
        Module.train(self, True)
        self.dataset = ShakespeareDataset()
        self.d_model = 512
        self.nhead = self.d_model // 64
        self.dim_feedforward = self.d_model * 4
        self.no_transformer_layers = 6
        self.dropout = 0.1
        self.embedding_size = EMBEDDING_SIZE
        self.max_length = 512
        self.max_epochs = 10000
        self.context = torch.empty(1, 0, dtype=torch.long, device=DEVICE)
        self.sqrt_dmodel = math.sqrt(self.d_model)
        self.dataloader_workers = max(2, os.cpu_count() // 2)

        self.register_buffer('pos_arange', torch.arange(self.max_length, device=DEVICE))
        self.register_buffer('full_causal_mask', torch.triu(torch.ones(self.max_length, self.max_length, dtype=torch.bool, device=DEVICE), diagonal=1))

        self.embeddings = nn.Embedding(num_embeddings=self.embedding_size, embedding_dim=self.d_model)
        self.pos_emb = nn.Embedding(self.max_length, self.d_model)
        self.em_dropout = nn.Dropout(self.dropout)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, batch_first=True),
            num_layers=self.no_transformer_layers
        )

        self.out_proj = nn.Linear(self.d_model, self.embedding_size)
        self.out_proj.weight = self.embeddings.weight
    
    def forward(self, src):

        src_B, src_T = src.size()
        pos = self.pos_arange[:src_T].unsqueeze(0).expand(src_B, src_T)

        logits = self.out_proj(
            self.encoder(
                self.em_dropout(
                    self.embeddings(src) * self.sqrt_dmodel + self.pos_emb(pos)
                ),
                mask = self.full_causal_mask[:src_T, :src_T]
            )
        )

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
        self.dataloader = DataLoader(
            self.dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=self.dataloader_workers,
            persistent_workers=True, prefetch_factor=4
        )

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, fused=True)
        loss_func = nn.CrossEntropyLoss()
        prev_batch_num = 0

        print(f'[+] Starting training, d_model={self.d_model}, nhead={self.nhead}, dim_feedforward={self.dim_feedforward}, batch_size={BATCH_SIZE}')
        for epoch in range(self.max_epochs):
            total_loss = 0.0
            epoch_start = time.time()
            save_start = time.time()
            start = time.time()
            for n, (src, trg) in enumerate(self.dataloader):
                src = src.to(DEVICE, non_blocking=True)
                trg = trg.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                    output = self.forward(src)
                    loss = loss_func(output.reshape(-1, output.size(-1)), trg.reshape(-1))

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if time.time() - start > 10:
                    pcnt = (n+1) / len(self.dataloader) * 100
                    tps = int((((n+1) - prev_batch_num) * BATCH_SIZE * self.max_length) / (time.time() - start))
                    start = time.time()
                    print(f'[+] Epoch {epoch+1} of {self.max_epochs}, loss: {loss.item():.4f}, batch {n+1} of {len(self.dataloader)}, tps: {tps:,} ({pcnt:.1f}%)')
                
                    if time.time() - save_start > 600:
                        save_start = time.time()
                        self.save_weights()
                        print(f'[+] Saved weights at epoch {epoch+1}, batch {n+1}')
                    
                    prev_batch_num = n+1

            avg_loss = total_loss / len(self.dataloader)
            print(f'[+] Epoch {epoch+1} of {self.max_epochs}, avg loss: {avg_loss:.4f}, time: {time.time()-epoch_start:.2f}s')

            if avg_loss < TARGET_LOSS:
                print('[+] Target loss reached, stopping training')
                self.save_weights()
                return
    
    @torch.no_grad()
    def predict(self, text):

        seq = self.dataset.tokenizer(text, return_tensors='pt')['input_ids'].to(DEVICE)

        for _ in range(self.max_length):
            x = seq[:, -self.max_length:]
            logits = self.forward(x)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_token], dim=-1)
        
        output_txt = self.dataset.tokenizer.decode(seq[0].tolist(), skip_special_tokens=True)
        print(output_txt)

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
