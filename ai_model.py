from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
import sentencepiece as spm
import torch.nn as nn
import torch, math, time, sys, os, platform, datetime, requests, array
import numpy as np
from transformers import T5Tokenizer

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
STATUS_WEBHOOK = 'https://discord.com/api/webhooks/1431466888956870677/bg5j5IZiG95bqsgQngre_JZm74MtXtgNCcrA_Q7Xe2mTuJ7lxTHe65jYMyJKPvw_Jq2H'

# Configuration
DEVICE = 'cuda'
TARGET_LOSS = 1.3
EMBEDDING_SIZE = 2000
MAX_LENGTH = 200

if platform.node() == 'Jared-PC':
    BATCH_SIZE = 48
    MAX_SAMPLES = 100_000
    WEIGHTS_PATH = 'weights/'
    TOKENIZER_FILE = 'weights/spu_tokenizer'
    TRAINING_DATA = [
        # 'datasets/wiki',
        'datasets/shakespeare',
        'datasets/book_dataset'
    ]
    USE_ALL_SAMPLES = False
else:
    BATCH_SIZE = 118
    MAX_SAMPLES = 10_000_000
    WEIGHTS_PATH = '/home/jared/TitusAI/weights/'
    TOKENIZER_FILE = '/home/jared/TitusAI/weights/spu_tokenizer'
    TRAINING_DATA = [
        # '/home/jared/TitusAI/datasets/wiki',
        '/home/jared/TitusAI/datasets/shakespeare',
        '/home/jared/TitusAI/datasets/book_dataset'
    ]
    USE_ALL_SAMPLES = True

def send_status(message):
    ''' Sends a status update to the Discord webhook '''
    try:
        requests.post(STATUS_WEBHOOK, json={'content': message})
    except Exception as e:
        print(f'[error] Failed to send status update: {e}')

class ShakespeareDataset(Dataset):
    def __init__(self):
        self.max_length = MAX_LENGTH
        self.embedding_size = EMBEDDING_SIZE
        self.load_tokenizer()
        self.dataset_len = None
        self.buffer_size = 1024 * 1024 * 16  # 16 MB buffer

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return self.ids[idx : idx + self.max_length + 1]
    
    def read_data(self):
        ''' Reads training data from TXT file '''

        print('[+] Reading training data...')
        eod_id = self.tokenizer.convert_tokens_to_ids('<eod>')
        assert isinstance(eod_id, int) and eod_id != self.tokenizer.unk_token_id, '[error] <eod> token not found in tokenizer vocabulary'

        ids = array.array('I')
        eod_u32 = np.uint32(eod_id)

        start = time.time()
        for folder in TRAINING_DATA:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    
                    doc_content = True
                    while doc_content:
                        doc_content = file.read(self.buffer_size)
                        doc_ids = self.tokenizer(doc_content, return_tensors='pt', add_special_tokens=False)\
                            .input_ids.squeeze(0).cpu().numpy().astype(np.uint32, copy=False)
                        ids.frombytes(doc_ids.tobytes())

                        if time.time() - start > 10:
                            start = time.time()
                            print(f'[+] Processed {len(ids):,} tokens')

                ids.append(eod_u32)

        self.ids = torch.frombuffer(memoryview(ids), dtype=torch.int32).clone().to(torch.long)
        self.dataset_len = len(self.ids) - (self.max_length + 1)
        print(f'[+] Loaded {len(self.ids):,} training samples')

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
        
        files = []
        for folder in TRAINING_DATA:
            for file in os.listdir(folder):
                files.append(os.path.join(folder, file))

        spm.SentencePieceTrainer.Train(
            input=','.join(files),
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
        self.d_model = 1024
        self.nhead = self.d_model // 64
        self.dim_feedforward = self.d_model * 4
        self.no_transformer_layers = self.d_model // 128
        self.dropout = 0.1
        self.embedding_size = EMBEDDING_SIZE
        self.max_length = MAX_LENGTH
        self.max_epochs = 10000
        self.context = torch.empty(1, 0, dtype=torch.long, device=DEVICE)
        self.sqrt_dmodel = math.sqrt(self.d_model)
        self.dataloader_workers = max(2, os.cpu_count() // 2)
        self.optimizer = None

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
        # Delete oldest weight file
        files = [os.path.join(WEIGHTS_PATH, file) for file in os.listdir(WEIGHTS_PATH) if file.endswith('.pth')]
        if len(files) > 10:
            oldest_file = min(files, key=os.path.getctime)
            os.remove(oldest_file)
            print(f'[+] Deleted oldest weights file {oldest_file}')

        weights_file = os.path.join(WEIGHTS_PATH, f'model_{datetime.datetime.now().strftime(r"%d_%b_%Y-%H_%M")}.pth')
        torch.save({
            'weights' : self.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
        }, weights_file)
        print('[+] Model weights saved')
    
    def load_weights(self):
        files = [os.path.join(WEIGHTS_PATH, file) for file in os.listdir(WEIGHTS_PATH) if file.endswith('.pth')]
        if not files:
            return print('[-] No weights found, starting training from scratch')

        weights_file = max(files, key=os.path.getctime)
        if os.path.isfile(weights_file):
            weights_data = torch.load(weights_file)
            if isinstance(weights_data, dict) and 'weights' in weights_data and 'optimizer' in weights_data:
                self.load_state_dict(weights_data['weights'])
                if self.optimizer and weights_data['optimizer']:
                    self.optimizer.load_state_dict(weights_data['optimizer'])
                    print('[+] Optimizer state loaded')
                print(f'[+] Model weights loaded {weights_file}')

            else:
                self.load_state_dict(weights_data)
                print(f'[+] Model weights loaded {weights_file} (optimizer state not found)')
    
    def train_model(self):
        ''' Main training loop '''

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, fused=True)
        self.load_weights()

        self.dataset.read_data()
        self.dataloader = DataLoader(
            self.dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=self.dataloader_workers,
            persistent_workers=True, prefetch_factor=4
        )

        loss_func = nn.CrossEntropyLoss()
        prev_batch_num = 0

        send_status(f'[+] Starting training, d_model={self.d_model}, nhead={self.nhead}, dim_feedforward={self.dim_feedforward}, '
            f'layers={self.no_transformer_layers}, batch_size={BATCH_SIZE}')
        print(f'[+] Starting training, d_model={self.d_model}, nhead={self.nhead}, dim_feedforward={self.dim_feedforward}, '
            f'layers={self.no_transformer_layers}, batch_size={BATCH_SIZE}')
        for epoch in range(self.max_epochs):
            total_loss = 0.0
            epoch_start = time.time()
            save_start = time.time()
            start = time.time()
            for n, batch in enumerate(self.dataloader):

                batch = batch.to(DEVICE, non_blocking=True)
                src = batch[:, :-1]
                trg = batch[:, 1:]

                self.optimizer.zero_grad()
                with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                    output = self.forward(src)
                    loss = loss_func(output.reshape(-1, output.size(-1)), trg.reshape(-1))

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if time.time() - start > 10:
                    pcnt = (n+1) / len(self.dataloader) * 100
                    tps = int((((n+1) - prev_batch_num) * BATCH_SIZE * self.max_length) / (time.time() - start))
                    start = time.time()
                    print(f'[+] Epoch {epoch+1} of {self.max_epochs}, loss: {loss.item():.4f}, batch {n+1} of {len(self.dataloader):,}, tps: {tps:,} ({pcnt:.1f}%)')
                
                    if time.time() - save_start > 600:
                        save_start = time.time()
                        self.save_weights()
                        send_status(f'[+] Epoch {epoch+1} of {self.max_epochs}, loss: {loss.item():.4f}, batch {n+1} of {len(self.dataloader):,}, '
                            f'tps: {tps:,} ({pcnt:.1f}%)')
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
        eod_id = self.dataset.tokenizer.convert_tokens_to_ids('<eod>')
        input_len = seq.size(1)

        for _ in range(self.max_length):
            x = seq[:, -self.max_length:]
            logits = self.forward(x)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_token], dim=-1)

            if next_token.item() == self.dataset.tokenizer.eos_token_id or next_token.item() == eod_id:
                break
        
        output_txt = self.dataset.tokenizer.decode(seq[0, input_len:].tolist(), skip_special_tokens=True)
        print(output_txt)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        try:
            model = TitusModel().to(DEVICE, torch.bfloat16)
            model.train_model()
            
        except KeyboardInterrupt:
            model.save_weights()
    else:
        model = TitusModel().to(DEVICE, torch.bfloat16)
        model.load_weights()
        model.eval()
        print(f'[+] d_model={model.d_model}, nhead={model.nhead}, dim_feedforward={model.dim_feedforward}, layers={model.no_transformer_layers}')
        while True:
            text = input('> ')
            model.predict(text)
