from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
import torch.nn as nn
import torch, math, time, sys, os, platform, datetime, requests, array, re
from transformers import AutoTokenizer
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
STATUS_WEBHOOK = os.getenv('STATUS_WEBHOOK')

# Configuration
TARGET_LOSS = 1.3
MAX_LENGTH = 200

if platform.node() == 'Jared-PC':
    DEVICE = 'cuda'
    BATCH_SIZE = 28
    MAX_SAMPLES = 100_000
    WEIGHTS_PATH = 'weights/'
    TOKENIZER_FILE = 'weights/spu_tokenizer'
    TRAINING_DATA = [
        # 'datasets/book_dataset',
        # 'datasets/falcon-distillation/outputs_dataset_1/',
        # 'datasets/falcon-distillation/outputs_dataset_2/',
        # 'datasets/falcon-distillation/outputs_dataset_3/',
        # 'datasets/falcon-distillation/outputs_dataset_4/',
        # 'datasets/falcon-distillation/outputs_dataset_5/',
        'datasets/chatgpt-questions/falcon_outputs'
    ]
    USE_ALL_SAMPLES = False

elif platform.node() == 'Jared-server':
    DEVICE = 'cuda:0'
    BATCH_SIZE = 140
    MAX_SAMPLES = 10_000_000
    WEIGHTS_PATH = '/home/jared/TitusAI/weights/'
    TOKENIZER_FILE = '/home/jared/TitusAI/weights/spu_tokenizer'
    TRAINING_DATA = [
        # '/home/jared/TitusAI/datasets/book_dataset',
        # '/home/jared/TitusAI/datasets/falcon-distillation/outputs_dataset_1/',
        # '/home/jared/TitusAI/datasets/falcon-distillation/outputs_dataset_2/',
        # '/home/jared/TitusAI/datasets/falcon-distillation/outputs_dataset_3/',
        # '/home/jared/TitusAI/datasets/falcon-distillation/outputs_dataset_4/',
        # '/home/jared/TitusAI/datasets/falcon-distillation/outputs_dataset_5/',
        '/home/jared/TitusAI/datasets/chatgpt-questions/falcon_outputs'
    ]
    USE_ALL_SAMPLES = True

else:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 1
    MAX_SAMPLES = 1

def send_status(message):
    ''' Sends a status update to the Discord webhook '''
    try:
        requests.post(STATUS_WEBHOOK, json={'content': message})
        print(message)
    except Exception as e:
        print(f'[error] Failed to send status update: {e}')

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'[+][{func.__name__}] took {end - start:.2f} seconds')
        return result
    return wrapper

class ShakespeareDataset(Dataset):
    def __init__(self):
        self.max_length = MAX_LENGTH
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.dataset_len = None
        self.buffer_size = 1024 * 1024 * 16  # 16 MB buffer

        self.tokenizer.add_special_tokens({
            'eos_token' : '<EOS>',
            'bos_token' : '<BOS>',
        })

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return self.ids[idx : idx + self.max_length + 1]
    
    def read_data(self):
        ''' Reads training data from TXT file '''

        print('[+] Reading training data...')
        contains_letters = re.compile('[a-zA-Z]+')
        ids = array.array('I')
        start = time.time()
        for folder in TRAINING_DATA:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as file:

                    for row in file:
                        if not contains_letters.search(row):
                            continue

                        if row == '[BOS]\n':
                            ids.extend([self.tokenizer.bos_token_id])

                        elif row == '[EOS]\n':
                            ids.extend([self.tokenizer.eos_token_id])

                        else:
                            token_ids = self.tokenizer.encode(row, add_special_tokens=False)
                            ids.extend(token_ids)
                            
                        if time.time() - start > 10:
                            start = time.time()
                            print(f'[+] Processed {len(ids):,} tokens')

        self.ids = torch.frombuffer(memoryview(ids), dtype=torch.int32).clone().to(torch.long)
        self.dataset_len = len(self.ids) - (self.max_length + 1)
        print(f'[+] Loaded {len(self.ids):,} training samples')

    def save_tensors(self):
        filename = os.path.join(WEIGHTS_PATH, f'tensors_{datetime.datetime.now().strftime(r"%d_%b_%Y-%H_%M")}.data')
        torch.save(self.ids, filename)

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
        self.embedding_size = len(self.dataset.tokenizer)
        self.max_length = MAX_LENGTH
        self.max_epochs = 5
        self.context = torch.empty(1, 0, dtype=torch.long, device=DEVICE)
        self.sqrt_dmodel = math.sqrt(self.d_model)
        self.dataloader_workers = max(2, os.cpu_count() // 2)
        self.optimizer = None
        self.weights_file = None
        self.training_started = False

        self.register_buffer('pos_arange', torch.arange(self.max_length, device=DEVICE))
        self.register_buffer('full_causal_mask', torch.triu(torch.ones(self.max_length, self.max_length, dtype=torch.bool, device=DEVICE), diagonal=1))

        self.embeddings = nn.Embedding(num_embeddings=self.embedding_size, embedding_dim=self.d_model)
        self.pos_emb = nn.Embedding(self.max_length, self.d_model)
        self.em_dropout = nn.Dropout(self.dropout)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, batch_first=True),
            num_layers=self.no_transformer_layers
        )

        self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=self.d_model,
            n_classes=self.embedding_size,
            cutoffs=[2000, 10_000],
            div_value=4.0
        )

    def forward(self, src):

        src_B, src_T = src.size()
        pos = self.pos_arange[:src_T].unsqueeze(0).expand(src_B, src_T)

        logits = self.encoder(
            self.em_dropout(
                self.embeddings(src) * self.sqrt_dmodel + self.pos_emb(pos)
            ),
            mask = self.full_causal_mask[:src_T, :src_T]
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
    
    def load_weights(self, weight_path=None):
        weight_path = weight_path if weight_path else WEIGHTS_PATH
        files = [os.path.join(weight_path, file) for file in os.listdir(weight_path) if file.endswith('.pth')]
        if not files:
            return print('[-] No weights found, starting training from scratch')

        weights_file = max(files, key=os.path.getctime)
        if os.path.isfile(weights_file):
            weights_data = torch.load(weights_file, map_location=DEVICE)
            if isinstance(weights_data, dict) and 'weights' in weights_data and 'optimizer' in weights_data:
                self.load_state_dict(weights_data['weights'])
                if self.optimizer and weights_data['optimizer']:
                    self.optimizer.load_state_dict(weights_data['optimizer'])
                    print('[+] Optimizer state loaded')
                self.weights_file = os.path.basename(weights_file)
                print(f'[+] Model weights loaded {weights_file}')

            else:
                self.load_state_dict(weights_data)
                self.weights_file = os.path.basename(weights_file)
                print(f'[+] Model weights loaded {weights_file} (optimizer state not found)')

    def train_model(self):
        ''' Main training loop '''

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, fused=True)
        self.load_weights()

        self.dataset.read_data()
        self.dataset.save_tensors()
        self.dataloader = DataLoader(
            self.dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=self.dataloader_workers,
            persistent_workers=True, prefetch_factor=4
        )
        prev_batch_num = 0

        send_status(f'[+] Starting training ({torch.cuda.get_device_name(DEVICE)}), d_model={self.d_model}, nhead={self.nhead}, '
            f'dim_feedforward={self.dim_feedforward}, layers={self.no_transformer_layers}, batch_size={BATCH_SIZE}, embedding_size={self.embedding_size}')
        self.training_started = True
        for epoch in range(self.max_epochs):
            total_loss = 0.0
            epoch_start = time.time()
            save_start = time.time()
            start = time.time()
            for n, batch in enumerate(self.dataloader):
                try:

                    batch = batch.to(DEVICE, non_blocking=True)
                    src = batch[:, :-1]
                    trg = batch[:, 1:]

                    self.optimizer.zero_grad()
                    
                    out = self.forward(src)
                    out2d = out.reshape(-1, out.size(-1))
                    tgt1d = trg.reshape(-1)

                    out = self.adaptive_softmax(out2d, tgt1d)
                    loss = out.loss

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

                except torch.cuda.OutOfMemoryError:
                    print('[error] CUDA out of memory, skipping batch')
                    send_status('[error] CUDA out of memory, skipping batch')
                    torch.cuda.empty_cache()
                    time.sleep(5)
                    continue

            avg_loss = total_loss / len(self.dataloader)
            print(f'[+] Epoch {epoch+1} of {self.max_epochs}, avg loss: {avg_loss:.4f}, time: {time.time()-epoch_start:.2f}s')

            if avg_loss < TARGET_LOSS:
                print('[+] Target loss reached, stopping training')
                self.save_weights()
                return
    
    def repetition_penalty(self, probs, generated, penalty):
        if penalty <= 1.0:
            return probs
        
        probs = probs.clone()

        ids_tensor = torch.tensor(generated, device=probs.device)
        unique_ids, counts = torch.unique(ids_tensor, return_counts=True)

        unique_ids = unique_ids[unique_ids != self.dataset.tokenizer.eos_token_id]
        counts = counts[unique_ids != self.dataset.tokenizer.eos_token_id] if unique_ids.numel() > 0 else counts

        if unique_ids.numel() > 0:
            probs[0, unique_ids] -= counts.float() * math.log(penalty)
        
        return probs
    
    def repeat_ngram_penalty(self, log_probs, seq, cur_len, no_repeat_ngram_size):
        n = no_repeat_ngram_size
        if n is None or n <= 1 or cur_len < n:
            return log_probs
        
        tokens = seq[:cur_len]
        context = tokens[-(n-1):]
        banned = set()

        for idx in range(0, cur_len - n + 1):
            window = tokens[idx:idx + n - 1]
            if torch.equal(window, context):
                next_token_id = int(tokens[idx + n - 1].item())
                banned.add(next_token_id)
        
        if not banned:
            return log_probs
        
        lp = log_probs.clone()
        lp[0, list(banned)] = float('-inf')
        return lp
    
    def sample_next_token(self, probs, temperature, top_k, top_p):

        if temperature <= 0.0:
            return probs.argmax(dim=-1, keepdim=True)
        
        log_probs = probs / temperature
        if top_k > 0:
            values, _ = torch.topk(log_probs, top_k, dim=-1)
            log_probs = torch.where(log_probs < values[..., -1, None], torch.full_like(log_probs, float('-inf')), log_probs)
        
        probs = torch.softmax(log_probs, dim=-1)
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)

            mask = cumulative > top_p
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, idx)
            return next_token
        
        else:
            return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate_k(self, text, length_multiplier, temperature, top_k, top_p, repetition_penalty, no_repeat_ngram_size, k):

        prompt = f'Q: {text}\nA: '
        base_seq = self.dataset.tokenizer(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        prompt_len = base_seq.size(1)
        max_steps = int(self.max_length * length_multiplier)
        total_max_length = prompt_len + max_steps
        eos_id = self.dataset.tokenizer.eos_token_id

        seq = torch.full((k, total_max_length), fill_value=eos_id, dtype=torch.long, device=DEVICE)
        seq[:, :prompt_len] = base_seq.expand(k, -1)

        lengths = [prompt_len for i in range(k)]
        outputs = ['' for i in range(k)]
        generated_ids = [[] for i in range(k)]
        finished = [False for i in range(k)]

        x = torch.empty((k, self.max_length), dtype=torch.long, device=DEVICE)
        last_indices = torch.empty(k, dtype=torch.long, device=DEVICE)

        for step in range(max_steps):
            if all(finished):
                break

            for i in range(k):
                L_i = lengths[i]
                ctx_len = min(L_i, self.max_length)
                x[i, :ctx_len] = seq[i, L_i - ctx_len:L_i]
                if ctx_len < self.max_length:
                    x[i, ctx_len:] = eos_id
                
                last_indices[i] = ctx_len - 1
            
            logits = self.forward(x)
            last_hidden = logits[torch.arange(k, device=DEVICE), last_indices, :]
            log_probs = self.adaptive_softmax.log_prob(last_hidden)

            for i in range(k):
                if finished[i]:
                    continue

                lp_i = self.repetition_penalty(log_probs[i:i+1, :], generated_ids[i], repetition_penalty)
                lp_i = self.repeat_ngram_penalty(lp_i, seq[i], lengths[i], no_repeat_ngram_size)
                next_token = self.sample_next_token(lp_i, temperature=temperature, top_k=top_k, top_p=top_p)
                item = next_token.item()

                if item == eos_id:
                    if re.search(r'[a-zA-Z0-9]', outputs[i]):
                        finished[i] = True
                    continue

                token_text = self.dataset.tokenizer.decode([item], skip_special_tokens=True)

                if token_text.strip() == '' and not re.search(r'[a-zA-Z0-9]', outputs[i]):
                    generated_ids[i].append(item)
                    continue

                if lengths[i] >= total_max_length:
                    finished[i] = True
                    continue

                seq[i, lengths[i]] = item
                lengths[i] += 1

                outputs[i] += token_text
                generated_ids[i].append(item)
        
        return [out.strip() for out in outputs]

    def cosine_similarity(self, a, b):
        ''' Compute cosine similarity between two Counters '''
        intersection = a.keys() & b.keys()
        product = sum(a[x] * b[x] for x in intersection)

        sum1 = math.sqrt(sum(v * v for v in a.values()))
        sum2 = math.sqrt(sum(v * v for v in b.values()))
        return product / (sum1 * sum2) if sum1 and sum2 else 0.0
    
    def predict(self, text, length_multiplier=1.0, temperature=0.4, top_k=30, top_p=0.8, repetition_penalty=1.1, no_repeat_ngram_size=4):
        ''' Generate single answer '''

        answer = self.generate_k(text, length_multiplier, temperature, top_k, top_p, repetition_penalty, no_repeat_ngram_size, k=1)
        return answer[0]
    
    @timeit
    def think_longer(self, text, k = 3):
        ''' Pick the best answer '''

        answers = self.generate_k(text, length_multiplier=1.0, temperature=0.6, top_k=50, top_p=0.9, repetition_penalty=1.2, no_repeat_ngram_size=4, k=k)

        counters = []
        for answer in answers:
            ids = self.dataset.tokenizer.encode(answer, add_special_tokens=False)
            counters.append(Counter(ids))
        
        avg_sim = []
        for i in range(len(answers)):
            sims = []
            for j in range(len(answers)):
                if i != j:
                    sims.append(self.cosine_similarity(counters[i], counters[j]))
            avg_sim.append(sum(sims) / len(sims))
        
        best_idx = max(range(len(avg_sim)), key=lambda i: avg_sim[i])
        best_answer = answers[best_idx]

        return best_answer

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        try:
            model = TitusModel().to(DEVICE)
            model.train_model()
            
        except KeyboardInterrupt:
            if model.training_started:
                model.save_weights()
            else:
                print('[-] Training was not started, no weights to save')
    else:
        model = TitusModel().to(DEVICE)
        model.load_weights()
        model.eval()
        print(f'[+] d_model={model.d_model}, nhead={model.nhead}, dim_feedforward={model.dim_feedforward}, layers={model.no_transformer_layers}')
        while True:
            text = input('> ')
            print(model.predict(text))
            # print(model.think_longer(text, k=3))
