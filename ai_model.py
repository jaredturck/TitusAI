from torch.utils.data import Dataset
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import csv
from transformers import AutoTokenizer

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
        self.d_model = 512
        self.nhead = self.d_model // 4
        self.dim_feedforward = self.d_model * 4
        self.dropout = 0.1

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, batch_first=True),
            num_layers=6
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, batch_first=True),
            num_layers=6
        )

    def train(self):
        self.dataset.read_data()



if __name__ == "__main__":
    dataset = ShakespearModel()
    dataset.train()
