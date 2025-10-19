from torch.utils.data import Dataset
from transformers import AutoTokenizer
import csv

class ShakespeareDataset(Dataset):
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        return self.training_data[idx]
    
    def collate_fn(self, batch):
        return batch
    
    def read_data(self):
        ''' Reads training data from CSV file '''
        self.training_data = []
        with open('datasets/training_data.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                src_ids = self.tokenizer(row[0], truncation=True, max_length=self.max_length, return_tensors='pt')
                target_ids = self.tokenizer(row[1], truncation=True, max_length=self.max_length, return_tensors='pt')
                self.training_data.append((src_ids['input_ids'].squeeze(0), target_ids['input_ids'].squeeze(0)))

if __name__ == "__main__":
    dataset = ShakespeareDataset()
    dataset.read_data()
