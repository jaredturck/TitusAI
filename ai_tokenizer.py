import os, re

class Tokenizer:

    def __init__(self):
        self.dataset_path = [
            'datasets/book_dataset'
        ]
        self.max_embeddings = 20_000
        self.charset = {chr(i) : n for n,i in enumerate(range(32, 126))}
        self.get_pieces(self.read_files(), topk=self.max_embeddings - len(self.charset))

    def read_files(self):
        ''' Read files and return text with non ASCII 32-126 chars removed '''
        text = ''
        for folder in self.dataset_path:
            for filename in os.listdir(folder):
                with open(os.path.join(folder, filename), 'r', encoding='utf-8', errors='ignore') as file:
                    text += file.read()

        return re.sub(r'[^\x20-\x7E]', '', text)
    
    def get_tokp(self, x):
        return (x[1] * (len(x[0])-1), len(x[0]), x[0])

    def get_pieces(self, text, topk = 100):
        ''' Get most common pieces from text '''

        assert topk > 0, 'topk must be positive integer'
        counters = {}
        for n in range(2,8):
            for i in range(len(text)):
                c = text[i : i + n]
                counters.setdefault(c, 0)
                counters[c] += 1

        sorted_pieces = [i[0] for i in sorted(counters.items(), key=self.get_tokp, reverse=True)[:topk]]
        pieces = '|'.join([re.escape(i) for i in sorted_pieces])
        self.best_pieces_re = re.compile(f'({pieces})|(.)')

        largest = max(self.charset.values())
        for n,piece in enumerate(sorted_pieces):
            self.charset[piece] = largest + n
        
        unk_id = max(self.charset.values()) + 1
        self.charset['<unk>'] = unk_id
        self.charset_unk_id = unk_id

    def tokenize(self, text):
        ''' Convert text into list of integers '''

        ids = []
        for m in self.best_pieces_re.finditer(text):
            sub_word, char = m.groups()
            if sub_word:
                ids.append(self.charset[sub_word])
            else:
                ids.append(self.charset.get(char, self.charset_unk_id))
        
        return ids

t = Tokenizer()

path = '/home/jared/Documents/Dropdown Documents/TitusAI/datasets/book_dataset/Fifty Shades 5 Darker (E L James) (Z-Library).txt'

with open(path, 'r', encoding='utf-8', errors='ignore') as file:
    text = file.read()

ids = t.tokenize(text)

print([t.charset])
print([len(text), len(ids)])
