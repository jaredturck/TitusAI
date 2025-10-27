import os, re
from collections import Counter

class Tokenizer:

    def __init__(self, max_embeddings=256):
        self.dataset_path = [
            'datasets/book_dataset'
        ]
        self.max_window_size = 16
        self.max_embeddings = max_embeddings
        self.charset = {chr(i) : n for n,i in enumerate(range(32, 126))}

        file_text = self.read_files()
        file_text = SAMPLE_TEXT

        self.get_pieces(file_text, topk=self.max_embeddings - len(self.charset))
    
    def clean_text(self, text):
        ''' Clean text by removing non ASCII 32-126 chars '''
        cur_txt = re.sub(r'([ \-\n\r])\1+', r'\1', text)
        cur_txt = re.sub(r'[^\x20-\x7E]', '', cur_txt)
        return cur_txt

    def read_files(self):
        ''' Read files and return text with non ASCII 32-126 chars removed '''
        text = ''
        for folder in self.dataset_path:
            for filename in os.listdir(folder):
                with open(os.path.join(folder, filename), 'r', encoding='utf-8', errors='ignore') as file:
                    text += self.clean_text(file.read())

        return text
    
    def get_tokp(self, x):
        return x[1] * (len(x[0])-1)

    def get_pieces(self, text, topk = 100):
        ''' Get most common pieces from text '''

        assert topk > 0, 'topk must be positive integer'
        text = list(text)

        for n in range(1):

            i = 0
            counters = {}

            # Loop through each possible candidate of the text
            for _ in range(len(text)):
                if i + 1 >= len(text):
                    break

                slice = text[i] + text[i + 1]
                if not slice.isalpha():
                    i += 1
                    continue
                
                if slice in counters:
                    text[i] = slice
                    del text[i + 1]
                    continue
                else:
                    counters[slice] = 1

            # Add candidates that appear more than once to global candidates
            # for c in consider_candidates:
            #     counters[c] = local_counters[c]
            print(text)
            print([len(text)])
        
        input('STOP')

        sorted_pieces = [i[0] for i in sorted(counters.items(), key=self.get_tokp, reverse=True)[:topk]]
        pieces = '|'.join([re.escape(i) for i in sorted_pieces])
        self.best_pieces_re = re.compile(f'({pieces})|(.)')

        largest = max(self.charset.values())
        for n,piece in enumerate(sorted_pieces):
            self.charset[piece] = largest + n
        
        unk_id = max(self.charset.values()) + 1
        self.charset['<unk>'] = unk_id
        self.charset_unk_id = unk_id
        self.rcharset = {v : k for k,v in self.charset.items()}

    def tokenize(self, text):
        ''' Convert text into list of integers '''

        ctext = self.clean_text(text)
        ids = []
        for m in self.best_pieces_re.finditer(ctext):
            sub_word, char = m.groups()
            if sub_word:
                ids.append(self.charset[sub_word])
            else:
                ids.append(self.charset.get(char, self.charset_unk_id))
        
        return ids
    
    def detokenize(self, ids):
        ''' Convert ids back into text '''

        return ''.join(self.rcharset[num] for num in ids)
            

# path = '/home/jared/Documents/Dropdown Documents/TitusAI/datasets/book_dataset/Fifty Shades 5 Darker (E L James) (Z-Library).txt'

# with open(path, 'r', encoding='utf-8', errors='ignore') as file:
#     text = file.read()

SAMPLE_TEXT = '''
In this tiny, tidy paragraph, the thing about things is their repeating, repeating rhythm: thinking, rethinking, overthinking, 
lightly and likely aligning with friendly, finally, happily tidy endings. The thoughtful algorithm compresses connection, correction, 
collection, and direction, detecting common sub-pieces like th, ing, ly, tion, and ment. It pre-processes, reprocesses, post-processes, 
and decompresses the same simple samples, steadily, silently, reliably. Whether the weather is withering or thriving, 
the method methodically gathers gathered gatherings, compiling and recombining components. Thus the text textually illustrates repetition, 
iteration, and segmentation for efficient, sufficient, and magnificent compression.
'''

t = Tokenizer()
# ids = t.tokenize(text)
# print(t.charset)
# print([f'topk={t.max_embeddings}', len(text), len(ids)])
