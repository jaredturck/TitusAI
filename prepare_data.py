''' Download and tokenize the training datasets. '''

import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

WEIGHTS_PATH = Path('weights')
PRETRAIN_DATA_PATH = WEIGHTS_PATH / 'data.bin'
POSTTRAIN_DATA_PATH = WEIGHTS_PATH / 'posttrain.bin'
POSTTRAIN_MASK_PATH = WEIGHTS_PATH / 'posttrain_mask.bin'
PRETRAIN_DATASET_NAME = 'Salesforce/wikitext'
PRETRAIN_DATASET_CONFIG = 'wikitext-103-raw-v1'
POSTTRAIN_DATASET_NAME = 'open-thoughts/OpenThoughts-114k'
POSTTRAIN_DATASET_CONFIG = 'metadata'
TOKENIZER_NAME = 'gpt2'
READ_BATCH_SIZE = 10000
TOKENIZER_BATCH_SIZE = 512
POSTTRAIN_CONTEXT_LENGTH = 1024

class DataPreparer:
    ''' Prepare pretraining or reasoning post-training data. '''

    def __init__(self, stage):
        ''' Initialize the requested data preparation stage. '''
        assert stage in ('pretrain', 'posttrain')
        self.stage = stage
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        WEIGHTS_PATH.mkdir(exist_ok=True)

    def is_article_title(self, text):
        ''' Check whether a WikiText row starts a new article. '''
        text = text.strip()
        return text.startswith('= ') and text.endswith(' =') and not text.startswith('= =')

    def write_documents(self, file, documents):
        ''' Tokenize documents and append them to the packed token stream. '''
        encoded = self.tokenizer(documents, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False, verbose=False)['input_ids']
        token_count = sum(len(token_ids) + 1 for token_ids in encoded)
        tokens = np.empty(token_count, dtype=np.uint16)
        offset = 0

        for token_ids in encoded:
            length = len(token_ids)
            tokens[offset:offset + length] = token_ids
            tokens[offset + length] = self.tokenizer.eos_token_id
            offset += length + 1

        tokens.tofile(file)
        return token_count

    def prepare_pretrain(self):
        ''' Prepare the packed WikiText pretraining stream. '''
        dataset = load_dataset(PRETRAIN_DATASET_NAME, PRETRAIN_DATASET_CONFIG, split='train')
        documents = []
        document = []
        token_count = 0
        article_count = 0

        with PRETRAIN_DATA_PATH.open('wb') as file, tqdm(total=len(dataset), desc='Preparing', unit='lines') as progress:
            for batch in dataset.iter(batch_size=READ_BATCH_SIZE):
                for text in batch['text']:
                    if self.is_article_title(text):
                        if document:
                            documents.append('\n'.join(document) + '\n')
                            article_count += 1
                        document = [text]
                    elif document:
                        document.append(text)

                    if len(documents) >= TOKENIZER_BATCH_SIZE:
                        token_count += self.write_documents(file, documents)
                        documents.clear()

                progress.update(len(batch['text']))

            if document:
                documents.append('\n'.join(document) + '\n')
                article_count += 1

            if documents:
                token_count += self.write_documents(file, documents)

        print(f'Saved {token_count:,} tokens from {article_count:,} articles to {PRETRAIN_DATA_PATH}')

    def write_reasoning_batch(self, data_file, mask_file, batch):
        ''' Write fixed reasoning samples and assistant loss masks. '''
        prompts = [f'User:\n{problem}\nAssistant:\n<think>\n' for problem in batch['problem']]
        responses = [f'{reasoning}\n</think>\n{solution}' for reasoning, solution in zip(batch['deepseek_reasoning'], batch['deepseek_solution'])]
        prompt_ids = self.tokenizer(prompts, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False, verbose=False)['input_ids']
        response_ids = self.tokenizer(responses, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False, verbose=False)['input_ids']
        sequence_length = POSTTRAIN_CONTEXT_LENGTH + 1
        tokens = np.full((len(prompts), sequence_length), self.tokenizer.eos_token_id, dtype=np.uint16)
        masks = np.zeros((len(prompts), sequence_length), dtype=np.uint8)
        sample_count = 0

        for prompt, response in zip(prompt_ids, response_ids):
            response = response + [self.tokenizer.eos_token_id]
            length = len(prompt) + len(response)

            if length > sequence_length:
                continue

            tokens[sample_count, :len(prompt)] = prompt
            tokens[sample_count, len(prompt):length] = response
            masks[sample_count, len(prompt):length] = 1
            sample_count += 1

        tokens[:sample_count].tofile(data_file)
        masks[:sample_count].tofile(mask_file)
        return sample_count

    def prepare_posttrain(self):
        ''' Prepare masked OpenThoughts reasoning samples. '''
        dataset = load_dataset(POSTTRAIN_DATASET_NAME, POSTTRAIN_DATASET_CONFIG, split='train')
        sample_count = 0

        with POSTTRAIN_DATA_PATH.open('wb') as data_file, POSTTRAIN_MASK_PATH.open('wb') as mask_file, tqdm(total=len(dataset), desc='Preparing', unit='samples') as progress:
            for batch in dataset.iter(batch_size=TOKENIZER_BATCH_SIZE):
                sample_count += self.write_reasoning_batch(data_file, mask_file, batch)
                progress.update(len(batch['problem']))

        print(f'Saved {sample_count:,} reasoning samples to {POSTTRAIN_DATA_PATH}')

    def prepare(self):
        ''' Prepare the selected training dataset. '''
        if self.stage == 'pretrain':
            self.prepare_pretrain()
        else:
            self.prepare_posttrain()

if __name__ == '__main__':
    preparer = DataPreparer(sys.argv[1])
    preparer.prepare()
