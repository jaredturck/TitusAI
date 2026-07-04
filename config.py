from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / 'data'
PROCESSED_DATA_PATH = DATA_PATH / 'processed'
TRAIN_DATA_PATH = PROCESSED_DATA_PATH / 'train'
VALIDATION_DATA_PATH = PROCESSED_DATA_PATH / 'validation'
WEIGHTS_PATH = PROJECT_ROOT / 'weights'
SNAPSHOT_PATH = WEIGHTS_PATH / 'snapshots'
CHECKPOINT_PATH = WEIGHTS_PATH / 'checkpoints'
TOKENIZER_PATH = PROJECT_ROOT / 'tokenizer_files'

TOKENIZER_NAME = 'HuggingFaceTB/SmolLM2-135M-Instruct'
DOCUMENT_END_TOKEN = '<|endoftext|>'
SPECIAL_TOKENS = [
    '<|direct|>',
    '<|reason|>',
    '<|think|>',
    '<|/think|>',
    '<|final|>',
    '<|/final|>',
]

MODEL_CONFIG = {
    'vocab_size': None,
    'd_model': 576,
    'num_layers': 30,
    'num_heads': 9,
    'num_kv_heads': 3,
    'head_dim': 64,
    'ffn_dim': 2000,
    'max_seq_len': 2048,
    'rms_norm_eps': 1e-5,
    'rope_theta': 10_000.0,
    'dropout': 0.0,
    'loss_chunk_size': 1024,
}

TRAIN_CONFIG = {
    'run_name': 'pretrain',
    'train_data_path': TRAIN_DATA_PATH,
    'validation_data_path': VALIDATION_DATA_PATH,
    'initial_weights': None,
    'seed': 1337,
    'micro_batch_size': 4,
    'gradient_accumulation_steps': 8,
    'learning_rate': 3e-4,
    'min_learning_rate': 3e-5,
    'warmup_ratio': 0.02,
    'weight_decay': 0.1,
    'gradient_clip': 1.0,
    'max_train_tokens': 13_000_000_000,
    'validation_interval_steps': 1000,
    'validation_batches': 200,
    'checkpoint_interval_steps': 5000,
    'checkpoint_keep': 3,
    'snapshot_interval_seconds': 600,
    'snapshot_keep': 10,
    'log_interval_seconds': 10,
    'num_workers': 6,
    'prefetch_factor': 4,
    'pin_memory': True,
    'persistent_workers': True,
    'gradient_checkpointing': False,
    'isolate_packed_documents': False,
    'resume_training': True,
}

PREPARE_CONFIG = {
    'sequence_length': MODEL_CONFIG['max_seq_len'],
    'sequences_per_shard': 8192,
    'validation_fraction': 0.001,
    'minimum_document_characters': 200,
    'maximum_document_characters': 2_000_000,
    'exact_deduplication': True,
    'paragraph_deduplication': True,
    'paragraph_minimum_characters': 200,
    'deduplication_database': DATA_PATH / 'deduplication.sqlite3',
    'shuffle_buffer_size': 10_000,
    'random_seed': 1337,
    'max_total_tokens': 13_000_000_000,
}

DATA_SOURCES = [
    {
        'name': 'nemotron_math',
        'dataset': 'nvidia/Nemotron-CC-Math-v1',
        'config': '4plus',
        'data_dir': None,
        'split': 'train',
        'text_fields': ['text'],
        'id_fields': [],
        'columns': ['text'],
        'target_tokens': 780_000_000,
        'paragraph_deduplication': True,
    },
    {
        'name': 'swallowcode',
        'dataset': 'tokyotech-llm/swallow-code-v2',
        'config': None,
        'data_dir': None,
        'split': 'train',
        'loader': 'jsonl_text',
        'data_glob': 'stage5-auto-format/python/medium/train*.jsonl',
        'jsonl_text_field': 'text',
        'text_fields': ['text'],
        'id_fields': [],
        'columns': ['text'],
        'target_tokens': 1_560_000_000,
        'paragraph_deduplication': False,
    },
    {
        'name': 'cosmopedia',
        'dataset': 'HuggingFaceTB/smollm-corpus',
        'config': 'cosmopedia-v2',
        'data_dir': None,
        'split': 'train',
        'text_fields': ['text'],
        'id_fields': [],
        'columns': ['text'],
        'target_tokens': 260_000_000,
        'paragraph_deduplication': True,
    },
    {
        'name': 'dclm',
        'dataset': 'HuggingFaceFW/dclm_100BT-shuffled',
        'config': None,
        'data_dir': None,
        'split': 'train',
        'text_fields': ['text'],
        'id_fields': [],
        'columns': ['text'],
        'target_tokens': 10_400_000_000,
        'paragraph_deduplication': True,
        'pre_shuffled': True,
    },
]

INSTRUCTION_CONFIG = {
    'dataset': 'HuggingFaceTB/smol-smoltalk',
    'config': None,
    'split': 'train',
    'messages_field': 'messages',
    'sequence_length': MODEL_CONFIG['max_seq_len'],
    'sequences_per_shard': 1024,
    'validation_fraction': 0.002,
    'maximum_conversation_tokens': MODEL_CONFIG['max_seq_len'] - 1,
    'output_path': PROCESSED_DATA_PATH / 'instructions',
}


DISCORD_CONFIG = {
    'enabled': True,
    'webhook_url': None,
    'webhook_path': PROJECT_ROOT / 'discord_webhook.txt',
    'username': 'TitusAI Training',
    'status_interval_seconds': 600,
    'request_timeout_seconds': 5,
}

GENERATION_CONFIG = {
    'max_new_tokens': 256,
    'temperature': 0.7,
    'top_k': 50,
    'top_p': 0.9,
    'repetition_penalty': 1.08,
    'no_repeat_ngram_size': 4,
    'reasoning_token_budget': 192,
}

INFERENCE_CONFIG = {
    'snapshot_run': 'pretrain',
    'device': 'cpu',
    'threads': 24,
    'system_prompt': 'You are Titus, a helpful and concise assistant.',
}
