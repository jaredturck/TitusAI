# TitusAI design notes

## Decoder block

Every block uses the following residual order:

```text
hidden
  ├─ RMSNorm → grouped-query attention → add
  └─ RMSNorm → SwiGLU                 → add
```

There are 30 blocks. Linear projections have no biases and dropout is zero.

## Tensor shapes

For a batch size `B` and sequence length `T`:

```text
input_ids                  B × T
embedded hidden states     B × T × 576
queries                    B × 9 × T × 64
keys                       B × 3 × T × 64
values                     B × 3 × T × 64
attention output           B × T × 576
SwiGLU gate/value          B × T × 2,000
final hidden states        B × T × 576
vocabulary logits          B × T × vocabulary size
```

Grouped-query attention assigns three query heads to each key/value head. The generation cache stores only the three key/value heads per layer.

## Normalization

`RMSNorm` calculates variance in FP32 and returns the original activation dtype. This keeps BF16 numerical stability without promoting the attention tensors out of the optimized BF16 path.

Queries and keys receive an additional per-head RMSNorm before RoPE.

## RoPE

RoPE is applied to queries and keys after QK normalization. Cached generation supplies `start_pos`, so newly generated tokens receive absolute positions that continue from the prompt.

## Attention implementation

The ordinary training path calls PyTorch scaled-dot-product attention with:

```python
F.scaled_dot_product_attention(
    queries,
    keys,
    values,
    is_causal=True,
    enable_gqa=True,
)
```

This permits PyTorch to choose an optimized exact-attention kernel. Both pretraining and conversational fine-tuning use this path. Segment IDs remain in prepared records for boundary bookkeeping, but conversational training does not build a document-isolation attention mask.

## Tied vocabulary projection

There is no independent language-model head. Logits are produced with:

```python
F.linear(hidden_states, token_embedding.weight)
```

The same matrix therefore performs input embedding and output classification.

During training, vocabulary logits are calculated in chunks. This reduces peak memory without changing the cross-entropy objective.

## Data records

A stored record contains 2,049 token IDs. Runtime inputs and labels are:

```text
inputs = record[0:2048]
labels = record[1:2049]
```

The packer advances by 2,048 tokens, so the final token of one record becomes the first input token of the next record without being trained as a target twice.

Each token also has a local segment ID identifying its source document. A target is replaced by `-100` only when it crosses from one packed document to another, unless an optional loss-mask file is present.

Pretraining and conversational shards do not store loss-mask files. Conversational records contain natural message text separated by newlines and end with the existing document-end token. Every within-conversation token contributes to next-token loss, while later packed conversations may still attend to earlier context through ordinary causal attention.

## Checkpoint consistency

Snapshots and checkpoints are first written to a `.writing` path and then committed with `os.replace()`. The rename is atomic on the same filesystem.

Full checkpoints are taken only after an optimizer update. They record the number of per-rank samples committed in the current deterministic shard order. The current conversation run resumes its own newest full checkpoint. A fresh conversation run otherwise inherits model weights from the newest full checkpoint across earlier runs and starts a new optimizer and scheduler.

## Reasoning controls

The tokenizer adds:

```text
<|direct|>
<|reason|>
<|think|>
<|/think|>
<|final|>
<|/final|>
```

These tokens remain in the vocabulary for checkpoint compatibility. The conversational fine-tune and default inference path use plain newline-separated turns instead of role, mode, or reasoning prefixes.
