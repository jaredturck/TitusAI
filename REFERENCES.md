# Research and implementation references

The project deliberately uses a conservative modern dense decoder rather than combining every recent architecture proposal.

## Architecture

- Mehta et al., **MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases**  
  https://arxiv.org/abs/2402.14905
- Shazeer, **GLU Variants Improve Transformer**  
  https://arxiv.org/abs/2002.05202
- Xiong et al., **On Layer Normalization in the Transformer Architecture**  
  https://arxiv.org/abs/2002.04745
- Zhang and Sennrich, **Root Mean Square Layer Normalization**  
  https://arxiv.org/abs/1910.07467
- Su et al., **RoFormer: Enhanced Transformer with Rotary Position Embedding**  
  https://arxiv.org/abs/2104.09864
- Ainslie et al., **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**  
  https://arxiv.org/abs/2305.13245
- Press and Wolf, **Using the Output Embedding to Improve Language Models**  
  https://arxiv.org/abs/1608.05859
- Allal et al., **SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model**  
  https://arxiv.org/abs/2502.02737
- Yang et al., **Qwen3 Technical Report**  
  https://arxiv.org/abs/2505.09388

## Data

- Hugging Face DCLM 100BT (globally shuffled)  
  https://huggingface.co/datasets/HuggingFaceFW/dclm_100BT-shuffled
- NVIDIA Nemotron-CC-Math-v1  
  https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1
- SwallowCode-v2  
  https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2
- SmolLM-Corpus / Cosmopedia v2  
  https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
- SODA  
  https://huggingface.co/datasets/allenai/soda
- DailyDialog  
  https://huggingface.co/datasets/OpenRL/daily_dialog
- SmolLM2-135M-Instruct tokenizer  
  https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct

## PyTorch

- Scaled dot-product attention  
  https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- DistributedDataParallel  
  https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
- Automatic mixed precision  
  https://pytorch.org/docs/stable/amp.html
