TitusAI is starting over.

## Training

Run from the project root.

```bash
source .venv/bin/activate
torchrun --standalone --nproc-per-node=2 train.py
```

Training starts from `model.pt` if it exists. Otherwise it starts from random weights. The run saves the pretrained weights to `model.pt`, then fine-tunes the same model and saves the result to `titusai-finetuned.pt`.
