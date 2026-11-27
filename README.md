TitusAI is starting over.

## Training

Run from the project root.

```bash
torchrun --standalone --nproc-per-node=2 train.py
```

Training starts from `model.pt` if it exists. Otherwise it starts from random weights. The run saves the pretrained weights to `model.pt`, then fine-tunes the same model and saves the result to `titusai-finetuned.pt`.

## Browser chat

Run from the project root.

```bash
npm --prefix frontend install
npm --prefix frontend run dev
```

To build or refresh the browser model, export the fine-tuned checkpoint and its SQLite tokenizer from the project root.

```bash
python -m pip install onnx onnxruntime
python frontend/export_model.py
```
