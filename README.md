TitusAI is starting over.

## Training

Run from the project root.

```bash
torchrun --standalone --nproc-per-node=2 train.py
```

Training starts from `model.pt` if it exists. Otherwise it starts from random weights. Both pretraining and fine-tuning save to `model.pt`.

Fine-tuning runs five epochs with about 80% chat tokens and 20% pretraining replay tokens.

## Browser chat

Run from the project root.

```bash
npm --prefix frontend install
npm --prefix frontend run dev
```

### Manual website releases

The browser loads `frontend/public/model/titusai.onnx` and `frontend/public/model/tokenizer.model`. Training saves its checkpoint in the project root. Updating the browser model is a separate, manual step performed only when explicitly requested.

1. Export the latest project-root `model.pt` to a single ONNX file with 4-bit weight quantization. [ONNX Runtime quantization guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#quantize-to-int4uint4)
2. Verify that the quantized model loads and generates a reply in ONNX Runtime Web.
3. Copy it to `frontend/public/model/titusai.onnx` and keep the matching tokenizer at `frontend/public/model/tokenizer.model`.

The current website uses an INT8 export. `frontend/export_model.py` uses INT8 today, so prepare the 4-bit export when the next release is requested.
