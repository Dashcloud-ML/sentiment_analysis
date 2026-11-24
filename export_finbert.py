
from pathlib import Path
import torch
import onnx
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_ID = "ProsusAI/finbert"
OUTPUT_DIR = Path("backend_old/models/finbert")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading FinBERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()


class FinBERTWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).logits


wrapper = FinBERTWrapper(model)
wrapper.eval()

inputs = tokenizer(
    "Apple reported strong revenue growth.",
    return_tensors="pt",
    truncation=True,
    max_length=512
)

fp32_path = OUTPUT_DIR / "model.onnx"
int8_path = OUTPUT_DIR / "model_int8.onnx"

print("Exporting FinBERT to ONNX...")

with torch.no_grad():
    torch.onnx.export(
        wrapper,
        (
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"]
        ),
        str(fp32_path),
        input_names=[
            "input_ids",
            "attention_mask",
            "token_type_ids"
        ],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"}
        },
        opset_version=17,
        dynamo=False
    )

onnx.checker.check_model(str(fp32_path))

print("Applying INT8 quantization...")

quantize_dynamic(
    model_input=str(fp32_path),
    model_output=str(int8_path),
    weight_type=QuantType.QInt8
)

print("Saving tokenizer and model configuration...")

tokenizer.save_pretrained(OUTPUT_DIR)
model.config.save_pretrained(OUTPUT_DIR)

print("\nConversion completed successfully!")

print(
    f"Original ONNX size: "
    f"{fp32_path.stat().st_size / 1024**2:.2f} MB"
)

print(
    f"Quantized ONNX size: "
    f"{int8_path.stat().st_size / 1024**2:.2f} MB"
)

print("Model labels:", model.config.id2label)
