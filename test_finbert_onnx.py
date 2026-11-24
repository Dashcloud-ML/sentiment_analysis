
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_DIR = "backend_old/models/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False

session = ort.InferenceSession(
    f"{MODEL_DIR}/model_int8.onnx",
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

labels = ["positive", "negative", "neutral"]

test_cases = [
    "Apple reported record profits and strong revenue growth.",
    "The company suffered significant losses and declining sales.",
    "The company announced its quarterly earnings report today."
]

for text in test_cases:

    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding=True,
        max_length=512
    )

    onnx_inputs = {
        item.name: inputs[item.name].astype(np.int64)
        for item in session.get_inputs()
    }

    logits = session.run(None, onnx_inputs)[0][0]

    exp = np.exp(logits - np.max(logits))
    probabilities = exp / exp.sum()

    label_idx = int(np.argmax(probabilities))

    print("\nText:", text)
    print("Sentiment:", labels[label_idx])
    print("Confidence:", round(float(probabilities[label_idx]), 4))

    print("Scores:", {
        labels[i]: round(float(probabilities[i]), 4)
        for i in range(len(labels))
    })

print("\nONNX inference completed successfully!")
