
from pathlib import Path
from threading import Lock

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class SentimentAnalyzer:

    _instance = None
    _instance_lock = Lock()
    _model_lock = Lock()
    _inference_lock = Lock()

    MODEL_DIR = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "finbert"
    )

    LABELS = ["positive", "negative", "neutral"]

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.tokenizer = None
                cls._instance.model = None

        return cls._instance

    def load_model(self):
        with self._model_lock:
            if self.model is None:

                print("Loading optimized FinBERT...", flush=True)

                tokenizer = Tokenizer.from_file(
                    str(self.MODEL_DIR / "tokenizer.json")
                )

                tokenizer.enable_truncation(max_length=512)

                options = ort.SessionOptions()

                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                options.enable_cpu_mem_arena = False
                options.enable_mem_pattern = False

                model = ort.InferenceSession(
                    str(self.MODEL_DIR / "model_int8.onnx"),
                    sess_options=options,
                    providers=["CPUExecutionProvider"]
                )

                self.tokenizer = tokenizer
                self.model = model

                print(
                    "Optimized FinBERT loaded successfully",
                    flush=True
                )

    def analyze_sentiment(self, text):

        if not text or len(text.strip()) < 10:
            return {
                "label": "neutral",
                "score": 1.0,
                "scores": {
                    "positive": 0.0,
                    "negative": 0.0,
                    "neutral": 1.0
                }
            }

        self.load_model()

        with self._inference_lock:

            encoding = self.tokenizer.encode(text)

            inputs = {
                "input_ids": np.array(
                    [encoding.ids], dtype=np.int64
                ),
                "attention_mask": np.array(
                    [encoding.attention_mask], dtype=np.int64
                ),
                "token_type_ids": np.array(
                    [encoding.type_ids], dtype=np.int64
                )
            }

            onnx_inputs = {
                item.name: inputs[item.name]
                for item in self.model.get_inputs()
            }

            logits = self.model.run(
                None, onnx_inputs
            )[0][0]

            exp = np.exp(logits - np.max(logits))
            probabilities = exp / exp.sum()

        label_idx = int(np.argmax(probabilities))

        return {
            "label": self.LABELS[label_idx],
            "score": float(probabilities[label_idx]),
            "scores": {
                self.LABELS[i]: float(score)
                for i, score in enumerate(probabilities)
            }
        }

    def analyze_articles(self, articles):

        for article in articles:

            text = (
                f"{article['title']}. "
                f"{article.get('content') or ''}"
            )

            article["sentiment"] = self.analyze_sentiment(text)

        return articles
