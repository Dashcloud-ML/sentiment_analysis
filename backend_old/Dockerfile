
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PORT=7860
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

# Download optimized FinBERT from Hugging Face
RUN python -c "from pathlib import Path; from urllib.request import urlretrieve; p = Path('/app/models/finbert'); p.mkdir(parents=True, exist_ok=True); base = 'https://huggingface.co/Dashcloud/dristikon-finbert-int8/resolve/main/'; [urlretrieve(base + f, p / f) for f in ['model_int8.onnx', 'tokenizer.json', 'config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt']]"

RUN useradd -m appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "180", "--bind", "0.0.0.0:7860", "run:app"]
