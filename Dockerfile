FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    PATH=/app/.venv/bin:$PATH

RUN pip install --no-cache-dir 'uv>=0.9,<1'
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
RUN uv sync --frozen --no-dev --extra cuda

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=3s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/ready', timeout=2)"

CMD ["./scripts/run-with-cuda-libs", "whisper-server", "--host", "0.0.0.0"]
