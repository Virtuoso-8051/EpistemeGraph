# EpistemeGraph - container image (Hugging Face Spaces "Docker" SDK, Render, Railway, Fly.io, any VM)
#
#   docker build -t epistemegraph .
#   docker run -p 7860:7860 --env-file .env -e EPISTEME_PUBLIC_MODE=1 -e EPISTEME_ADMIN_TOKEN=<secret> epistemegraph
#
# Secrets (OPENAI_API_KEY, NEO4J_*, EPISTEME_ADMIN_TOKEN) are passed at run time - never baked into the image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf_cache \
    EPISTEME_CPU_THREADS=2 PORT=7860

RUN useradd -m -u 1000 app
WORKDIR /app

# CPU-only PyTorch first (the default wheel would pull ~2 GB of CUDA libraries)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt setup.py ./
COPY kg_audit ./kg_audit
RUN pip install -r requirements.txt

# bake the two small local models into the image (embedding + NLI), so the first query is fast
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/nli-deberta-v3-xsmall')"

COPY . .
RUN chmod +x deploy/start.sh && chown -R app:app /app
USER app
EXPOSE 7860
CMD ["deploy/start.sh"]
