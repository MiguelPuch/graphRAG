FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libxcb1 \
    libxcb-shm0 \
    libxcb-render0 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libx11-6 \
    libx11-xcb1 \
    libxext6 \
    libxrender1 \
    libxfixes3 \
    libxdamage1 \
    libxcomposite1 \
    libxcursor1 \
    libxrandr2 \
    libxi6 \
    libxtst6 \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libice6 \
    libfontconfig1 \
    libfreetype6 \
    libjpeg62-turbo \
    libpng16-16 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

COPY config.py .
COPY rag_engine.py .
COPY graph_rag_engine.py .
COPY graph_text_utils.py .
COPY graph_legal_utils.py .
COPY routing.py .
COPY hybrid_engine.py .
COPY api.py .
COPY normativa_sync.py .

RUN mkdir -p /app/data/documents /app/docs /app/normativa

RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appgroup /app

ENV RAG_DATA_DIR=/app/data
ENV RAG_DOCS_DIR=/app/data/documents
ENV RAG_API_HOST=0.0.0.0
ENV RAG_API_PORT=8001
ENV MILVUS_URI=http://milvus:19530
ENV NEO4J_URI=bolt://neo4j:7687
ENV RAG_NORMATIVA_SOURCE_DIR=/app/normativa

EXPOSE 8001

USER appuser

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001", "--timeout-keep-alive", "300"]
