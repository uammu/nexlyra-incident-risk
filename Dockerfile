FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-api.txt requirements-api.lock* ./
RUN pip install -r requirements-api.txt $([ -f requirements-api.lock ] && echo "-c requirements-api.lock")
COPY pyproject.toml ./
COPY src ./src
COPY api ./api
COPY configs ./configs
RUN pip install --no-deps -e . \
 && useradd -m -u 1000 appuser && mkdir -p logs models && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"
# ONE worker: suppression state and metrics live in process memory.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]