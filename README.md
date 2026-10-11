# Laya MLX JEV Endpoint

FastAPI endpoint exposing `laya-mlx` to mimic the JEV / System-1 decision API.

## Requirements

- Apple Silicon Mac (macOS 14+)
- Python 3.11+
- [uv](https://docs.astral.sh/)

## Setup & Run

```bash
uv sync --python 3.12
uv run python server.py
```

Runs on `http://127.0.0.1:8000`.

> **Note on Workers**: Uvicorn runs with `--workers 1`. MLX utilizes Apple Silicon's unified memory and internal threadpool directly. Multiple worker processes cause redundant model copies in memory and Metal pipeline lock contention.

## Endpoints

### 1. Health check

```bash
curl http://127.0.0.1:8000/health
```

### 2. Model catalog

```bash
curl http://127.0.0.1:8000/v1/models
```

### 3. Decision inference

```bash
curl -X POST http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jev-latest",
    "state": "I was billed twice. Please refund the duplicate today.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
          "billing": "invoices, payments, refunds",
          "technical": "bugs and outages",
          "sales": "new purchases"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical"]
      },
      "is_refund": {
        "type": "noul",
        "instructions": "Does the customer ask for money back?"
      }
    }
  }'
```

Supports string, dictionary, or list states. Question types supported: `choice`, `score`, `noul`.

## Run Tests

```bash
uv run pytest -v
```
