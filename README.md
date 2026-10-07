# LLM-Powered Code Review Assistant

## Overview
The **LLM-Powered Code Review Assistant** is a web service that leverages large language models (LLMs) to automatically review Python code, provide suggestions, and flag potential issues. Built with FastAPI and Hugging Face Transformers, the assistant can be integrated into CI pipelines, IDE extensions, or used as a standalone API.

## Features
- **Automated code analysis** using state‑of‑the‑art LLMs.
- **Detailed feedback**: style suggestions, bug detection, and best‑practice recommendations.
- **RESTful API** for easy integration.
- **Dockerized** for consistent deployment across environments.
- **Test suite** with `pytest` for reliable CI.

## Architecture
```
┌─────────────────────┐
│   FastAPI Server    │
│ (uvicorn + reload)  │
└───────▲───────▲─────┘
        │       │
        │       │
        │   ┌───▼───────┐
        │   │  Model    │
        │   │ (Transformers)│
        │   └───────────┘
        │
        ▼
┌─────────────────────┐
│   Docker Container  │
└─────────────────────┘
```

## Tech Stack
- **Python 3.11**
- **FastAPI** – high‑performance API framework.
- **Uvicorn** – ASGI server.
- **Transformers** – Hugging Face models for LLM inference.
- **Pytest** – testing framework.
- **Docker** – containerization.

## Prerequisites
- Python 3.11 or newer.
- `git` (to clone the repository).
- Docker (optional, for containerized deployment).

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-org/llm-powered-code-review-assistant.git
cd llm-powered-code-review-assistant
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install Python dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Install the model locally
The default model (`microsoft/codebert-base`) will be downloaded automatically on first request. To pre‑download it:
```bash
python -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
AutoModelForSeq2SeqLM.from_pretrained('microsoft/codebert-base'); \
AutoTokenizer.from_pretrained('microsoft/codebert-base')"
```

## Configuration
Environment variables can be set in a `.env` file at the project root:

| Variable                | Description                                 | Default                     |
|-------------------------|---------------------------------------------|-----------------------------|
| `HOST`                  | Host address for the API server             | `0.0.0.0`                   |
| `PORT`                  | Port number for the API server              | `8000`                      |
| `MODEL_NAME`            | Hugging Face model identifier                | `microsoft/codebert-base`   |
| `MAX_INPUT_TOKENS`      | Maximum tokens accepted per request         | `1024`                      |
| `LOG_LEVEL`             | Logging verbosity (`debug`, `info`, ...)    | `info`                      |

Load the variables with `python -m dotenv run -- python -m app.main` or rely on Docker to inject them.

## Running the Service

### Locally (development)
```bash
uvicorn app.main:app --host $HOST --port $PORT --reload
```
The API will be reachable at `http://localhost:8000`.

### Production (Gunicorn)
```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b $HOST:$PORT app.main:app
```

### Docker
```bash
docker build -t llm-code-review .
docker run -d -p 8000:8000 --env-file .env llm-code-review
```

## API Endpoints

### `POST /review`
Accepts a JSON payload with a Python code snippet and returns a structured review.

**Request**
```json
{
  "code": "def add(a, b):\n    return a + b"
}
```

**Response**
```json
{
  "issues": [
    {
      "line": 1,
      "type": "style",
      "message": "Consider adding a docstring."
    }
  ],
  "suggestions": [
    "Add type hints for parameters and return value."
  ],
  "summary": "2 suggestions, 0 critical issues."
}
```

### `GET /health`
Simple health‑check endpoint returning `200 OK` with `{"status":"healthy"}`.

## Testing

Run the full test suite with:
```bash
pytest -vv
```

The tests cover:
- API contract validation.
- Model inference integration (mocked for speed).
- Edge‑case handling (empty payloads, oversized inputs).

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make changes** and ensure all new code has docstrings and type hints.

3. **Run tests** and format code:
   ```bash
   black .
   isort .
   pytest
   ```

4. **Commit and push**:
   ```bash
   git add .
   git commit -m "feat: description"
   git push origin feature/your-feature
   ```

5. Open a Pull Request and request a review.

## Contributing
Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style (Black, isort, flake8).
- Writing tests.
- Updating documentation.

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- The model implementation is based on the work of the Hugging Face community.
- FastAPI documentation and examples for building production‑grade APIs.