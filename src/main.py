import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from starlette.responses import JSONResponse

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Pipeline, pipeline

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("code_review_assistant")

# --------------------------------------------------------------------------- #
# FastAPI application definition
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="LLM-Powered Code Review Assistant",
    description=(
        "A lightweight service that leverages a Hugging Face transformer model "
        "to generate automated code reviews."
    ),
    version="1.0.0",
)

# Allow all origins for simplicity; in production restrict as needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Pydantic models for request/response validation
# --------------------------------------------------------------------------- #
class CodeReviewRequest(BaseModel):
    """Payload for a code review request."""

    code: str = Field(..., description="The source code to be reviewed.")
    language: Optional[str] = Field(
        None,
        description=(
            "Programming language of the source code. If omitted, the model "
            "will attempt to infer it."
        ),
    )
    instructions: Optional[str] = Field(
        None,
        description=(
            "Additional instructions for the reviewer, e.g., 'focus on performance'."
        ),
    )

    @validator("code")
    def code_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("code must contain non‑empty content")
        return v


class CodeReviewResponse(BaseModel):
    """Response containing the generated review."""

    review: str = Field(..., description="Generated code review text.")
    model: str = Field(..., description="Identifier of the model used.")
    usage: Optional[dict] = Field(
        None,
        description="Optional token usage statistics (if available).",
    )


# --------------------------------------------------------------------------- #
# Model loading utilities
# --------------------------------------------------------------------------- #
_MODEL_NAME_ENV = "CODE_REVIEW_MODEL"
_DEFAULT_MODEL = "google/flan-t5-base"

_model: Optional[AutoModelForSeq2SeqLM] = None
_tokenizer: Optional[AutoTokenizer] = None
_review_pipeline: Optional[Pipeline] = None


def _load_model() -> None:
    """Load the transformer model and tokenizer based on environment configuration."""
    global _model, _tokenizer, _review_pipeline

    model_name = os.getenv(_MODEL_NAME_ENV, _DEFAULT_MODEL)
    logger.info("Loading model %s for code review", model_name)

    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _review_pipeline = pipeline(
            "text2text-generation",
            model=_model,
            tokenizer=_tokenizer,
            device=0 if torch.cuda.is_available() else -1,  # type: ignore
        )
        logger.info("Model %s loaded successfully", model_name)
    except Exception as exc:
        logger.exception("Failed to load model %s: %s", model_name, exc)
        raise RuntimeError(f"Unable to load model '{model_name}'") from exc


@app.on_event("startup")
async def startup_event() -> None:
    """FastAPI startup hook to initialise the LLM."""
    _load_model()


# --------------------------------------------------------------------------- #
# Health check endpoint
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["Utility"])
async def health_check() -> JSONResponse:
    """Simple health check returning service status."""
    return JSONResponse(
        {"status": "ok", "model_loaded": _model is not None},
        status_code=status.HTTP_200_OK,
    )


# --------------------------------------------------------------------------- #
# Core review endpoint
# --------------------------------------------------------------------------- #
@app.post(
    "/review",
    response_model=CodeReviewResponse,
    status_code=status.HTTP_200_OK,
    tags=["Review"],
)
async def review_code(payload: CodeReviewRequest, request: Request) -> CodeReviewResponse:
    """
    Generate a code review for the supplied source code.

    The request payload must contain the code to be reviewed. Optional language
    and instructions fields can be used to guide the model.
    """
    if _review_pipeline is None:
        logger.error("Model not initialised when handling request")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded; try again later.",
        )

    # Build the prompt for the LLM.
    prompt_parts = ["You are a senior software engineer. Review the following code."]
    if payload.language:
        prompt_parts.append(f"The code is written in {payload.language}.")
    if payload.instructions:
        prompt_parts.append(f"Focus on: {payload.instructions}")
    prompt_parts.append("\n```")
    prompt_parts.append(payload.code)
    prompt_parts.append("```")
    prompt = "\n".join(prompt_parts)

    logger.debug("Generated prompt for review: %s", prompt)

    try:
        # The pipeline returns a list of dictionaries with a 'generated_text' key.
        result = _review_pipeline(
            prompt,
            max_new_tokens=512,
            do_sample=False,
            clean_up_tokenization_spaces=True,
        )
        review_text = result[0]["generated_text"].strip()
    except Exception as exc:
        logger.exception("Error during model inference: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate review.",
        ) from exc

    response = CodeReviewResponse(
        review=review_text,
        model=os.getenv(_MODEL_NAME_ENV, _DEFAULT_MODEL),
        usage=None,  # Placeholder – can be extended with token counters.
    )
    logger.info("Generated review for request from %s", request.client.host)
    return response


# --------------------------------------------------------------------------- #
# Application entry point for local development
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
        reload=False,
    )