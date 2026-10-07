import os
import logging
from functools import lru_cache
from typing import Optional

from transformers import pipeline, Pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import TextGenerationPipeline
from transformers.utils import logging as hf_logging

# Suppress overly verbose warnings from transformers
hf_logging.set_verbosity_error()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReviewerError(Exception):
    """Base exception for all reviewer related errors."""


def _get_model_name() -> str:
    """
    Retrieve the model name to be used for code review generation.

    The model name can be provided via the ``REVIEWER_MODEL`` environment variable.
    If not set, a lightweight default model is used.

    Returns
    -------
    str
        The Hugging Face model identifier.
    """
    return os.getenv("REVIEWER_MODEL", "gpt2")  # gpt2 is small and always available


@lru_cache(maxsize=1)
def _load_generation_pipeline(
    model_name: Optional[str] = None,
    device: int = -1,
    **generation_kwargs,
) -> TextGenerationPipeline:
    """
    Load a text generation pipeline with the specified model.

    The pipeline is cached to avoid re‑loading the model on every request,
    which would be prohibitively slow.

    Parameters
    ----------
    model_name : str, optional
        Hugging Face model identifier. If ``None``, the value from
        :func:`_get_model_name` is used.
    device : int, default ``-1``
        Device to run the model on. ``-1`` means CPU, ``0`` means first GPU.
    **generation_kwargs
        Additional keyword arguments passed to the underlying ``generate`` call.

    Returns
    -------
    TextGenerationPipeline
        A ready‑to‑use pipeline for generating review comments.
    """
    model_name = model_name or _get_model_name()
    try:
        logger.info("Loading model %s (device=%s)", model_name, device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # Move model to the appropriate device if a GPU is available
        if device >= 0:
            model.to(device)

        gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            **generation_kwargs,
        )
        return gen_pipeline
    except Exception as exc:
        raise ReviewerError(f"Failed to load model '{model_name}': {exc}") from exc


def _build_prompt(code: str) -> str:
    """
    Construct a prompt for the LLM that asks it to review the supplied code.

    The prompt is deliberately simple but can be extended with more elaborate
    instructions or few‑shot examples.

    Parameters
    ----------
    code : str
        The source code to be reviewed.

    Returns
    -------
    str
        Prompt string ready for the generation pipeline.
    """
    template = (
        "You are an expert software engineer. Review the following Python code and "
        "provide constructive feedback, pointing out bugs, style issues, potential "
        "optimizations, and any other relevant suggestions.\n\n"
        "```python\n{code}\n```\n\n"
        "Review:"
    )
    return template.format(code=code.rstrip())


def _truncate_code(code: str, max_chars: int = 3000) -> str:
    """
    Truncate the input code to a maximum number of characters.

    This protects the model from receiving inputs that are too large for the
    context window, which would otherwise raise an error.

    Parameters
    ----------
    code : str
        Original source code.
    max_chars : int, default ``3000``
        Maximum number of characters to keep.

    Returns
    -------
    str
        Possibly truncated source code.
    """
    if len(code) <= max_chars:
        return code
    logger.warning(
        "Code length (%d) exceeds %d characters; truncating for LLM input.",
        len(code),
        max_chars,
    )
    # Keep the beginning and the end, which often contain the most important
    # definitions and imports.
    half = max_chars // 2 - 10
    return f"{code[:half]}\n... [truncated] ...\n{code[-half:]}"


class CodeReviewer:
    """
    High‑level interface for generating code review comments using a language model.

    Example
    -------
    >>> reviewer = CodeReviewer()
    >>> review = reviewer.review_code("def add(a,b): return a+b")
    >>> print(review)  # doctest: +SKIP
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: int = -1,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> None:
        """
        Initialise the reviewer with a specific model and generation parameters.

        Parameters
        ----------
        model_name : str, optional
            Hugging Face model identifier. If ``None``, the environment variable
            ``REVIEWER_MODEL`` or the default model is used.
        device : int, default ``-1``
            Device index for model execution. ``-1`` for CPU, ``0`` for first GPU.
        max_new_tokens : int, default ``256``
            Maximum number of tokens the model may generate for the review.
        temperature : float, default ``0.7``
            Sampling temperature; higher values increase randomness.
        top_p : float, default ``0.95``
            Nucleus sampling probability cutoff.
        """
        self.model_name = model_name or _get_model_name()
        self.device = device
        self.generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "eos_token_id": None,  # Let the model decide when to stop
        }
        self._pipeline = _load_generation_pipeline(
            model_name=self.model_name,
            device=self.device,
            **self.generation_kwargs,
        )
        logger.info(
            "CodeReviewer initialised with model=%s, device=%s",
            self.model_name,
            "CPU" if self.device < 0 else f"GPU:{self.device}",
        )

    def review_code(self, code: str) -> str:
        """
        Generate a code review for the supplied Python source.

        Parameters
        ----------
        code : str
            The Python source code to be reviewed.

        Returns
        -------
        str
            The LLM‑generated review text.

        Raises
        ------
        ReviewerError
            If the underlying generation pipeline fails.
        """
        if not isinstance(code, str):
            raise ReviewerError("Code must be a string.")
        if not code.strip():
            raise ReviewerError("Empty code supplied for review.")

        # Ensure the code fits within the model's context window.
        safe_code = _truncate_code(code)

        prompt = _build_prompt(safe_code)

        try:
            logger.debug("Sending prompt to generation pipeline.")
            # The pipeline returns a list of dicts; we take the first result.
            result = self._pipeline(prompt, return_full_text=False)  # type: ignore[arg-type]
            if not result or "generated_text" not in result[0]:
                raise ReviewerError("Generation pipeline returned an unexpected format.")
            review = result[0]["generated_text"].strip()
            logger.debug("Generated review: %s", review)
            return review
        except Exception as exc:
            raise ReviewerError(f"Failed to generate review: {exc}") from exc

    def set_generation_parameters(
        self,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> None:
        """
        Update generation parameters on the fly.

        The underlying pipeline is re‑initialised with the new settings.

        Parameters
        ----------
        max_new_tokens : int, optional
            New maximum token count.
        temperature : float, optional
            New sampling temperature.
        top_p : float, optional
            New nucleus sampling cutoff.
        """
        if max_new_tokens is not None:
            self.generation_kwargs["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            self.generation_kwargs["temperature"] = temperature
        if top_p is not None:
            self.generation_kwargs["top_p"] = top_p

        # Re‑load the pipeline with updated kwargs.
        self._pipeline = _load_generation_pipeline(
            model_name=self.model_name,
            device=self.device,
            **self.generation_kwargs,
        )
        logger.info("Generation parameters updated: %s", self.generation_kwargs)