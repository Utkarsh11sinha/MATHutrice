import time
from collections.abc import Callable

from openai import OpenAI

# How long a student waits for an evaluation before giving up: assumption D of
# the Design Document, EPF-MDE/MATHutrice#62.
EVALUATION_DEADLINE_S = 300.0


class EvaluationTimeout(Exception):
    """The evaluation's deadline passed: no further LLM call will be made."""


class LLMDeadline:
    """Every LLM call of one evaluation, under one deadline.

    The clock starts when the object is built. Callers pass the client they
    would have called directly; tests pass a fake one, and a fake clock.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        *,
        clock: Callable[[], float] = time.monotonic,
        seconds: float = EVALUATION_DEADLINE_S,
    ) -> None:
        self._client = client
        self._model = model
        self._clock = clock
        self._ends_at = clock() + seconds

    def complete(self, messages: list[dict]) -> str:
        """The model's answer to `messages`, as text."""
        raise NotImplementedError
