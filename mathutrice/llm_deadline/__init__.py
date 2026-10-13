"""One deadline over every LLM call of an evaluation."""

from ._deadline import EVALUATION_DEADLINE_S, EvaluationTimeout, LLMDeadline

__all__ = ["EVALUATION_DEADLINE_S", "EvaluationTimeout", "LLMDeadline"]
