"""An evaluation's LLM calls share one 5-minute deadline.

Spec: LLM Evaluation Deadline Module, a comment on the Design Document,
EPF-MDE/MATHutrice#62. The numbers are the Design Document's assumptions:
A, 40 LLM calls per evaluation; B, 2.3 s per call; D, 5 minutes before a
student gives up.

No test waits in real time, and none patches a global: the module is handed a
fake client and a fake clock, and time passes only when the fake client answers.
"""

from types import SimpleNamespace

import httpx
import openai
import pytest

from mathutrice.llm_deadline import (
    EVALUATION_DEADLINE_S,
    EvaluationTimeout,
    LLMDeadline,
)

FIVE_MINUTES = 300.0  # Assumption D
CALLS_PER_EVALUATION = 40  # Assumption A
TYPICAL_CALL_S = 2.3  # Assumption B
MESSAGES = [{"role": "user", "content": "Pose une question."}]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeClient:
    """Stands in for `openai.OpenAI`: every answer takes `seconds_per_call`.

    Like the real client, a call that outlasts its timeout raises
    `openai.APITimeoutError` once its retries are spent. Unconfigured, it uses
    the library's defaults: a 600 s timeout and two retries.
    """

    def __init__(self, clock, seconds_per_call, timeout=600.0, max_retries=2, log=None):
        self._clock = clock
        self._seconds_per_call = seconds_per_call
        self._timeout = timeout
        self._max_retries = max_retries
        self.log = log if log is not None else SimpleNamespace(starts=[], options=[])
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def with_options(self, *, timeout, max_retries):
        self.log.options.append({"timeout": timeout, "max_retries": max_retries})
        return FakeClient(
            self._clock, self._seconds_per_call, timeout, max_retries, self.log
        )

    def _create(self, *, model, messages):
        self.log.starts.append(self._clock.now)
        if self._seconds_per_call <= self._timeout:
            self._clock.now += self._seconds_per_call
            message = SimpleNamespace(content="une réponse")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])
        self._clock.now += self._timeout * (1 + self._max_retries)
        raise openai.APITimeoutError(
            request=httpx.Request("POST", "https://llm.invalid/chat/completions")
        )


def run_evaluation(client, clock):
    """Makes every LLM call of the longest evaluation under one deadline."""
    deadline = LLMDeadline(client, "fake-model", clock=clock)
    return [deadline.complete(MESSAGES) for _ in range(CALLS_PER_EVALUATION)]


def test_the_deadline_is_five_minutes():
    assert EVALUATION_DEADLINE_S == FIVE_MINUTES


def test_acceptance_40_calls_stop_at_the_5_minute_deadline():
    """
    Acceptance criterion: 40 calls against a 5-minute deadline. At 10 s a call
    only 30 fit; the 31st must not start, and a timeout is reported instead.
    """
    clock = FakeClock()
    client = FakeClient(clock, seconds_per_call=10.0)

    with pytest.raises(EvaluationTimeout):
        run_evaluation(client, clock)

    assert len(client.log.starts) == 30
    assert clock.now == FIVE_MINUTES


def test_a_typical_evaluation_gets_every_answer():
    clock = FakeClock()
    client = FakeClient(clock, seconds_per_call=TYPICAL_CALL_S)

    answers = run_evaluation(client, clock)

    assert len(answers) == CALLS_PER_EVALUATION
    assert clock.now == pytest.approx(CALLS_PER_EVALUATION * TYPICAL_CALL_S)


def test_each_call_gets_only_the_remaining_time_and_no_retry():
    clock = FakeClock()
    client = FakeClient(clock, seconds_per_call=TYPICAL_CALL_S)
    deadline = LLMDeadline(client, "fake-model", clock=clock)

    deadline.complete(MESSAGES)
    deadline.complete(MESSAGES)

    assert client.log.options == [
        {"timeout": pytest.approx(FIVE_MINUTES), "max_retries": 0},
        {"timeout": pytest.approx(FIVE_MINUTES - TYPICAL_CALL_S), "max_retries": 0},
    ]


def test_a_hung_call_is_reported_as_a_timeout_at_the_deadline():
    clock = FakeClock()
    client = FakeClient(clock, seconds_per_call=float("inf"))

    with pytest.raises(EvaluationTimeout):
        run_evaluation(client, clock)

    assert len(client.log.starts) == 1
    assert clock.now == FIVE_MINUTES


def test_no_call_starts_once_the_deadline_has_passed():
    clock = FakeClock()
    client = FakeClient(clock, seconds_per_call=TYPICAL_CALL_S)
    deadline = LLMDeadline(client, "fake-model", clock=clock)
    clock.now = FIVE_MINUTES

    with pytest.raises(EvaluationTimeout):
        deadline.complete(MESSAGES)

    assert client.log.starts == []
