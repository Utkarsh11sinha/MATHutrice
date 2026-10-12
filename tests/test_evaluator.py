"""
Spec: LLM Evaluation Deadline Module (Issue #62).

The evaluator takes its OpenAI client as a dependency, holds one deadline for
the whole evaluation, gives each call only the time that remains, and stops
starting calls once the deadline is reached, reporting a timeout instead.

No test waits in real time: the evaluator's clock is replaced by a fake one,
which the mocked LLM moves forward by the duration of each call.
"""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest

START = 1000.0
DEADLINE_S = 300  # Assumption D: 5 min before a student gives up
TYPICAL_CALL_S = 2.3  # Assumption B

PASSE_1 = {
    "competences_evaluees": [
        {
            "code": "EQ2",
            "nom": "Résoudre une équation du second degré",
            "notion": "Équations – Inéquations",
            "justification": "Calcul du discriminant puis des racines.",
        }
    ]
}

PASSE_2 = {
    "correct": False,
    "diagnostic": {
        "type_erreur": "prerequis_manquant",
        "gravite": "bloquante",
        "competences_lacunaires": [
            {
                "code": "RAD1",
                "nom": "Calculer une racine carrée",
                "notion": "Fractions – Puissances – Radicaux",
                "source": "detectee_passe2",
                "explication": "√1 = 0",
            }
        ],
    },
    "feedback_etudiant": "Revois la racine carrée de 1.",
    "confiance_diagnostic": 0.9,
}

EXERCICE = dict(
    notion="Équations – Inéquations",
    niveau="solide",
    enonce="Résoudre x² - 7x + 12 = 0",
    reponse_correcte="Δ = 1, x₁ = 4, x₂ = 3",
    reponse_etudiant="Comme √1 = 0, x = 3.5",
)


class FakeClock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def evaluator():
    # Imported after conftest has set the dummy LLM settings.
    return importlib.import_module("mathutrice.lacune_evaluation.LLM_as_Evaluator")


@pytest.fixture
def clock(evaluator, monkeypatch):
    fake = FakeClock(START)
    monkeypatch.setattr(
        evaluator, "time", SimpleNamespace(monotonic=fake), raising=False
    )
    return fake


@pytest.fixture
def global_client(evaluator, monkeypatch):
    """Stands in for the shared client: the evaluator must never touch it."""
    shared = MagicMock(name="global llm_client")
    monkeypatch.setattr("mathutrice.llm_client.client", shared)
    return shared


def make_client(clock, call_s):
    """A mocked OpenAI client whose every call takes `call_s` on the fake clock."""
    client = MagicMock(name="injected OpenAI client")
    client.with_options.return_value = client

    def create(*, model, messages, **kwargs):
        clock.now += call_s
        prompt = messages[-1]["content"]
        payload = PASSE_2 if "diagnostic pédagogique" in prompt else PASSE_1
        message = SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client.chat.completions.create.side_effect = create
    return client


def test_uses_injected_client_and_keeps_existing_result(
    evaluator, clock, global_client
):
    client = make_client(clock, TYPICAL_CALL_S)

    resultat = evaluator.diagnostiquer(
        **EXERCICE, client=client, deadline=START + DEADLINE_S
    )

    assert client.chat.completions.create.call_count == 2
    global_client.chat.completions.create.assert_not_called()
    assert resultat == {
        "competences_evaluees": PASSE_1["competences_evaluees"],
        "diagnostic": PASSE_2,
        "competences_dict": {"EQ2": True, "RAD1": False},
    }


def test_each_call_gets_only_the_remaining_time_and_no_retries(evaluator, clock):
    client = make_client(clock, TYPICAL_CALL_S)

    evaluator.diagnostiquer(**EXERCICE, client=client, deadline=START + DEADLINE_S)

    options = [c.kwargs for c in client.with_options.call_args_list]
    assert options == [
        {"timeout": pytest.approx(DEADLINE_S), "max_retries": 0},
        {"timeout": pytest.approx(DEADLINE_S - TYPICAL_CALL_S), "max_retries": 0},
    ]


def test_deadline_reached_between_passes_stops_before_pass_2(evaluator, clock):
    client = make_client(clock, DEADLINE_S + 1)

    with pytest.raises(evaluator.EvaluationTimeout):
        evaluator.diagnostiquer(
            **EXERCICE, client=client, deadline=START + DEADLINE_S
        )

    assert client.chat.completions.create.call_count == 1


def test_deadline_already_passed_starts_no_call(evaluator, clock):
    client = make_client(clock, TYPICAL_CALL_S)

    with pytest.raises(evaluator.EvaluationTimeout):
        evaluator.diagnostiquer(**EXERCICE, client=client, deadline=START - 1)

    client.chat.completions.create.assert_not_called()


def test_call_timing_out_in_flight_is_reported_as_timeout(evaluator, clock):
    client = make_client(clock, TYPICAL_CALL_S)
    client.chat.completions.create.side_effect = openai.APITimeoutError(
        request=httpx.Request("POST", "http://llm.invalid/v1/chat/completions")
    )

    with pytest.raises(evaluator.EvaluationTimeout):
        evaluator.diagnostiquer(
            **EXERCICE, client=client, deadline=START + DEADLINE_S
        )


def test_single_pass_diagnosis_honours_the_deadline(evaluator, clock):
    client = make_client(clock, TYPICAL_CALL_S)
    competence = {"code": "EQ2", "nom": "Résoudre", "niveau": "solide"}

    with pytest.raises(evaluator.EvaluationTimeout):
        evaluator.diagnostiquer_depuis_competence(
            **EXERCICE, competence_cible=competence, client=client, deadline=START
        )

    client.chat.completions.create.assert_not_called()


def test_acceptance_40_calls_against_a_5_minute_deadline(evaluator, clock):
    """
    Acceptance criterion: 40 LLM calls (20 diagnoses × 2 passes) share one
    5-minute deadline. At 10 s a call only 30 fit; the 31st must not start,
    and a timeout is reported instead of continuing.
    """
    client = make_client(clock, 10)
    deadline = START + DEADLINE_S

    with pytest.raises(evaluator.EvaluationTimeout):
        for _ in range(20):
            evaluator.diagnostiquer(**EXERCICE, client=client, deadline=deadline)

    assert client.chat.completions.create.call_count == 30
    assert clock.now == deadline
