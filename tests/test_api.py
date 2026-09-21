"""Layered checks: service up -> answers sane -> contract honoured -> deterministic."""
import math

import pytest

from fixtures import (
    CANCEL_STATE,
    CASUAL_STATE,
    HAPPY_STATE,
    ROUTING_CASES,
    TRIAGE_QUESTIONS,
    URGENT_STATE,
)

# --- layer 1: service ------------------------------------------------------


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz(client):
    body = client.get("/readyz").json()
    assert body["status"] == "ready"
    assert body["load_seconds"] > 0


def test_info(client):
    info = client.get("/info").json()
    assert info["ready"] is True
    assert info["device"] in {"cpu", "cuda", "mlx"}
    assert info["checkpoint"]


def test_info_names_the_backend(client):
    """Results are not comparable across backends -- the MLX port is an
    independent FP16 conversion -- so /info must always say which one ran."""
    info = client.get("/info").json()
    assert info["backend"] in {"torch", "mlx"}
    assert info["runtime"]
    if info["backend"] == "torch":
        assert info["torch"]
        assert info["dtype"] == "float32"
        assert info["device"] in {"cpu", "cuda"}
    else:
        assert info["device"] == "mlx"


# --- layer 2: the answers are reasonable -----------------------------------


@pytest.mark.parametrize("case_id,state,expected", ROUTING_CASES, ids=[c[0] for c in ROUTING_CASES])
def test_routing_picks_right_department(predict, case_id, state, expected):
    ans = predict(state, TRIAGE_QUESTIONS)["answers"]["department"]
    assert ans["choice"] == expected, f"{case_id}: got {ans['choice']} probs={ans['probabilities']}"
    assert ans["probabilities"][expected] > 0.5


def test_urgent_scores_above_casual(predict):
    """Relative ordering, not an absolute threshold: the rubric is ordinal."""
    urgent = predict(URGENT_STATE, TRIAGE_QUESTIONS)["answers"]["urgency"]["score"]
    casual = predict(CASUAL_STATE, TRIAGE_QUESTIONS)["answers"]["urgency"]["score"]
    assert urgent > casual, f"urgent={urgent} casual={casual}"


def test_explicit_cancellation_beats_happy_customer(predict):
    cancel = predict(CANCEL_STATE, TRIAGE_QUESTIONS)["answers"]["churn_risk"]["noul"]
    happy = predict(HAPPY_STATE, TRIAGE_QUESTIONS)["answers"]["churn_risk"]["noul"]
    assert cancel > happy, f"cancel={cancel} happy={happy}"


# --- layer 3: contract -----------------------------------------------------


def test_all_three_types_answered_in_one_pass(predict):
    res = predict(ROUTING_CASES[0][1], TRIAGE_QUESTIONS)
    answers = res["answers"]
    assert set(answers) == set(TRIAGE_QUESTIONS)
    assert isinstance(answers["department"]["choice"], str)
    assert 0.0 <= answers["urgency"]["score"] <= 2.0
    assert 0.0 <= answers["churn_risk"]["noul"] <= 1.0
    assert res["latency_ms"] > 0


def test_choice_probabilities_normalised(predict):
    ans = predict(ROUTING_CASES[0][1], TRIAGE_QUESTIONS)["answers"]["department"]
    assert set(ans["probabilities"]) == set(TRIAGE_QUESTIONS["department"]["criteria"])
    assert math.isclose(sum(ans["probabilities"].values()), 1.0, abs_tol=0.02)
    assert 0.0 <= ans["confidence"] <= 1.0
    assert ans["choice"] == max(ans["probabilities"], key=ans["probabilities"].get)


def test_plain_string_state_accepted(predict):
    res = predict("Please refund my duplicate invoice.", {"q": TRIAGE_QUESTIONS["department"]})
    assert res["answers"]["q"]["choice"] in TRIAGE_QUESTIONS["department"]["criteria"]


def test_oversized_state_truncates_rather_than_failing(predict):
    """English checkpoint is 512 tokens; a 20k-word state must not 500."""
    res = predict({"body": "refund invoice payment " * 5000}, {"q": TRIAGE_QUESTIONS["department"]})
    assert res["answers"]["q"]["choice"] in TRIAGE_QUESTIONS["department"]["criteria"]


@pytest.mark.parametrize(
    "payload",
    [
        {"state": "x", "questions": {}},
        {"state": "x", "questions": {"q": {"type": "nonsense", "instructions": "hi"}}},
        {"state": "x", "questions": {"q": {"type": "choice", "instructions": "hi", "criteria": {"only": "one"}}}},
        {"state": "x", "questions": {"q": {"type": "score", "instructions": "hi", "criteria": {"not": "a list"}}}},
        {"state": "x", "questions": {"q": {"type": "choice", "instructions": ""}}},
        {"questions": {"q": {"type": "noul", "instructions": "hi"}}},
    ],
    ids=["empty", "bad_type", "one_label", "score_not_list", "blank_instructions", "no_state"],
)
def test_malformed_requests_rejected_with_422(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422, f"expected 422, got {r.status_code}: {r.text[:300]}"


def test_presets_round_trip(client, predict):
    questions = client.get("/presets/triage").json()["questions"]
    assert questions
    res = predict(ROUTING_CASES[0][1], questions)
    assert set(res["answers"]) == set(questions)


def test_unknown_preset_404(client):
    assert client.get("/presets/does_not_exist").status_code == 404


# --- layer 4: determinism --------------------------------------------------


def test_identical_inputs_give_identical_probabilities(predict):
    """Non-autoregressive, no sampling: repeats must match exactly."""
    a = predict(ROUTING_CASES[0][1], TRIAGE_QUESTIONS)["answers"]
    b = predict(ROUTING_CASES[0][1], TRIAGE_QUESTIONS)["answers"]
    assert a["department"]["probabilities"] == b["department"]["probabilities"]
    assert a["urgency"]["score"] == b["urgency"]["score"]
    assert a["churn_risk"]["noul"] == b["churn_risk"]["noul"]


def test_question_order_does_not_change_answers(predict):
    """Questions share one forward pass; one must not leak into another."""
    base = predict(ROUTING_CASES[0][1], TRIAGE_QUESTIONS)["answers"]["department"]["choice"]
    alone = predict(ROUTING_CASES[0][1], {"department": TRIAGE_QUESTIONS["department"]})["answers"]["department"]["choice"]
    assert base == alone
