# File: tests/test_schemas.py

import pytest
from pydantic import ValidationError
from src.recommender.schemas import (
    RecommendationRequest,
    Recommendation,
    RecommendationResponse,
    UserHistoryRequest,
    HistoryItem,
    UserHistoryResponse,
)


def test_recommendation_request_valid():
    req = RecommendationRequest(user_id="u123", n_recommendations=5, recommendation_type="hybrid")
    assert req.user_id == "u123"
    assert req.n_recommendations == 5
    assert req.recommendation_type == "hybrid"


def test_recommendation_model():
    rec = Recommendation(
        item_id="b456",
        score=0.92,
        title="Lesson Title",
        part="Part 1",
        part_id=1,
        subjects=["grammar"],
        duration_minutes=3.5,
        type="lecture"
    )
    assert rec.item_id == "b456"
    assert isinstance(rec.subjects, list)


def test_recommendation_response():
    response = RecommendationResponse(
        user_id="u1",
        recommendation_type="hybrid",
        recommendations=[
            Recommendation(
                item_id="b123",
                score=0.91,
                title="Intro",
                part="Part A",
                part_id=0,
                subjects=["listening"],
                duration_minutes=4.2,
                type="video"
            )
        ]
    )
    assert response.user_id == "u1"
    assert len(response.recommendations) == 1


def test_user_history_request():
    req = UserHistoryRequest(user_id="u42", limit=20)
    assert req.limit == 20


def test_history_item_and_response():
    item = HistoryItem(
        question_id="q789",
        bundle_id="b1",
        timestamp="2024-01-01T12:00:00",
        is_correct=True,
        elapsed_time=12.3,
        part="Part 3",
        subjects=["reading"]
    )
    resp = UserHistoryResponse(user_id="u5", history=[item])
    assert len(resp.history) == 1


def test_recommendation_invalid_score():
    with pytest.raises(ValidationError):
        Recommendation(
            item_id="b001",
            score="bad_score",
            title="Bad",
            part="A",
            part_id=1,
            subjects=[],
            duration_minutes=2.0,
            type="quiz"
        )
