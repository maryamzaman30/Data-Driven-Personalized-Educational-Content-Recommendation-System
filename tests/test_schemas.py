# =========================================================
# File: tests/test_schemas.py
# Description:
#   Unit tests for Pydantic schemas used in the recommendation API.
#   Validates correct object creation, field constraints, and error handling.
# =========================================================

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

# =========================================================
# 1. RecommendationRequest Model Tests
# =========================================================

def test_recommendation_request_valid():
    """
    Test that a valid RecommendationRequest object is created successfully.
    """
    req = RecommendationRequest(
        user_id="u123",
        n_recommendations=5,
        recommendation_type="hybrid"
    )
    assert req.user_id == "u123"
    assert req.n_recommendations == 5
    assert req.recommendation_type == "hybrid"

# =========================================================
# 2. Recommendation Model Tests
# =========================================================

def test_recommendation_model():
    """
    Test creation of a Recommendation object with all valid fields.
    """
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


def test_recommendation_invalid_score():
    """
    Test that an invalid score type raises a validation error.
    """
    with pytest.raises(ValidationError):
        Recommendation(
            item_id="b001",
            score="bad_score",  # Invalid: should be a float
            title="Bad",
            part="A",
            part_id=1,
            subjects=[],
            duration_minutes=2.0,
            type="quiz"
        )

# =========================================================
# 3. RecommendationResponse Model Tests
# =========================================================

def test_recommendation_response():
    """
    Test that a RecommendationResponse with nested Recommendation objects works.
    """
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

# =========================================================
# 4. UserHistoryRequest Model Tests
# =========================================================

def test_user_history_request():
    """
    Test that UserHistoryRequest correctly stores user_id and limit.
    """
    req = UserHistoryRequest(user_id="u42", limit=20)
    assert req.limit == 20

# =========================================================
# 5. HistoryItem & UserHistoryResponse Tests
# =========================================================

def test_history_item_and_response():
    """
    Test that HistoryItem and UserHistoryResponse work together.
    """
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