# File: src/recommender/schemas.py

from pydantic import BaseModel, Field, ConfigDict
from typing import List


class RecommendationRequest(BaseModel):
    user_id: str
    n_recommendations: int = 10
    recommendation_type: str = "hybrid"


class Recommendation(BaseModel):
    item_id: str
    score: float
    title: str
    part: str
    part_id: int
    subjects: List[str]
    duration_minutes: float
    type: str

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "item_id": "bundle_1",
                    "score": 0.85,
                    "title": "Intro to Grammar",
                    "part": "Part 1",
                    "part_id": 1,
                    "subjects": ["grammar", "english"],
                    "duration_minutes": 12.5,
                    "type": "lecture"
                }
            ]
        }
    )


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Recommendation]
    recommendation_type: str


class UserHistoryRequest(BaseModel):
    user_id: str
    limit: int = 50


class HistoryItem(BaseModel):
    question_id: str
    bundle_id: str
    timestamp: str
    is_correct: bool
    elapsed_time: float
    part: str
    subjects: List[str]


class UserHistoryResponse(BaseModel):
    user_id: str
    history: List[HistoryItem]