# File: src/recommender/schemas.py

from pydantic import BaseModel, Field
from typing import List

class RecommendationRequest(BaseModel):
    user_id: str
    n_recommendations: int = 10
    recommendation_type: str = "hybrid"



class Recommendation(BaseModel):
    item_id: str
    score: float = Field(..., example=0.85)
    title: str
    part: str
    part_id: int
    subjects: List[str]
    duration_minutes: float
    type: str


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
