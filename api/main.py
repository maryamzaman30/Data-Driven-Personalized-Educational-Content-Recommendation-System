# File: api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging
import pandas as pd

from src.utils.preprocessing import load_clean_data
from src.utils.mappings import get_toeic_part_mapping, get_subject_categories, enrich_recommendations_with_metadata
from src.recommender.loader import load_all_models
from src.recommender.logic import (
    get_sbert_recommendations,
    get_ncf_recommendations,
    get_hybrid_advanced_recommendations
)
from src.recommender.schemas import (
    RecommendationRequest, RecommendationResponse,
    UserHistoryRequest, UserHistoryResponse,
    HistoryItem
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("recommendation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="TOEIC Recommendation System")

from src.recommender.loader import load_all_models

models = load_all_models()
lectures_df = models["lectures_df"]
merged_df = models["merged_df"]
content_model = models["content_model"]
hybrid_model = models["hybrid_model"]
advanced_model = models["advanced_model"]
sbert_model = advanced_model["sbert_model"]
ncf_model = advanced_model["ncf_model"]
device = advanced_model["device"]


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/users")
async def get_users():
    try:
        users = sorted(merged_df['user_id'].unique())
        return {"users": users}
    except Exception as e:
        logger.error(f"Failed to load users: {e}")
        raise HTTPException(status_code=500, detail="Error loading users")

@app.post("/recommendations", response_model=RecommendationResponse)
async def recommend(req: RecommendationRequest):
    user_id = req.user_id
    n = req.n_recommendations

    try:
        if user_id not in models['user_map']:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        if req.recommendation_type == "hybrid":
            recommendations = get_hybrid_advanced_recommendations(user_id, n, models, merged_df)
        elif req.recommendation_type == "content_based":
            recommendations = get_sbert_recommendations(user_id, n, models, merged_df)
        elif req.recommendation_type == "collaborative":
            recommendations = get_ncf_recommendations(user_id, n, models)
        else:
            raise HTTPException(status_code=400, detail="Invalid recommendation type")

        enriched = enrich_recommendations_with_metadata(recommendations, lectures_df)

        return RecommendationResponse(
            user_id=user_id,
            recommendations=enriched,
            recommendation_type=req.recommendation_type
        )
    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/user/{user_id}/history", response_model=UserHistoryResponse)
async def get_user_history(user_id: str, limit: int = 50):
    try:
        history = merged_df[merged_df['user_id'] == user_id]
        history = history.sort_values("timestamp", ascending=False).head(limit)
        part_names = get_toeic_part_mapping()
        items = [
            HistoryItem(
                question_id=row['question_id'],
                bundle_id=row['bundle_id'],
                timestamp=row['timestamp'],
                is_correct=row['user_answer'] == row['correct_answer'],
                elapsed_time=row['elapsed_time'],
                part=part_names.get(float(row['part']), f"Part {row['part']}"),
                subjects=get_subject_categories(row['tags'])
            )
            for _, row in history.iterrows()
        ]
        return UserHistoryResponse(user_id=user_id, history=items)
    except Exception as e:
        logger.error(f"User history error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch user history")
