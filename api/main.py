# File: api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.recommender.schemas import RecommendationRequest
from src.utils.preprocessing import load_all_models
from src.recommender.logic import (
    get_sbert_recommendations,
    get_ncf_recommendations,
    get_hybrid_advanced_recommendations
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = load_all_models()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/users")
def get_users():
    user_ids = sorted(models["merged_df"]["user_id"].unique())
    return {"users": user_ids}

@app.get("/user/{user_id}/history")
def user_history(user_id: str):
    df = models["merged_df"]
    if user_id not in df["user_id"].unique():
        raise HTTPException(status_code=404, detail="User not found")

    history = df[df["user_id"] == user_id].sort_values("timestamp")
    return {
        "history": history.to_dict(orient="records"),
        "total_interactions": len(history)
    }

@app.post("/recommendations")
def recommend(req: RecommendationRequest):
    user_id = req.user_id
    n = req.n_recommendations
    rec_type = req.recommendation_type

    df = models["merged_df"]
    if user_id not in df["user_id"].unique():
        raise HTTPException(status_code=404, detail="User not found")

    try:
        if rec_type == "content":
            return {"recommendations": get_sbert_recommendations(
                user_id=user_id,
                merged_df=models["merged_df"],
                bundle_info=models["advanced_model"]["bundle_info"],
                sbert_embeddings=models["advanced_model"]["sbert_embeddings"],
                n=n
            )}

        elif rec_type == "collaborative":
            return {"recommendations": get_ncf_recommendations(
                user_id=user_id,
                recommender=models["hybrid_model"]["recommender"],
                item_map=models["hybrid_model"]["item_map"],
                reverse_item_map=models["hybrid_model"]["reverse_item_map"],
                bundle_info=models["advanced_model"]["bundle_info"],
                ncf_model=models["advanced_model"]["ncf_model"],
                device=models["advanced_model"]["device"],
                n=n
            )}

        elif rec_type == "advanced_hybrid":
            return {"recommendations": get_hybrid_advanced_recommendations(
                user_id=user_id,
                merged_df=models["merged_df"],
                sbert_embeddings=models["advanced_model"]["sbert_embeddings"],
                bundle_info=models["advanced_model"]["bundle_info"],
                recommender=models["hybrid_model"]["recommender"],
                item_map=models["hybrid_model"]["item_map"],
                reverse_item_map=models["hybrid_model"]["reverse_item_map"],
                ncf_model=models["advanced_model"]["ncf_model"],
                device=models["advanced_model"]["device"],
                meta_learner=models["advanced_model"]["meta_learner"],
                n=n
            )}

        elif rec_type == "hybrid":
            return {"recommendations": get_ncf_recommendations(
                user_id=user_id,
                recommender=models["hybrid_model"]["recommender"],
                item_map=models["hybrid_model"]["item_map"],
                reverse_item_map=models["hybrid_model"]["reverse_item_map"],
                bundle_info=models["advanced_model"]["bundle_info"],
                ncf_model=models["advanced_model"]["ncf_model"],
                device=models["advanced_model"]["device"],
                n=n
            )}

        else:
            raise HTTPException(status_code=400, detail="Invalid recommendation type")

    except Exception as e:
        print(f"[ERROR] Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
