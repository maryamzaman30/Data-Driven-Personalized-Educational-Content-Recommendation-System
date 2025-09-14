# =========================================================
# File: api/main.py
# Description:
#   FastAPI backend for generating recommendations using multiple models
#   (Content-based, Collaborative, Hybrid, Advanced Hybrid).
# =========================================================

# Standard Library Imports
import logging

# Third-party Libraries
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Project Imports
from src.recommender.schemas import RecommendationRequest
from src.utils.preprocessing import load_all_models
from src.recommender.logic import (
    get_sbert_recommendations,               # Semantic-based recommendations
    get_ncf_recommendations,                 # Neural collaborative filtering
    get_hybrid_advanced_recommendations,     # Hybrid model combining multiple strategies
    get_svd_collaborative_recommendations    # Matrix factorization-based recommendations
)

# =========================================================
# 1. Logging Setup
# =========================================================

# Define a function to configure logging to both file & console
def setup_logging(log_file='recommendation.log'):
    logging.basicConfig(
        level=logging.INFO, # Set logging level to INFO
        format='%(asctime)s [%(levelname)s] %(message)s',  # Format log messages
        handlers=[
            logging.FileHandler(log_file), # Save logs to a file
            logging.StreamHandler() # Also print logs to the console
        ]
    )

# Call the logging setup function
setup_logging()

# Create a logger instance to use throughout the app
logger = logging.getLogger()

# Log a startup message to indicate the app is initializing
logger.info("Starting FastAPI application and loading models...")

# =========================================================
# 2. FastAPI App Initialization
# =========================================================

# Create a FastAPI app instance
app = FastAPI()

# Add CORS middleware to allow requests from any origin (use specific domains in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow requests from all domains
    allow_credentials=True, # Allow sending cookies & auth headers
    allow_methods=["*"], # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all request headers
)

# =========================================================
# 3. Load All Models
# =========================================================

# Load all pre-trained recommendation models into memory
models = load_all_models()

# Log a message confirming that models were loaded successfully
logger.info("Models loaded successfully.")

# =========================================================
# 4. Root Endpoint
# =========================================================

# Define a GET endpoint at the root URL to provide API documentation
@app.get("/")
async def root():
    """
    Root endpoint that provides basic API information, available endpoints,
    and links to interactive documentation.
    """
    return {
        "message": "Educational Content Recommendation System API",
        "description": (
            "This API provides personalized TOEIC study recommendations, "
            "user history access, and system health monitoring."
        ),
        "docs": {
            "Swagger UI": "/docs",
            "ReDoc": "/redoc"
        },
        "endpoints": {
            "GET /": "This information page",
            "GET /health": "Health check endpoint",
            "GET /users": "List all available users",
            "GET /user/{user_id}/history": "Get user interaction history",
            "POST /recommendations": (
                "Generate personalized recommendations "
                "(requires JSON body with user_id, recommendation_type, n_recommendations)"
            ),
        },
    }

# =========================================================
# 5. Health Check Endpoint
# =========================================================

# Define a GET endpoint at /health to check if the API is running
@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    Returns status if the server is up.
    """
    logger.info("Health check endpoint called.") # Log when this endpoint is accessed
    return {"status": "ok"} # Return a basic JSON response

# =========================================================
# 5. Get All Users
# =========================================================

# Define a GET endpoint at /users to fetch all user IDs
@app.get("/users")
def get_users():
    """
    Retrieve sorted list of all user IDs.
    """
    # Extract unique user IDs from the merged DataFrame and sort them
    user_ids = sorted(models["merged_df"]["user_id"].unique())

    # Log how many user IDs were fetched
    logger.info(f"Fetched {len(user_ids)} users.")

    # Return the list of user IDs as a JSON response
    return {"users": user_ids}

# =========================================================
# 6. Get User Interaction History
# =========================================================

# Define a GET endpoint to fetch a user's interaction history
@app.get("/user/{user_id}/history")
def user_history(user_id: str):
    """
    Retrieve the interaction history for a specific user.
    """
    # Log the request for tracking
    logger.info(f"Fetching history for user_id={user_id}")

    # Access the merged dataset containing user interactions
    df = models["merged_df"]
    
    # Check if the user ID exists in the dataset
    if user_id not in df["user_id"].unique():
        logger.warning(f"User not found: {user_id}")  # Log warning if user is missing
        raise HTTPException(status_code=404, detail="User not found")  # Return 404 error

    # Filter and sort the user's history by timestamp
    history = df[df["user_id"] == user_id].sort_values("timestamp")

    # Log how many records were found
    logger.info(f"User history found: {len(history)} records")

    # Return the history as a list of records and the total count
    return {
        "history": history.to_dict(orient="records"),
        "total_interactions": len(history)
    }

# =========================================================
# 7. Recommendation Endpoint
# =========================================================

# Define a POST endpoint to generate recommendations for a user
@app.post("/recommendations")
def recommend(req: RecommendationRequest):
    """
    Generate recommendations for a given user based on the selected method:
        - content (SBERT-based)
        - collaborative (SVD-based)
        - hybrid (NCF-based)
        - advanced_hybrid (Meta-learned)
    """
    # Extract request parameters
    user_id = req.user_id
    n = req.n_recommendations
    rec_type = req.recommendation_type

    # Log the incoming recommendation request
    logger.info(f"Received recommendation request: user={user_id}, type={rec_type}, top_k={n}")

    # Access the dataset containing user interactions
    df = models["merged_df"]
    
    # Check if the user ID exists in the dataset
    if user_id not in df["user_id"].unique():
        logger.warning(f"Invalid user_id in recommendation request: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")

    try:
        # =========================
        # Select Recommendation Method
        # =========================

        # -------------------------
        # SBERT-based content recommendation
        # -------------------------
        if rec_type == "content":
            recs = get_sbert_recommendations(
                user_id=user_id,
                merged_df=models["merged_df"],
                bundle_info=models["advanced_model"]["bundle_info"],
                sbert_embeddings=models["advanced_model"]["sbert_embeddings"],
                n=n
            )

        # -------------------------
        # SVD-based collaborative filtering
        # -------------------------
        elif rec_type == "collaborative":
            recs = get_svd_collaborative_recommendations(
                user_id=user_id,
                recommender=models["hybrid_model"]["recommender"],
                item_map=models["hybrid_model"]["item_map"],
                reverse_item_map=models["hybrid_model"]["reverse_item_map"],
                bundle_info=models["advanced_model"]["bundle_info"],
                n=n
            )

        # -------------------------
        # Meta-learned advanced hybrid recommendation
        # -------------------------
        elif rec_type == "advanced_hybrid":
            recs = get_hybrid_advanced_recommendations(
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
            )

        # -------------------------
        # NCF-based hybrid recommendation
        # -------------------------
        elif rec_type == "hybrid":
            recs = get_ncf_recommendations(
                user_id=user_id,
                recommender=models["hybrid_model"]["recommender"],
                item_map=models["hybrid_model"]["item_map"],
                reverse_item_map=models["hybrid_model"]["reverse_item_map"],
                bundle_info=models["advanced_model"]["bundle_info"],
                ncf_model=models["advanced_model"]["ncf_model"],
                device=models["advanced_model"]["device"],
                n=n
            )

        # -------------------------
        # Handle invalid recommendation type
        else:
            logger.warning(f"Invalid recommendation type received: {rec_type}")
            raise HTTPException(status_code=400, detail="Invalid recommendation type")

        # Log success and return recommendations
        logger.info(f"Successfully generated {len(recs)} {rec_type} recommendations for user={user_id}")
        return {"recommendations": recs}

    except Exception as e:
        # Log any unexpected errors with full traceback
        logger.error(f"Recommendation error for user={user_id}, type={rec_type}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))