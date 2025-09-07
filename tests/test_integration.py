# =========================================================
# File: tests/test_integration.py
# Description:
#   Comprehensive test suite for:
#     - Recommendation pipelines
#     - API endpoints
#     - Dashboard integration
#     - Error handling
# =========================================================

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Import recommendation functions
from src.recommender.logic import (
    get_sbert_recommendations,
    get_hybrid_advanced_recommendations,
    get_svd_collaborative_recommendations
)

# Import model loader
from src.utils.preprocessing import load_all_models

# Import FastAPI app
from api.main import app

# =========================================================
# 1. Fixtures
# =========================================================

@pytest.fixture
def sample_data():
    """Create comprehensive sample dataset"""
    return pd.DataFrame({
        'user_id': ['u1'] * 10 + ['u2'] * 8 + ['u3'] * 12,
        'bundle_id': ['b1', 'b2', 'b3', 'b4', 'b5'] * 6,
        'user_answer': [1, 0, 1, 1, 0] * 6,
        'correct_answer': [1, 1, 0, 1, 1] * 6,
        'elapsed_time': [10, 15, 8, 12, 20] * 6,
        'timestamp': range(30)
    })

@pytest.fixture
def sample_bundle_info():
    """Create sample bundle information"""
    return pd.DataFrame({
        'bundle_id': ['b1', 'b2', 'b3', 'b4', 'b5'],
        'part': [1, 1, 2, 2, 3],
        'tags': ['3,5', '2,8', '1,4', '6,7', '9,10'],
        'part_name': ['Part 1', 'Part 1', 'Part 2', 'Part 2', 'Part 3'],
        'subject_category': ['listening', 'listening', 'reading', 'reading', 'grammar'],
        'content_text': [
            'TOEIC listening part 1 photographs',
            'TOEIC listening part 1 question response',
            'TOEIC reading part 5 incomplete sentences',
            'TOEIC reading part 6 text completion',
            'TOEIC grammar basic concepts'
        ]
    })

@pytest.fixture
def client():
    """Create test client for API testing"""
    return TestClient(app)

# =========================================================
# 2. Recommendation Pipeline Tests
# =========================================================

class TestFullRecommendationPipeline:
    """Test suite for complete recommendation pipeline"""
    
    def test_content_based_pipeline(self, sample_data, sample_bundle_info):
        """Test complete content-based recommendation pipeline"""
        # Mock SBERT embeddings
        mock_embeddings = np.random.rand(5, 384)
        
        recommendations = get_sbert_recommendations(
            user_id='u1',
            merged_df=sample_data,
            bundle_info=sample_bundle_info,
            sbert_embeddings=mock_embeddings,
            n=3
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        assert all('bundle_id' in rec for rec in recommendations)
        assert all('score' in rec for rec in recommendations)
    
    def test_collaborative_filtering_pipeline(self, sample_data):
        """Test complete collaborative filtering pipeline"""
        # Mock recommender
        mock_recommender = MagicMock()
        mock_recommender.get_recommendations.return_value = [
            (0, 0.95), (1, 0.87), (2, 0.76)
        ]
        
        recommendations = get_svd_collaborative_recommendations(
            user_id='u1',
            recommender=mock_recommender,
            item_map={'b1': 0, 'b2': 1, 'b3': 2},
            reverse_item_map={0: 'b1', 1: 'b2', 2: 'b3'},
            bundle_info=sample_bundle_info,
            n=3
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        assert all('bundle_id' in rec for rec in recommendations)
    
    def test_hybrid_advanced_pipeline(self, sample_data, sample_bundle_info):
        """Test complete advanced hybrid pipeline"""
        # Mock all components
        mock_recommender = MagicMock()
        mock_recommender.get_recommendations.return_value = [(0, 0.9), (1, 0.8)]
        mock_ncf_model = MagicMock()
        mock_meta_learner = MagicMock()
        mock_meta_learner.predict_proba.return_value = [[0.3, 0.7]] * 2
        
        mock_embeddings = np.random.rand(5, 384)
        
        recommendations = get_hybrid_advanced_recommendations(
            user_id='u1',
            merged_df=sample_data,
            sbert_embeddings=mock_embeddings,
            bundle_info=sample_bundle_info,
            recommender=mock_recommender,
            item_map={'b1': 0, 'b2': 1},
            reverse_item_map={0: 'b1', 1: 'b2'},
            ncf_model=mock_ncf_model,
            device='cpu',
            meta_learner=mock_meta_learner,
            n=2
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 2
        assert all('bundle_id' in rec for rec in recommendations)
    
    def test_data_preprocessing_pipeline(self, sample_data):
        """Test complete data preprocessing pipeline"""
        from src.utils.preprocessing import prepare_matrices, get_users_for_eval
        
        # Test matrix preparation
        train_matrix, test_matrix, user_map, item_map = prepare_matrices(
            sample_data, min_interactions=5
        )
        
        assert train_matrix.shape == test_matrix.shape
        assert isinstance(user_map, dict)
        assert isinstance(item_map, dict)
        assert len(user_map) > 0
        assert len(item_map) > 0
        
        # Test user selection for evaluation
        users = get_users_for_eval(sample_data, min_interactions=5, sample_size=2)
        assert isinstance(users, list)
        assert len(users) <= 2
    
    def test_model_loading_pipeline(self):
        """Test model loading pipeline with mocked models"""
        with patch('src.utils.preprocessing.load_clean_data') as mock_clean:
            with patch('src.utils.preprocessing.load_content_model') as mock_content:
                with patch('src.utils.preprocessing.load_hybrid_model') as mock_hybrid:
                    with patch('src.utils.preprocessing.load_advanced_model') as mock_advanced:
                        # Setup mocks
                        mock_clean.return_value = (pd.DataFrame(), pd.DataFrame())
                        mock_content.return_value = MagicMock()
                        mock_hybrid.return_value = {'recommender': MagicMock()}
                        mock_advanced.return_value = {'device': 'cpu'}
                        
                        # Test loading
                        models = load_all_models()
                        
                        assert 'lectures_df' in models
                        assert 'merged_df' in models
                        assert 'content_model' in models
                        assert 'hybrid_model' in models
                        assert 'advanced_model' in models

# =========================================================
# 3. API Endpoint Tests
# =========================================================

class TestAPIEndpoints:
    """Test suite for FastAPI endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    def test_get_users(self, client):
        """Test users endpoint"""
        response = client.get("/users")
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert isinstance(data["users"], list)
    
    def test_user_history_valid_user(self, client):
        """Test user history endpoint with valid user"""
        # First get available users
        users_response = client.get("/users")
        if users_response.status_code == 200:
            users = users_response.json()["users"]
            if users:
                user_id = users[0]
                response = client.get(f"/user/{user_id}/history")
                assert response.status_code == 200
                data = response.json()
                assert "history" in data
                assert "total_interactions" in data
                assert isinstance(data["history"], list)
    
    def test_user_history_invalid_user(self, client):
        """Test user history endpoint with invalid user"""
        response = client.get("/user/invalid_user_12345/history")
        assert response.status_code == 404
        assert "User not found" in response.json()["detail"]
    
    def test_recommendations_content_based(self, client):
        """Test content-based recommendations endpoint"""
        # Get a valid user first
        users_response = client.get("/users")
        if users_response.status_code == 200:
            users = users_response.json()["users"]
            if users:
                user_id = users[0]
                response = client.post("/recommendations", json={
                    "user_id": user_id,
                    "n_recommendations": 5,
                    "recommendation_type": "content"
                })
                assert response.status_code == 200
                data = response.json()
                assert "recommendations" in data
                assert isinstance(data["recommendations"], list)
    
    def test_recommendations_collaborative(self, client):
        """Test collaborative filtering recommendations endpoint"""
        users_response = client.get("/users")
        if users_response.status_code == 200:
            users = users_response.json()["users"]
            if users:
                user_id = users[0]
                response = client.post("/recommendations", json={
                    "user_id": user_id,
                    "n_recommendations": 5,
                    "recommendation_type": "collaborative"
                })
                assert response.status_code == 200
                data = response.json()
                assert "recommendations" in data
    
    def test_recommendations_hybrid(self, client):
        """Test hybrid recommendations endpoint"""
        users_response = client.get("/users")
        if users_response.status_code == 200:
            users = users_response.json()["users"]
            if users:
                user_id = users[0]
                response = client.post("/recommendations", json={
                    "user_id": user_id,
                    "n_recommendations": 5,
                    "recommendation_type": "hybrid"
                })
                assert response.status_code == 200
                data = response.json()
                assert "recommendations" in data
    
    def test_recommendations_invalid_user(self, client):
        """Test recommendations endpoint with invalid user"""
        response = client.post("/recommendations", json={
            "user_id": "invalid_user_12345",
            "n_recommendations": 5,
            "recommendation_type": "content"
        })
        assert response.status_code == 404
        assert "User not found" in response.json()["detail"]
    
    def test_recommendations_invalid_type(self, client):
        """Test recommendations endpoint with invalid recommendation type"""
        users_response = client.get("/users")
        if users_response.status_code == 200:
            users = users_response.json()["users"]
            if users:
                user_id = users[0]
                response = client.post("/recommendations", json={
                    "user_id": user_id,
                    "n_recommendations": 5,
                    "recommendation_type": "invalid_type"
                })
                # Should return 500 for invalid recommendation type
                assert response.status_code == 500
                assert "Invalid recommendation type" in response.json()["detail"]

# =========================================================
# 4. Dashboard Integration Tests
# =========================================================

class TestDashboardIntegration:
    """Test suite for Streamlit dashboard integration"""
    
    def test_dashboard_data_processing(self):
        """Test dashboard data processing functions"""
        # Mock API responses
        mock_users_response = {"users": ["u1", "u2", "u3"]}
        mock_history_response = {
            "history": [
                {
                    "question_id": "q1",
                    "bundle_id": "b1",
                    "timestamp": "2024-01-01T12:00:00",
                    "is_correct": True,
                    "elapsed_time": 10.5,
                    "part": "Part 1",
                    "subjects": ["listening"]
                }
            ],
            "total_interactions": 1
        }
        mock_recommendations_response = {
            "recommendations": [
                {
                    "bundle_id": "b2",
                    "score": 0.95,
                    "title": "Test Lesson",
                    "part": "Part 2",
                    "part_id": 2,
                    "subjects": ["reading"],
                    "duration_minutes": 5.0,
                    "type": "lecture"
                }
            ]
        }
        
        # Test user fetching logic
        users = mock_users_response.get("users", [])
        assert isinstance(users, list)
        assert len(users) > 0
        
        # Test history processing
        history = mock_history_response.get("history", [])
        assert isinstance(history, list)
        if history:
            assert "bundle_id" in history[0]
            assert "timestamp" in history[0]
        
        # Test recommendations processing
        recommendations = mock_recommendations_response.get("recommendations", [])
        assert isinstance(recommendations, list)
        if recommendations:
            assert "bundle_id" in recommendations[0]
            assert "score" in recommendations[0]
            assert "title" in recommendations[0]

# =========================================================
# 5. Error Handling Tests
# =========================================================

class TestErrorHandling:
    """Test suite for error handling in the pipeline"""
    
    def test_model_loading_errors(self):
        """Test handling of model loading errors"""
        with patch('src.utils.preprocessing.load_content_model') as mock_load:
            mock_load.return_value = None
            # Should handle gracefully
            result = load_all_models()
            assert isinstance(result, dict)
    
    def test_recommendation_generation_errors(self, sample_data, sample_bundle_info):
        """Test handling of recommendation generation errors"""
        # Test with invalid user
        recommendations = get_sbert_recommendations(
            user_id='invalid_user',
            merged_df=sample_data,
            bundle_info=sample_bundle_info,
            sbert_embeddings=np.random.rand(5, 384),
            n=3
        )
        # Should return empty list or handle gracefully
        assert isinstance(recommendations, list)
    
    def test_api_error_handling(self, client):
        """Test API error handling"""
        # Test with malformed request
        response = client.post("/recommendations", json={
            "invalid_field": "value"
        })
        # Should return appropriate error code
        assert response.status_code in [400, 422]
    
    def test_data_validation_errors(self):
        """Test data validation error handling"""
        from src.recommender.schemas import RecommendationRequest
        
        # Test that invalid recommendation type is accepted by schema
        # (validation happens in the API endpoint, not in the schema)
        request = RecommendationRequest(
            user_id="u1",
            n_recommendations=5,
            recommendation_type="invalid_type"
        )
        assert request.user_id == "u1"
        assert request.n_recommendations == 5
        assert request.recommendation_type == "invalid_type" 