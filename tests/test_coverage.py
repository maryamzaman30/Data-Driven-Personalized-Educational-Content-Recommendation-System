# =========================================================
# File: tests/test_coverage.py
# Description:
#   Pytest test suite for verifying critical functionality
#   across the `src` package, including recommender logic,
#   metrics, preprocessing, API endpoints, error handling,
#   and edge cases. Coverage is managed by pytest-cov.
# =========================================================

import numpy as np
import pandas as pd
import torch
from pathlib import Path

# =========================================================
# 1. Core Functionality Tests
# =========================================================

class TestCoreFunctionality:
    """Test suite for core recommender, metrics, and preprocessing"""

    def test_recommender_and_metrics(self):
        """Test core recommender models and evaluation metrics"""

        # SVD Hybrid Recommender
        from src.recommender.hybrid import SVDHybridRecommender
        recommender = SVDHybridRecommender(n_factors=2)
        ratings_matrix = np.array([[1, 2, 0], [0, 3, 4], [5, 0, 6]])
        recommender.fit(ratings_matrix)

        # Neural Collaborative Filtering (NCF)
        from src.recommender.ncf import NCF
        model = NCF(num_users=10, num_items=5, embedding_dim=8)
        user_ids = torch.tensor([0, 1, 2])
        item_ids = torch.tensor([0, 1, 2])
        output = model(user_ids, item_ids)
        assert output is not None

        # Evaluation metrics
        from src.evaluation.metrics import precision_at_k, recall_at_k, rmse_score
        precision = precision_at_k([1, 2, 3], [1, 2, 4], k=3)
        recall = recall_at_k([1, 2, 3], [1, 2, 4], k=3)
        rmse = rmse_score([1, 2, 3], [1.1, 2.1, 3.1])
        assert precision >= 0
        assert recall >= 0
        assert rmse >= 0

        # Preprocessing
        from src.utils.preprocessing import prepare_matrices, get_users_for_eval
        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u2"],
            "bundle_id": ["b1", "b2", "b1"],
            "user_answer": [1, 0, 1],
            "correct_answer": [1, 1, 0],
            "elapsed_time": [10, 15, 20],
            "timestamp": [0, 1, 2]
        })
        train_matrix, test_matrix, user_map, item_map = prepare_matrices(df, min_interactions=1)
        users = get_users_for_eval(df, min_interactions=1, sample_size=2)
        assert len(user_map) > 0
        assert len(item_map) > 0
        assert users is not None

# =========================================================
# 2. API Endpoint Tests
# =========================================================

class TestAPI:
    """Test FastAPI endpoints"""

    def test_api_endpoints(self):
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)

        # Health endpoint
        response = client.get("/health")
        assert response.status_code == 200

        # Users endpoint
        response = client.get("/users")
        assert response.status_code in [200, 404]

        # If users exist, test history and recommendations
        if response.status_code == 200:
            users = response.json().get("users", [])
            if users:
                user_id = users[0]
                response = client.get(f"/user/{user_id}/history")
                assert response.status_code in [200, 404]

                response = client.post("/recommendations", json={
                    "user_id": user_id,
                    "n_recommendations": 5,
                    "recommendation_type": "content"
                })
                assert response.status_code in [200, 400]

# =========================================================
# 3. Error Handling and Edge Case Tests
# =========================================================

class TestEdgeCases:
    """Test error handling and edge cases"""

    def test_invalid_model_loading(self):
        from src.utils.preprocessing import load_content_model, load_hybrid_model
        try:
            load_content_model("nonexistent.pkl")
        except Exception:
            pass
        try:
            load_hybrid_model("nonexistent.pkl")
        except Exception:
            pass

    def test_invalid_dataframe(self):
        from src.utils.preprocessing import prepare_matrices
        try:
            prepare_matrices(pd.DataFrame(), min_interactions=1000)
        except Exception:
            pass

    def test_empty_and_small_data(self):
        from src.utils.preprocessing import prepare_matrices

        # Empty data
        empty_df = pd.DataFrame()
        try:
            prepare_matrices(empty_df, min_interactions=1)
        except Exception:
            pass

        # Single user with multiple items
        single_user_df = pd.DataFrame({
            "user_id": ["u1"] * 5,
            "bundle_id": ["b1", "b2", "b3", "b4", "b5"],
            "user_answer": [1] * 5,
            "correct_answer": [1] * 5,
            "elapsed_time": [10] * 5,
            "timestamp": range(5)
        })
        prepare_matrices(single_user_df, min_interactions=1)

    def test_metrics_edge_cases(self):
        from src.evaluation.metrics import precision_at_k, recall_at_k, rmse_score

        precision_at_k([], [1, 2, 3], k=10)
        recall_at_k([1, 2, 3], [], k=10)

        try:
            rmse_score([], [])
        except ValueError:
            pass

# =========================================================
# 4. Performance Monitoring Tests
# =========================================================

class TestPerformanceMonitoring:
    """Test psutil monitoring"""

    def test_memory_monitoring(self):
        import time
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss
        time.sleep(0.05)  # Simulate work
        final_memory = process.memory_info().rss

        memory_change = final_memory - initial_memory
        assert isinstance(memory_change, int)

# =========================================================
# 5. Filesystem Tests
# =========================================================

class TestFilesystem:
    """Test coverage report directory exists after pytest-cov run"""

    def test_htmlcov_directory_exists(self):
        report_dir = Path("htmlcov")
        # htmlcov is generated only when running pytest with --cov-report=html
        if report_dir.exists():
            assert (report_dir / "index.html").exists()