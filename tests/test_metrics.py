# =========================================================
# File: tests/test_metrics.py
# Description:
#   Unit tests for recommendation system evaluation metrics.
#   Tests include:
#       - Precision@K
#       - Recall@K
#       - RMSE
#       - AUC
#       - End-to-end precision/recall evaluation for a user model
# =========================================================

import pytest
import numpy as np
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    rmse_score,
    auc_score,
    evaluate_user_model
)

# =========================================================
# 1. Fixtures
# =========================================================

@pytest.fixture
def recommended():
    """
    Fixture: Mock list of recommended item IDs.
    """
    return [1, 2, 3, 4, 5]

@pytest.fixture
def relevant():
    """
    Fixture: Mock list of relevant (ground truth) item IDs.
    """
    return [3, 4, 5, 6, 7]

# =========================================================
# 2. Metric Unit Tests
# =========================================================

def test_precision_at_k(recommended, relevant):
    """
    Test Precision@K:
    Precision = (# of relevant items in top-K) / K
    """
    result = precision_at_k(recommended, relevant, k=5)
    assert result == 3 / 5


def test_recall_at_k(recommended, relevant):
    """
    Test Recall@K:
    Recall = (# of relevant items in top-K) / (# of relevant items in ground truth)
    """
    result = recall_at_k(recommended, relevant, k=5)
    assert result == 3 / 5


def test_rmse_score():
    """
    Test RMSE (Root Mean Squared Error) between predicted and true values.
    """
    true = [1, 0, 1, 1]
    pred = [0.9, 0.1, 0.8, 0.95]
    rmse = rmse_score(true, pred)
    assert np.isclose(rmse, 0.126, atol=1e-2)  # Allow small floating-point error


def test_auc_score():
    """
    Test AUC (Area Under the ROC Curve) score.
    """
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]
    auc = auc_score(y_true, y_scores)
    assert np.isclose(auc, 0.75)

# =========================================================
# 3. End-to-End Model Evaluation Test
# =========================================================

def test_evaluate_user_model_precision_and_recall():
    """
    Test evaluate_user_model():
    Simulates a recommender function and ground truth retrieval function,
    then verifies that Precision@K and Recall@K are computed correctly.
    """

    # Mock test data
    user_ids = ['u1']

    def recommend_fn(user_id):
        return [1, 2, 3]  # Mock recommendations

    def ground_truth_fn(user_id):
        return [2, 3, 4]  # Mock ground truth

    # Run evaluation
    results = evaluate_user_model(
        user_ids=user_ids,
        recommend_fn=recommend_fn,
        ground_truth_fn=ground_truth_fn,
        k_vals=(3,)
    )

    # Assertions
    assert 'precision' in results[3]
    assert 'recall' in results[3]
    assert np.isclose(results[3]['precision'], 2 / 3)
    assert np.isclose(results[3]['recall'], 2 / 3)