# File: tests/test_metrics.py

import pytest
import numpy as np
from src.evaluation.metrics import precision_at_k, recall_at_k, rmse_score, auc_score, evaluate_user_model

# --- Fixtures ---
@pytest.fixture
def recommended():
    return [1, 2, 3, 4, 5]

@pytest.fixture
def relevant():
    return [3, 4, 5, 6, 7]

# --- Unit Tests ---
def test_precision_at_k(recommended, relevant):
    result = precision_at_k(recommended, relevant, k=5)
    assert result == 3/5

def test_recall_at_k(recommended, relevant):
    result = recall_at_k(recommended, relevant, k=5)
    assert result == 3/5

def test_rmse_score():
    true = [1, 0, 1, 1]
    pred = [0.9, 0.1, 0.8, 0.95]
    rmse = rmse_score(true, pred)
    assert np.isclose(rmse, 0.126, atol=1e-2)


def test_auc_score():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]
    auc = auc_score(y_true, y_scores)
    assert np.isclose(auc, 0.75)

def test_evaluate_user_model_precision_and_recall():
    user_ids = ['u1']

    def recommend_fn(user_id):
        return [1, 2, 3]

    def ground_truth_fn(user_id):
        return [2, 3, 4]

    results = evaluate_user_model(
        user_ids=user_ids,
        recommend_fn=recommend_fn,
        ground_truth_fn=ground_truth_fn,
        k_vals=(3,)
    )

    assert 'precision' in results[3]
    assert 'recall' in results[3]
    assert np.isclose(results[3]['precision'], 2/3)
    assert np.isclose(results[3]['recall'], 2/3)
