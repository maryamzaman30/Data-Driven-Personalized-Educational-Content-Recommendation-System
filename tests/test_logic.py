# =========================================================
# File: tests/test_logic.py
# Description:
#   Unit tests for recommender logic functions:
#     - get_sbert_recommendations
#     - get_ncf_recommendations
#     - get_hybrid_advanced_recommendations
#   Uses mocks and minimal test data for fast execution.
# =========================================================

import pytest
import pandas as pd
import torch
from unittest.mock import MagicMock
from src.recommender.logic import (
    get_sbert_recommendations,
    get_ncf_recommendations,
    get_hybrid_advanced_recommendations
)

# =========================================================
# 1. Test: SBERT Recommendations
# =========================================================

def test_sbert_recommendations_mock():
    """
    Test get_sbert_recommendations using mocked data.
    Ensures:
        - Output is a list
        - Each recommendation has 'item_id'
    """
    # Mock user interaction history
    mock_df = pd.DataFrame({
        "user_id": ["u1"] * 3,
        "bundle_id": ["b1", "b2", "b3"]
    })
    # Mock bundle info (required by SBERT logic)
    mock_bundle_info = pd.DataFrame({
        "bundle_id": ["b1", "b2", "b3"],
        "text": ["part a", "part b", "part c"]
    })
    # Random SBERT embeddings (shape: num_bundles x embedding_dim)
    mock_embeddings = torch.rand(3, 384)

    # Call function
    results = get_sbert_recommendations(
        user_id="u1",
        merged_df=mock_df,
        bundle_info=mock_bundle_info,
        sbert_embeddings=mock_embeddings,
        n=2
    )
    # Assertions
    assert isinstance(results, list)
    assert all("item_id" in r for r in results)

# =========================================================
# 2. Test: NCF Recommendations
# =========================================================

def test_ncf_recommendations_mock():
    """
    Test get_ncf_recommendations with mocks.
    Ensures:
        - Output is a list
    """
    results = get_ncf_recommendations(
        user_id="u1",
        recommender=MagicMock(),
        item_map={"b1": 0},
        reverse_item_map={0: "b1"},
        bundle_info=pd.DataFrame({"bundle_id": ["b1"]}),
        ncf_model=MagicMock(),
        device="cpu",
        n=1
    )
    # Assertions
    assert isinstance(results, list)

# =========================================================
# 3. Test: Advanced Hybrid Recommendations
# =========================================================

def test_hybrid_advanced_mock():
    """
    Test get_hybrid_advanced_recommendations using minimal mock data.
    Ensures:
        - Output is a list
    """
    # Mock user history
    merged_df = pd.DataFrame({
        "user_id": ["u1"],
        "bundle_id": ["b1"]
    })
    # Mock bundle info with additional metadata fields
    bundle_info = pd.DataFrame({
        "bundle_id": ["b1", "b2"],
        "text": ["a", "b"],
        "part": [1, 2],
        "tags": ["3,5", "2,8"], 
        "part_name": ["Part 1", "Part 2"],
        "subject_category": ["grammar_basic", "listening"]
    })
    # Random SBERT embeddings
    sbert_embeddings = torch.rand(2, 384)

    # Call function
    results = get_hybrid_advanced_recommendations(
        user_id="u1",
        merged_df=merged_df,
        sbert_embeddings=sbert_embeddings,
        bundle_info=bundle_info,
        recommender=MagicMock(),
        item_map={"b1": 0, "b2": 1},
        reverse_item_map={0: "b1", 1: "b2"},
        ncf_model=MagicMock(),
        device="cpu",
        meta_learner=MagicMock(predict_proba=lambda x: [[0.1, 0.9]] * len(x)),
        n=1
    )
    # Assertions
    assert isinstance(results, list)