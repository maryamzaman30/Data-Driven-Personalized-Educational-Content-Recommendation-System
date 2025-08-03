import pytest
import pandas as pd
import torch
from unittest.mock import MagicMock
from src.recommender.logic import (
    get_sbert_recommendations,
    get_ncf_recommendations,
    get_hybrid_advanced_recommendations
)

# --- SBERT ---
def test_sbert_recommendations_mock():
    mock_df = pd.DataFrame({
        "user_id": ["u1"] * 3,
        "bundle_id": ["b1", "b2", "b3"]
    })
    mock_bundle_info = pd.DataFrame({
        "bundle_id": ["b1", "b2", "b3"],
        "text": ["part a", "part b", "part c"]
    })
    mock_embeddings = torch.rand(3, 384)

    results = get_sbert_recommendations(
        user_id="u1",
        merged_df=mock_df,
        bundle_info=mock_bundle_info,
        sbert_embeddings=mock_embeddings,
        n=2
    )
    assert isinstance(results, list)
    assert all("item_id" in r for r in results)

# --- NCF ---
def test_ncf_recommendations_mock():
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
    assert isinstance(results, list)

# --- Hybrid Advanced ---
def test_hybrid_advanced_mock():
    merged_df = pd.DataFrame({
        "user_id": ["u1"],
        "bundle_id": ["b1"]
    })
    bundle_info = pd.DataFrame({
        "bundle_id": ["b1", "b2"],
        "text": ["a", "b"],
        "part": [1, 2],
        "tags": ["3,5", "2,8"], 
        "part_name": ["Part 1", "Part 2"],
        "subject_category": ["grammar_basic", "listening"]
    })
    sbert_embeddings = torch.rand(2, 384)

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
    assert isinstance(results, list)

