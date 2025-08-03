import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.utils import preprocessing

def test_preprocess_content_text():
    assert preprocessing.preprocess_content_text("This; is ;a test;") == "this is a test"
    assert preprocessing.preprocess_content_text("Text   with   space") == "text with space"
    assert preprocessing.preprocess_content_text(None) == ""

def test_get_users_for_eval():
    df = pd.DataFrame({
        "user_id": ["u1"] * 10 + ["u2"] * 5,
        "interaction": list(range(15))
    })
    users = preprocessing.get_users_for_eval(df, min_interactions=10, sample_size=1, random_state=42)
    assert users == ["u1"]

def test_prepare_matrices():
    df = pd.DataFrame({
        "user_id": ["u1"] * 6 + ["u2"] * 6,
        "bundle_id": ["b1", "b1", "b2", "b2", "b3", "b3"] * 2,
        "user_answer": [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        "correct_answer": [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    })
    train, test, user_map, item_map = preprocessing.prepare_matrices(df, min_interactions=5)
    assert train.shape == test.shape
    assert isinstance(user_map, dict) and isinstance(item_map, dict)

def test_load_content_model_invalid():
    assert preprocessing.load_content_model("nonexistent.pkl") is None

def test_load_hybrid_model_invalid():
    with pytest.raises(Exception):
        preprocessing.load_hybrid_model("nonexistent.pkl")

def test_load_advanced_model_invalid():
    with pytest.raises(FileNotFoundError):
        preprocessing.load_advanced_model("nonexistent.pkl")

def test_load_all_models_mocked():
    with patch("src.utils.preprocessing.load_clean_data") as mock_clean:
        with patch("src.utils.preprocessing.load_content_model") as mock_content:
            with patch("src.utils.preprocessing.load_hybrid_model") as mock_hybrid:
                with patch("src.utils.preprocessing.load_advanced_model") as mock_advanced:
                    mock_clean.return_value = (pd.DataFrame(), pd.DataFrame())
                    mock_content.return_value = MagicMock()
                    mock_hybrid.return_value = {}
                    mock_advanced.return_value = {"device": "cpu"}
                    all_models = preprocessing.load_all_models()
                    assert "lectures_df" in all_models
                    assert "advanced_model" in all_models
