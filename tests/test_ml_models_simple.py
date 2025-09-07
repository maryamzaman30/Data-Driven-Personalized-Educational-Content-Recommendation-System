# =========================================================
# File: tests/test_ml_models_simple.py
# Description:
#   Unit tests for:
#     - SVDHybridRecommender (SVD-based hybrid)
#     - NCF (Neural Collaborative Filtering)
#     - Content-based similarity (TF-IDF + cosine similarity)
#     - Model persistence (pickle + torch.save)
# =========================================================

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import MagicMock, patch
from src.recommender.hybrid import SVDHybridRecommender
from src.recommender.ncf import NCF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# 1. Fixtures (Test Data & Setup)
# =========================================================

@pytest.fixture
def sample_data():
    """
    Create a sample ratings matrix and mappings for testing recommenders.
    """
    ratings_matrix = np.array([
        [5, 3, 0, 1],
        [0, 4, 5, 2],
        [1, 0, 4, 5],
        [2, 5, 0, 3]
    ])
    user_map = {'u1': 0, 'u2': 1, 'u3': 2, 'u4': 3}
    item_map = {'i1': 0, 'i2': 1, 'i3': 2, 'i4': 3}
    reverse_item_map = {v: k for k, v in item_map.items()}
    return ratings_matrix, user_map, item_map, reverse_item_map

@pytest.fixture
def content_similarity():
    """
    Create a sample item-item content similarity matrix.
    """
    return np.array([
        [1.0, 0.8, 0.3, 0.1],
        [0.8, 1.0, 0.2, 0.4],
        [0.3, 0.2, 1.0, 0.7],
        [0.1, 0.4, 0.7, 1.0]
    ])

@pytest.fixture
def ncf_model():
    """
    Create an NCF model instance with default parameters.
    """
    return NCF(num_users=100, num_items=50, embedding_dim=16)

@pytest.fixture
def temp_model_file(tmp_path):
    """
    Create a temporary file path for saving models.
    """
    return tmp_path / "test_model.pkl"

@pytest.fixture
def sample_texts():
    """
    Provide sample texts for TF-IDF similarity testing.
    """
    return [
        "TOEIC listening part 1 photographs",
        "TOEIC listening part 2 question response",
        "TOEIC reading part 5 incomplete sentences",
        "TOEIC reading part 6 text completion"
    ]

# =========================================================
# 2. Test SVDHybridRecommender
# =========================================================

class TestSVDHybridRecommender:
    """Test suite for SVD-based hybrid recommender"""
    
    def test_svd_hybrid_initialization(self, sample_data):
        """Test SVD recommender initialization"""
        ratings_matrix, user_map, item_map, reverse_item_map = sample_data
        recommender = SVDHybridRecommender(n_factors=2, combine_weight=0.7)
        recommender.initialize(ratings_matrix, user_map, item_map, reverse_item_map)
        
        assert recommender.train_matrix is not None
        assert recommender.user_map == user_map
        assert recommender.item_map == item_map
        assert recommender.combine_weight == 0.7
    
    def test_svd_hybrid_fit(self, sample_data, content_similarity):
        """Test SVD model training"""
        ratings_matrix, user_map, item_map, reverse_item_map = sample_data
        recommender = SVDHybridRecommender(n_factors=2, combine_weight=0.7)
        recommender.initialize(ratings_matrix, user_map, item_map, reverse_item_map)
        
        recommender.fit(ratings_matrix, content_similarity)
        
        assert recommender.user_factors is not None
        assert recommender.item_factors is not None
        assert recommender.user_factors.shape[1] == 2
        assert recommender.item_factors.shape[1] == 2
        assert recommender.content_similarity is not None
    
    def test_svd_hybrid_predict_rating(self, sample_data, content_similarity):
        """Test rating prediction"""
        ratings_matrix, user_map, item_map, reverse_item_map = sample_data
        recommender = SVDHybridRecommender(n_factors=2, combine_weight=0.7)
        recommender.initialize(ratings_matrix, user_map, item_map, reverse_item_map)
        recommender.fit(ratings_matrix, content_similarity)
        
        prediction = recommender.predict_rating(0, 1)
        assert isinstance(prediction, (int, float))
        assert not np.isnan(prediction)
    
    def test_svd_hybrid_recommendations(self, sample_data, content_similarity):
        """Test recommendation generation"""
        ratings_matrix, user_map, item_map, reverse_item_map = sample_data
        recommender = SVDHybridRecommender(n_factors=2, combine_weight=0.7)
        recommender.initialize(ratings_matrix, user_map, item_map, reverse_item_map)
        recommender.fit(ratings_matrix, content_similarity)
        
        recommendations = recommender.get_recommendations(
            user_idx=0, 
            n=3, 
            exclude_seen=True, 
            known_items=[0, 1]
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in recommendations)
    
    def test_svd_hybrid_empty_matrix(self):
        """Test handling of empty rating matrix"""
        recommender = SVDHybridRecommender()
        empty_matrix = np.array([])
        
        # Should not raise exception
        recommender.fit(empty_matrix)
        # For empty matrices, factors remain None (early return)
        assert recommender.user_factors is None
    
    def test_svd_hybrid_validation(self):
        """Test validation methods"""
        recommender = SVDHybridRecommender()
        
        with pytest.raises(ValueError, match="Recommender not properly initialized"):
            recommender.validate()
        
        # Initialize but don't train
        recommender.initialize(np.array([[1]]), {'u1': 0}, {'i1': 0}, {0: 'i1'})
        with pytest.raises(ValueError, match="Model not trained"):
            recommender.validate()

# =========================================================
# 3. Test NCF Model
# =========================================================

class TestNCFModel:
    """Test suite for Neural Collaborative Filtering model"""
    
    def test_ncf_initialization(self, ncf_model):
        """Test NCF model initialization"""
        assert ncf_model.user_gmf is not None
        assert ncf_model.item_gmf is not None
        assert ncf_model.user_mlp is not None
        assert ncf_model.item_mlp is not None
        assert ncf_model.mlp is not None
        assert ncf_model.final is not None
    
    def test_ncf_forward_pass(self, ncf_model):
        """Test NCF forward pass"""
        user_ids = torch.tensor([0, 1, 2])
        item_ids = torch.tensor([0, 1, 2])
        
        output = ncf_model(user_ids, item_ids)
        
        assert output.shape == (3,)
        assert torch.all((output >= 0) & (output <= 1))  # Sigmoid output
        assert not torch.isnan(output).any()
    
    def test_ncf_different_batch_sizes(self, ncf_model):
        """Test NCF with different batch sizes"""
        # Single prediction
        output1 = ncf_model(torch.tensor([0]), torch.tensor([0]))
        assert output1.shape == torch.Size([])  # Scalar tensor for single prediction
        
        # Batch prediction
        output2 = ncf_model(torch.tensor([0, 1]), torch.tensor([0, 1]))
        assert output2.shape == (2,)
    
    def test_ncf_embedding_dimensions(self):
        """Test NCF with different embedding dimensions"""
        model = NCF(num_users=10, num_items=5, embedding_dim=32)
        user_ids = torch.tensor([0])
        item_ids = torch.tensor([0])
        
        output = model(user_ids, item_ids)
        assert output.shape == torch.Size([])  # Scalar tensor for single prediction
    
    def test_ncf_custom_hidden_dims(self):
        """Test NCF with custom hidden dimensions"""
        model = NCF(num_users=10, num_items=5, embedding_dim=16, hidden_dims=[32, 16, 8])
        user_ids = torch.tensor([0])
        item_ids = torch.tensor([0])
        
        output = model(user_ids, item_ids)
        assert output.shape == torch.Size([])  # Scalar tensor for single prediction

# =========================================================
# 4. Test Content-Based Similarity
# =========================================================

class TestContentSimilarity:
    """Test suite for content-based similarity methods"""
    
    def test_tfidf_similarity(self, sample_texts):
        """Test TF-IDF similarity calculation"""
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sample_texts)
        
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        assert similarity_matrix.shape == (4, 4)
        assert np.all(np.diag(similarity_matrix) == 1.0)  # Self-similarity
        assert np.all((similarity_matrix >= 0) & (similarity_matrix <= 1))  # Bounded
    
    def test_similarity_matrix_properties(self, sample_texts):
        """Test properties of similarity matrices"""
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(sample_texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Test symmetry
        assert np.allclose(similarity_matrix, similarity_matrix.T)
        
        # Test diagonal is 1
        assert np.allclose(np.diag(similarity_matrix), 1.0)
        
        # Test bounded between 0 and 1
        assert np.all((similarity_matrix >= 0) & (similarity_matrix <= 1))

# =========================================================
# 5. Test Model Persistence
# =========================================================

class TestModelPersistence:
    """Test suite for model saving and loading"""
    
    def test_svd_model_persistence(self, sample_data, temp_model_file):
        """Test SVD model saving and loading"""
        ratings_matrix, user_map, item_map, reverse_item_map = sample_data
        recommender = SVDHybridRecommender(n_factors=2)
        recommender.initialize(ratings_matrix, user_map, item_map, reverse_item_map)
        recommender.fit(ratings_matrix)
        
        # Save model
        import pickle
        with open(temp_model_file, 'wb') as f:
            pickle.dump({
                'recommender': recommender,
                'user_map': user_map,
                'item_map': item_map,
                'reverse_item_map': reverse_item_map
            }, f)
        
        # Load model
        with open(temp_model_file, 'rb') as f:
            loaded_model = pickle.load(f)
        
        assert loaded_model['recommender'].user_factors is not None
        assert loaded_model['recommender'].item_factors is not None
        assert loaded_model['user_map'] == user_map
    
    def test_ncf_model_persistence(self, ncf_model, tmp_path):
        """Test NCF model saving and loading"""
        model_path = tmp_path / "ncf_model.pth"
        
        # Save model
        torch.save(ncf_model.state_dict(), model_path)
        
        # Load model
        loaded_model = NCF(num_users=100, num_items=50, embedding_dim=16)
        loaded_model.load_state_dict(torch.load(model_path))
        
        # Test loaded model
        user_ids = torch.tensor([0])
        item_ids = torch.tensor([0])
        output = loaded_model(user_ids, item_ids)
        assert output.shape == torch.Size([])  # Scalar tensor for single prediction 