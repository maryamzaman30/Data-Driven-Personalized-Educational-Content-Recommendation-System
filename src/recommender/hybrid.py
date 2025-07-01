# File: src/recommender/hybrid.py

import numpy as np
from scipy.sparse.linalg import svds
import numpy as np
from scipy.sparse import csr_matrix
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SVDHybridRecommender:
    """SVD-based hybrid recommender system"""

    def __init__(self, n_factors=20, combine_weight=0.7):
        self.n_factors = n_factors
        self.combine_weight = combine_weight
        self.user_factors = None
        self.item_factors = None
        self.content_similarity = None
        self.mean_rating = None
        self.train_matrix = None
        self.user_map = None
        self.item_map = None
        self.reverse_item_map = None

    def initialize(self, train_matrix, user_map, item_map, reverse_item_map):
        """Initialize the recommender with required matrices and mappings"""
        self.train_matrix = train_matrix
        self.user_map = user_map
        self.item_map = item_map
        self.reverse_item_map = reverse_item_map

    def validate(self):
        """Check if the recommender is properly initialized"""
        if self.train_matrix is None or self.user_map is None or self.item_map is None:
            raise ValueError("Recommender not properly initialized")
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not trained")

    def fit(self, ratings_matrix, content_similarity=None):
        self.content_similarity = content_similarity
        self.mean_rating = np.mean(ratings_matrix[ratings_matrix > 0])
        
        # Early return for empty matrices
        if ratings_matrix.size == 0:
            logger.warning("Empty ratings matrix received")
            return
            
        # Use a smaller number of factors for faster computation
        n_factors = min(self.n_factors, min(ratings_matrix.shape) - 1)
        if n_factors < 1:
            n_factors = 1
            
        # Convert to sparse matrix if not already
        if not isinstance(ratings_matrix, csr_matrix):
            ratings_matrix = csr_matrix(ratings_matrix)
            
        ratings_filled = ratings_matrix.copy()
        ratings_filled[ratings_filled == 0] = self.mean_rating
        
        # Add timeout handling
        try:
            u, sigma, vt = svds(ratings_filled, k=n_factors)
            self.user_factors = u
            self.item_factors = vt.T
            logger.info(f"SVD model trained successfully with {n_factors} factors")
            logger.info(f"User factors shape: {self.user_factors.shape}")
            logger.info(f"Item factors shape: {self.item_factors.shape}")
        except Exception as e:
            logger.error(f"Error in SVD computation: {str(e)}")
            # Use mean rating as fallback
            self.user_factors = np.ones((ratings_matrix.shape[0], n_factors))
            self.item_factors = np.ones((ratings_matrix.shape[1], n_factors))
            logger.warning("Using fallback factors due to SVD computation error")

    def predict_rating(self, user_idx, item_idx):
        if self.user_factors is None or self.item_factors is None:
            return self.mean_rating
        return np.dot(self.user_factors[user_idx], self.item_factors[item_idx])

    def _get_content_score(self, known_items, item_idx):
        if self.content_similarity is None or len(known_items) == 0:
            return 0.0
        return np.mean([self.content_similarity[item_idx, idx] for idx in known_items])

    def get_recommendations(self, user_idx, n=10, exclude_seen=True, known_items=None):
        try:
            self.validate()
            
            cf_scores = np.dot(self.user_factors[user_idx], self.item_factors.T)
            final_scores = cf_scores.copy()

            # Convert numpy arrays to lists for safe handling
            if isinstance(known_items, np.ndarray):
                known_items = known_items.tolist()

            if self.content_similarity is not None and known_items is not None and len(known_items) > 0:
                for i in range(len(final_scores)):
                    cb_score = self._get_content_score(known_items, i)
                    final_scores[i] = (self.combine_weight * cf_scores[i]) + ((1 - self.combine_weight) * cb_score)

            all_items = np.arange(len(final_scores))
            if exclude_seen and known_items is not None and len(known_items) > 0:
                mask = np.ones_like(final_scores, dtype=bool)
                mask[known_items] = False
                all_items = all_items[mask]
                final_scores = final_scores[mask]

            top_indices = np.argsort(-final_scores)[:n]
            top_items = all_items[top_indices]
            top_scores = final_scores[top_indices]

            logger.info(f"Generated {len(top_items)} recommendations for user {user_idx}")
            return list(zip(top_items, top_scores))
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
