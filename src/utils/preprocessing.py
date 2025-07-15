import re
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def preprocess_content_text(text):
    """
    Clean content text by removing semicolons and collapsing whitespace,
    while preserving meaningful content.
    """
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[;]', ' ', text.lower())  # Only remove semicolons
    return ' '.join(text.split())  # Collapse whitespace

def load_clean_data():
    """Load preprocessed lecture and merged data"""
    try:
        # Load data using relative paths from notebook
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Construct absolute paths
        lectures_path = os.path.join(project_root, 'data', 'cleaned', 'cleaned_lectures.csv')
        merged_path = os.path.join(project_root, 'data', 'cleaned', 'merged_cleaned_data.csv')
        
        # Debug print paths
        print(f"Loading from paths:\nLectures: {lectures_path}\nMerged: {merged_path}")
        
        # Check if files exist
        if not os.path.exists(lectures_path):
            raise FileNotFoundError(f"Lectures file not found at {lectures_path}")
        if not os.path.exists(merged_path):
            raise FileNotFoundError(f"Merged file not found at {merged_path}")
        
        # Load data
        lectures_df = pd.read_csv(lectures_path)
        merged_df = pd.read_csv(merged_path)
        print(f"Successfully loaded {len(lectures_df)} lectures and {len(merged_df)} interactions")
        return lectures_df, merged_df
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        raise
    except pd.errors.EmptyDataError:
        print("Data files are empty")
        raise
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        raise

def get_users_for_eval(merged_df, min_interactions=10, sample_size=50, random_state=42):
    """
    Select sample users for evaluation based on minimum interactions.

    Parameters:
    - merged_df: pd.DataFrame with interaction data
    - min_interactions: minimum number of interactions per user
    - sample_size: number of users to sample
    - random_state: for reproducibility

    Returns:
    - List of user IDs
    """
    user_counts = merged_df['user_id'].value_counts()
    eligible_users = user_counts[user_counts >= min_interactions].index
    return list(np.random.default_rng(random_state).choice(eligible_users, size=min(sample_size, len(eligible_users)), replace=False))

def prepare_matrices(merged_df, min_interactions=5):
    user_counts = merged_df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    filtered_data = merged_df[merged_df['user_id'].isin(valid_users)].copy()
    filtered_data['correct'] = (filtered_data['user_answer'] == filtered_data['correct_answer']).astype(int)
    user_item_data = filtered_data.groupby(['user_id', 'bundle_id']).agg(
        correctness_rate=('correct', 'mean'),
        interaction_count=('correct', 'count')
    ).reset_index()
    user_ids = filtered_data['user_id'].unique()
    bundle_ids = filtered_data['bundle_id'].unique()
    user_map = {user_id: idx for idx, user_id in enumerate(np.unique(user_ids))}
    item_map = {bundle_id: idx for idx, bundle_id in enumerate(np.unique(bundle_ids))}
    ratings_matrix = np.zeros((len(user_map), len(item_map)))
    for _, row in user_item_data.iterrows():
        if row['user_id'] in user_map and row['bundle_id'] in item_map:
            u = user_map[row['user_id']]
            i = item_map[row['bundle_id']]
            ratings_matrix[u, i] = row['interaction_count'] * (1 + 0.4 * row['correctness_rate'])
    interactions = [(u, i, ratings_matrix[u, i]) for u in range(ratings_matrix.shape[0])
                    for i in range(ratings_matrix.shape[1]) if ratings_matrix[u, i] > 0]
    train, test = train_test_split(interactions, test_size=0.2, random_state=42)
    train_matrix, test_matrix = np.zeros_like(ratings_matrix), np.zeros_like(ratings_matrix)
    for u, i, r in train: train_matrix[u, i] = r
    for u, i, r in test: test_matrix[u, i] = r
    return train_matrix, test_matrix, user_map, item_map

def load_content_model(path='models/content_based_model.pkl'):
    """Load precomputed content similarity model"""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Failed to load content model: {e}")
        return None

def load_hybrid_model(path='../models/svd_hybrid_model.pkl'):
    """Load pre-trained hybrid model and auto-initialize if needed"""
    try:
        with open(path, 'rb') as f:
            model_package = pickle.load(f)
            required_keys = ['recommender', 'train_matrix', 'user_map', 'item_map', 'reverse_item_map']
            if not all(key in model_package for key in required_keys):
                raise ValueError(f"Incomplete model package. Missing keys: {required_keys - set(model_package.keys())}")

            recommender = model_package['recommender']
            train_matrix = model_package['train_matrix']
            
            # Initialize the recommender with required components
            recommender.initialize(
                train_matrix=train_matrix,
                user_map=model_package['user_map'],
                item_map=model_package['item_map'],
                reverse_item_map=model_package['reverse_item_map']
            )

            # Ensure SVD factors are computed
            if recommender.user_factors is None or recommender.item_factors is None:
                recommender.fit(train_matrix, model_package.get('content_similarity'))

            # Validate initialization
            if recommender.train_matrix is None or \
               recommender.user_map is None or \
               recommender.item_map is None:
                raise ValueError("Recommender initialization failed")

            return model_package

    except FileNotFoundError:
        print(f"Model file not found at {path}")
        raise
    except Exception as e:
        print(f"Failed to load hybrid model: {e}")
        raise ValueError(f"Failed to load hybrid model: {e}")