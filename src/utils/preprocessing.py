import re
import os
import pandas as pd
import numpy as np

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
