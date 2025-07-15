# File: src/recommender/loader.py
import pickle
import torch
from src.recommender.ncf import NCF


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

def load_advanced_model(path='models/advanced_hybrid_model.pkl'):
    with open(path, 'rb') as f:
        model_package = pickle.load(f)

    required_keys = {'sbert_model', 'sbert_embeddings', 'ncf_model', 'meta_learner',
                     'bundle_info', 'user_map', 'item_map', 'reverse_item_map'}
    if not required_keys.issubset(model_package):
        raise ValueError("Advanced hybrid model missing components.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ncf_model = NCF(len(model_package['user_map']), len(model_package['item_map'])).to(device)
    ncf_model.load_state_dict(model_package['ncf_model'])
    ncf_model.eval()

    model_package['device'] = device
    model_package['ncf_model'] = ncf_model

    return model_package

def load_content_model(path='models/content_based_model.pkl'):
    """Load precomputed content similarity model"""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Failed to load content model: {e}")
        return None

def load_all_models():
    from src.utils.preprocessing import load_clean_data

    lectures_df, merged_df = load_clean_data()
    content_model = load_content_model('models/content_based_model_best.pkl')
    hybrid_model = load_hybrid_model('models/svd_hybrid_model_best.pkl')
    advanced_model = load_advanced_model('models/advanced_hybrid_model.pkl')

    return {
        "lectures_df": lectures_df,
        "merged_df": merged_df,
        "content_model": content_model,
        "hybrid_model": hybrid_model,
        "advanced_model": advanced_model
    }
