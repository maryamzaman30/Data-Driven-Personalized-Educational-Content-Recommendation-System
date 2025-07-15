# File: src/recommender/loader.py

import os
import pickle
import torch
from src.recommender.hybrid import SVDHybridRecommender
from src.recommender.ncf import NCF

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

def load_all_models():
    from src.utils.preprocessing import load_clean_data, load_content_model, load_hybrid_model

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
