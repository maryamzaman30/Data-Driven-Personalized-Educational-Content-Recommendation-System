# File: src/recommender/logic.py

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.mappings import get_subject_categories


def get_sbert_recommendations(user_id, merged_df, bundle_info, sbert_embeddings, n=10):
    user_history = merged_df[merged_df['user_id'] == user_id]
    seen = set(user_history['bundle_id'].unique())
    indices = pd.Series(bundle_info.index, index=bundle_info['bundle_id'])
    sim_matrix = cosine_similarity(sbert_embeddings)
    scores = {}

    for b in seen:
        if b not in indices:
            continue
        idx = indices[b]
        for i, score in enumerate(sim_matrix[idx]):
            b_id = bundle_info.iloc[i]['bundle_id']
            if b_id not in seen and score >= 0.1:
                scores[b_id] = scores.get(b_id, 0) + score

    if not scores:
        return []

    max_score = max(scores.values())
    top_bundles = sorted(((k, v / max_score) for k, v in scores.items()), key=lambda x: x[1], reverse=True)[:n]

    return [
        {
            'item_id': b,
            'score': round(score, 4),
            'title': f"{row['part_name']}: {row['subject_category'].replace('_', ' ').title()}",
            'part': row['part_name'],
            'part_id': row['part'],
            'subjects': get_subject_categories(row['tags']),
            'duration_minutes': 0,
            'type': 'bundle'
        }
        for b, score in top_bundles
        if not bundle_info[bundle_info['bundle_id'] == b].empty
        for row in [bundle_info[bundle_info['bundle_id'] == b].iloc[0]]
    ]

def get_ncf_recommendations(user_id, recommender, item_map, reverse_item_map, bundle_info, ncf_model, device, n=10):
    if user_id not in recommender.user_map:
        return []
    user_idx = recommender.user_map[user_id]
    known_items = np.where(recommender.train_matrix[user_idx] > 0)[0]
    candidate_items = np.setdiff1d(np.arange(len(item_map)), known_items)

    with torch.no_grad():
        user_tensor = torch.tensor([user_idx] * len(candidate_items), dtype=torch.long).to(device)
        item_tensor = torch.tensor(candidate_items, dtype=torch.long).to(device)
        scores = ncf_model(user_tensor, item_tensor).cpu().numpy()

    top_indices = np.argsort(-scores)[:n]
    top_items = candidate_items[top_indices]

    return [
        {
            'item_id': reverse_item_map[item_idx],
            'score': round(float(score), 4),
            'title': f"{row['part_name']}: {row['subject_category'].replace('_', ' ').title()}",
            'part': row['part_name'],
            'part_id': row['part'],
            'subjects': get_subject_categories(row['tags']),
            'duration_minutes': 0,
            'type': 'bundle'
        }
        for item_idx, score in zip(top_items, scores[top_indices])
        if not bundle_info[bundle_info['bundle_id'] == reverse_item_map[item_idx]].empty
        for row in [bundle_info[bundle_info['bundle_id'] == reverse_item_map[item_idx]].iloc[0]]
    ]

def generate_hybrid_features(user_id, merged_df, sbert_embeddings, bundle_info,
                              recommender, item_map, reverse_item_map,
                              ncf_model, device, n=20):
    sbert_recs = get_sbert_recommendations(user_id, merged_df, bundle_info, sbert_embeddings, n)
    ncf_recs = get_ncf_recommendations(user_id, recommender, item_map, reverse_item_map, bundle_info, ncf_model, device, n)

    features = {}
    for rec in sbert_recs:
        features[rec['item_id']] = {
            'bundle_id': rec['item_id'],
            'sbert_score': rec['score'],
            'ncf_score': 0,
        }
    for rec in ncf_recs:
        if rec['item_id'] not in features:
            features[rec['item_id']] = {
                'bundle_id': rec['item_id'],
                'sbert_score': 0,
                'ncf_score': rec['score'],
            }
        else:
            features[rec['item_id']]['ncf_score'] = rec['score']

    for feat in features.values():
        row = bundle_info[bundle_info['bundle_id'] == feat['bundle_id']].iloc[0]
        feat['part_id'] = row['part']
        feat['success_rate'] = row['success_rate']

    return pd.DataFrame(features.values())

def get_hybrid_advanced_recommendations(user_id, merged_df, sbert_embeddings, bundle_info,
                                        recommender, item_map, reverse_item_map,
                                        ncf_model, device, meta_learner, n=10):
    features_df = generate_hybrid_features(
        user_id, merged_df, sbert_embeddings, bundle_info,
        recommender, item_map, reverse_item_map, ncf_model, device, n
    )
    if features_df.empty:
        return []
    X = features_df[['sbert_score', 'ncf_score', 'part_id', 'success_rate']].values
    features_df['hybrid_score'] = meta_learner.predict(X)
    top_recs = features_df.sort_values('hybrid_score', ascending=False).head(n)

    return [
        {
            'item_id': row['bundle_id'],
            'score': round(float(row['hybrid_score']), 4),
            'title': f"{bundle_row['part_name']}: {bundle_row['subject_category'].replace('_', ' ').title()}",
            'part': bundle_row['part_name'],
            'part_id': bundle_row['part'],
            'subjects': get_subject_categories(bundle_row['tags']),
            'duration_minutes': 0,
            'type': 'bundle'
        }
        for _, row in top_recs.iterrows()
        for bundle_row in [bundle_info[bundle_info['bundle_id'] == row['bundle_id']].iloc[0]]
    ]
