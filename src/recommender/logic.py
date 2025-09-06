# =========================================================
# File: src/recommender/logic.py
# Description: Core recommendation logic combining
#   SBERT-based content similarity, NCF collaborative
#   filtering, SVD collaborative filtering, and a
#   meta-learned hybrid model.
# =========================================================

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.mappings import get_subject_categories

# =========================================================
# 1. SBERT Content-Based Recommendations
# =========================================================

def get_sbert_recommendations(user_id, merged_df, bundle_info, sbert_embeddings, n=10):
    # Get user's interaction history
    user_history = merged_df[merged_df['user_id'] == user_id]
    seen = set(user_history['bundle_id'].unique())

    # Map bundle IDs to their indices
    indices = pd.Series(bundle_info.index, index=bundle_info['bundle_id'])

    # Compute cosine similarity matrix for SBERT embeddings
    sim_matrix = cosine_similarity(sbert_embeddings)
    scores = {}

    # Accumulate similarity scores for unseen bundles
    for b in seen:
        if b not in indices:
            continue
        idx = indices[b]
        for i, score in enumerate(sim_matrix[idx]):
            b_id = bundle_info.iloc[i]['bundle_id']
            if b_id not in seen and score >= 0.1:
                scores[b_id] = scores.get(b_id, 0) + score

    # Return empty list if no scores found
    if not scores:
        return []

    # Normalize scores and select top-N bundles
    max_score = max(scores.values())
    top_bundles = sorted(((k, v / max_score) for k, v in scores.items()), key=lambda x: x[1], reverse=True)[:n]

    results = []
    # Build recommendation result with metadata
    for b, score in top_bundles:
        row_match = bundle_info[bundle_info['bundle_id'] == b]
        if row_match.empty:
            continue
        row = row_match.iloc[0]
        results.append({
            'item_id': str(row['bundle_id']),
            'score': round(float(score), 4),
            'title': f"{row['part_name']}: {row['subject_category'].replace('_', ' ').title()}",
            'part': row['part_name'],
            'part_id': int(row['part']),
            'subjects': get_subject_categories(row['tags']),
            'duration_minutes': float(row.get('video_minutes', row.get('duration_minutes', 0))),
            'type': 'bundle'
        })
    return results

# =========================================================
# 2. NCF Collaborative Filtering Recommendations
# =========================================================

def get_ncf_recommendations(user_id, recommender, item_map, reverse_item_map, bundle_info, ncf_model, device, n=10):
    # Return empty list if user not in training data
    if user_id not in recommender.user_map:
        return []

    # Get internal user index and known items
    user_idx = recommender.user_map[user_id]
    known_items = np.where(recommender.train_matrix[user_idx] > 0)[0]

    # Identify candidate items not yet interacted with
    candidate_items = np.setdiff1d(np.arange(len(item_map)), known_items)

    # Predict scores for candidate items using NCF model
    with torch.no_grad():
        user_tensor = torch.tensor([user_idx] * len(candidate_items), dtype=torch.long).to(device)
        item_tensor = torch.tensor(candidate_items, dtype=torch.long).to(device)
        scores = ncf_model(user_tensor, item_tensor).cpu().numpy()

    # Select top-N items based on predicted scores
    top_indices = np.argsort(-scores)[:n]
    top_items = candidate_items[top_indices]

    results = []
    # Build recommendation result with metadata
    for item_idx, score in zip(top_items, scores[top_indices]):
        b_id = reverse_item_map[item_idx]
        row_match = bundle_info[bundle_info['bundle_id'] == b_id]
        if row_match.empty:
            continue
        row = row_match.iloc[0]
        results.append({
            'item_id': b_id,
            'score': round(float(score), 4),
            'title': f"{row['part_name']}: {row['subject_category'].replace('_', ' ').title()}",
            'part': row['part_name'],
            'part_id': int(row['part']),
            'subjects': get_subject_categories(row['tags']),
            'duration_minutes': float(row.get('video_minutes', row.get('duration_minutes', 0))),
            'type': 'bundle'
        })
    return results

# =========================================================
# 3. Hybrid Feature Generation for Meta-Learning
# =========================================================

def generate_hybrid_features(user_id, merged_df, sbert_embeddings, bundle_info,
                              recommender, item_map, reverse_item_map,
                              ncf_model, device, n=20):
    # Get top-N recommendations from SBERT and NCF models
    sbert_recs = get_sbert_recommendations(user_id, merged_df, bundle_info, sbert_embeddings, n)
    ncf_recs = get_ncf_recommendations(user_id, recommender, item_map, reverse_item_map, bundle_info, ncf_model, device, n)

    features = {}
    # Initialize feature dict with SBERT scores
    for rec in sbert_recs:
        features[rec['item_id']] = {
            'bundle_id': rec['item_id'],
            'sbert_score': rec['score'],
            'ncf_score': 0,
        }
    # Add or update feature dict with NCF scores
    for rec in ncf_recs:
        if rec['item_id'] not in features:
            features[rec['item_id']] = {
                'bundle_id': rec['item_id'],
                'sbert_score': 0,
                'ncf_score': rec['score'],
            }
        else:
            features[rec['item_id']]['ncf_score'] = rec['score']

    # Enrich features with metadata from bundle_info
    for feat in features.values():
        row_match = bundle_info[bundle_info['bundle_id'] == feat['bundle_id']]
        if row_match.empty:
            continue
        row = row_match.iloc[0]
        feat['part_id'] = int(row.get('part', 0))
        feat['success_rate'] = float(row.get('success_rate', 0.0))

    # Return features as a DataFrame
    return pd.DataFrame(features.values())

# =========================================================
# 4. Hybrid Recommendations with Meta-Learner
# =========================================================

def get_hybrid_advanced_recommendations(user_id, merged_df, sbert_embeddings, bundle_info,
                                        recommender, item_map, reverse_item_map,
                                        ncf_model, device, meta_learner, n=10):
    # Generate combined features from SBERT and NCF models
    features_df = generate_hybrid_features(
        user_id, merged_df, sbert_embeddings, bundle_info,
        recommender, item_map, reverse_item_map, ncf_model, device, n
    )
    # Return empty list if no features available
    if features_df.empty:
        return []

    # Prepare input for meta-learner and predict hybrid scores
    X = features_df[['sbert_score', 'ncf_score', 'part_id', 'success_rate']].values
    features_df['hybrid_score'] = meta_learner.predict(X)

    # Select top-N recommendations based on hybrid score
    top_recs = features_df.sort_values('hybrid_score', ascending=False).head(n)

    results = []
    # Build final recommendation list with metadata
    for _, row in top_recs.iterrows():
        row_match = bundle_info[bundle_info['bundle_id'] == row['bundle_id']]
        if row_match.empty:
            continue
        bundle_row = row_match.iloc[0]
        results.append({
            'item_id': row['bundle_id'],
            'score': round(float(row['hybrid_score']), 4),
            'title': f"{bundle_row['part_name']}: {bundle_row['subject_category'].replace('_', ' ').title()}",
            'part': bundle_row['part_name'],
            'part_id': int(row['part_id']),
            'subjects': get_subject_categories(bundle_row['tags']),
            'duration_minutes': float(bundle_row.get('video_minutes', bundle_row.get('duration_minutes', 0))),
            'type': 'bundle'
        })
    return results

# =========================================================
# 5. Pure SVD Collaborative Filtering Recommendations
# =========================================================

def get_svd_collaborative_recommendations(user_id, recommender, item_map, reverse_item_map, bundle_info, n=10):
    """
    Generate pure collaborative filtering recommendations using SVDHybridRecommender.
    Assumes recommender was trained WITHOUT content similarity or with combine_weight=1.0.
    """
    # Return empty list if user not found in training data
    if user_id not in recommender.user_map:
        return []

    # Get internal user index and items they've interacted with
    user_idx = recommender.user_map[user_id]
    known_items = np.where(recommender.train_matrix[user_idx] > 0)[0]

    # Get top-N unseen item recommendations from SVD model
    recs = recommender.get_recommendations(user_idx, n=n, exclude_seen=True, known_items=known_items)

    results = []
    # Build recommendation list with metadata
    for item_idx, score in recs:
        b_id = reverse_item_map[item_idx]
        row_match = bundle_info[bundle_info['bundle_id'] == b_id]
        if row_match.empty:
            continue
        row = row_match.iloc[0]
        results.append({
            'item_id': b_id,
            'score': round(float(score), 4),
            'title': f"{row['part_name']}: {row['subject_category'].replace('_', ' ').title()}",
            'part': row['part_name'],
            'part_id': int(row['part']),
            'subjects': get_subject_categories(row['tags']),
            'duration_minutes': float(row.get('video_minutes', row.get('duration_minutes', 0))),
            'type': 'bundle'
        })
    return results