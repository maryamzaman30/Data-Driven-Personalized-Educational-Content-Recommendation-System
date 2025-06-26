import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, roc_auc_score

# Precision at K
def precision_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    if not recommended_k:
        return 0.0
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(recommended_k)

# Recall at K
def recall_at_k(recommended, relevant, k=10):
    if not relevant:
        return 0.0
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant)

# RMSE
def rmse_score(true_vals, pred_vals):
    return np.sqrt(mean_squared_error(true_vals, pred_vals))

# AUC
def auc_score(y_true, y_scores):
    try:
        return roc_auc_score(y_true, y_scores)
    except:
        return 0.0

# Evaluation function
def evaluate_user_model(user_ids, recommend_fn, ground_truth_fn, score_fn=None, true_score_fn=None, k_vals=(5, 10, 20)):
    results = {k: {'precision': [], 'recall': []} for k in k_vals}
    all_true, all_pred, all_scores = [], [], []

    for user_id in user_ids:
        recommended = recommend_fn(user_id)
        relevant = ground_truth_fn(user_id)

        if score_fn and true_score_fn:
            all_true += true_score_fn(user_id)
            all_pred += score_fn(user_id)
            all_scores += score_fn(user_id)  # Assumes scores are probability-like

        for k in k_vals:
            p = precision_at_k(recommended, relevant, k)
            r = recall_at_k(recommended, relevant, k)
            results[k]['precision'].append(p)
            results[k]['recall'].append(r)

    aggregated = {
        k: {
            'precision': np.mean(v['precision']),
            'recall': np.mean(v['recall']),
        } for k, v in results.items()
    }

    if all_true and all_pred:
        aggregated['rmse'] = rmse_score(all_true, all_pred)
        aggregated['auc'] = auc_score(all_true, all_scores)

    return aggregated

# TF-IDF parameter tuning
def tune_tfidf_params(param_grid, vectorizer_class, X_raw, y_true_fn, recommend_fn_builder,
                      ground_truth_fn, score_fn=None, true_score_fn=None, k=10, output_path=None):
    results = []
    for params in param_grid:
        print(f"Testing TF-IDF params: {params}")
        vectorizer = vectorizer_class(**params)
        try:
            tfidf_matrix = vectorizer.fit_transform(X_raw)
            recommender = recommend_fn_builder(tfidf_matrix, vectorizer)
            metrics = evaluate_user_model(
                user_ids=y_true_fn(),
                recommend_fn=recommender,
                ground_truth_fn=ground_truth_fn,
                score_fn=score_fn,
                true_score_fn=true_score_fn,
                k_vals=[k]
            )
            result = {
                'params': str(params),
                'precision': metrics[k]['precision'],
                'recall': metrics[k]['recall'],
                'rmse': metrics.get('rmse', None),
                'auc': metrics.get('auc', None)
            }
            print(result)
            results.append(result)
        except Exception as e:
            print(f"Failed with error: {e}")

    df = pd.DataFrame(results)
    if output_path:
        df.to_csv(output_path, index=False)

    # Plot precision scores
    if not df.empty:
        df_sorted = df_sorted[["params", "precision"]].set_index("params").plot(kind="bar", figsize=(10, 5), legend=False, title="Precision@K by TF-IDF Params")
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_sorted, x="params", y="precision")
        plt.title("TF-IDF Parameter Tuning (Precision@K)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    return df
