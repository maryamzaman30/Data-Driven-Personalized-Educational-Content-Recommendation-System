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
def tune_tfidf_params(param_grid, vectorizer_class, X_raw, y_true_fn, recommend_fn_builder, ground_truth_fn, k=10, output_path=None):
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
                k_vals=[k]
            )
            result = {
                'params': str(params),
                'precision': metrics[k]['precision'],
                'recall': metrics[k]['recall']
            }
            print(result)
            results.append(result)
        except Exception as e:
            print(f"Failed with error: {e}")

    df = pd.DataFrame(results)
    if output_path:
        df.to_csv(output_path, index=False)

    # Plot tuning results
    if not df.empty:
        df_sorted = df.sort_values("precision", ascending=False)
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_sorted, x="params", y="precision")
        plt.title("TF-IDF Parameter Tuning (Precision@K)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    return df

def tune_hybrid_weights(
    weights, model_class, train_matrix, content_sim, test_matrix,
    precision_k=10, output_path=None,
    user_map=None, item_map=None, reverse_item_map=None ):

    records = []
    for w in weights:
        print(f"Testing hybrid weight: {w}")
        model = model_class(n_factors=20, combine_weight=w)
        model.train_matrix = train_matrix
        model.initialize(train_matrix, user_map, item_map, reverse_item_map)
        model.fit(train_matrix, content_sim)
        
        precision, recall = calculate_precision_recall_at_k(model, test_matrix, k=precision_k)
        rmse = calculate_rmse(model, test_matrix)
        record = {
            'weight': w,
            'precision@10': precision,
            'recall@10': recall,
            'rmse': rmse
        }
        print(record)
        records.append(record)

    df = pd.DataFrame(records)
    if output_path:
        df.to_csv(output_path, index=False)

    # Plot results
    if not df.empty:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(df['weight'], df['precision@10'], marker='o', label='Precision@10')
        plt.plot(df['weight'], df['recall@10'], marker='o', label='Recall@10')
        plt.plot(df['weight'], df['rmse'], marker='o', label='RMSE')
        plt.title("Hybrid Blending Weight Tuning")
        plt.xlabel("Weight (α)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return df

# Support for hybrid tuning

def calculate_precision_recall_at_k(model, test_matrix, k=10):
    precision_sum = 0.0
    recall_sum = 0.0
    user_count = 0

    for u in range(test_matrix.shape[0]):
        test_items = np.where(test_matrix[u] > 0)[0]
        if len(test_items) == 0:
            continue

        known_items = np.where(model.train_matrix[u] > 0)[0]
        recs = model.get_recommendations(u, n=k, exclude_seen=True, known_items=known_items)
        if not recs:
            continue

        rec_items = [item for item, _ in recs]
        hits = len(set(rec_items) & set(test_items))

        precision_sum += hits / min(k, len(rec_items))
        recall_sum += hits / len(test_items)
        user_count += 1

    avg_precision = precision_sum / user_count if user_count > 0 else 0.0
    avg_recall = recall_sum / user_count if user_count > 0 else 0.0
    return avg_precision, avg_recall

def calculate_rmse(model, test_matrix):
    non_zero_indices = np.where(test_matrix > 0)
    y_true = []
    y_pred = []
    for u, i in zip(non_zero_indices[0], non_zero_indices[1]):
        y_true.append(test_matrix[u, i])
        y_pred.append(model.predict_rating(u, i))
    return np.sqrt(mean_squared_error(y_true, y_pred))
