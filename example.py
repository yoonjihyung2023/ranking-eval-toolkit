from ranking_eval.metrics import auc_score, logloss, precision_at_k, recall_at_k, ndcg_at_k

y_true = [0, 1, 0, 1, 1]
y_score = [0.1, 0.9, 0.2, 0.8, 0.7]

print("AUC:", round(auc_score(y_true, y_score), 6))
print("LogLoss:", round(logloss(y_true, y_score), 6))
print("Precision@3:", round(precision_at_k(y_true, y_score, 3), 6))
print("Recall@3:", round(recall_at_k(y_true, y_score, 3), 6))
print("NDCG@3:", round(ndcg_at_k(y_true, y_score, 3), 6))
