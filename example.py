from ranking_eval.metrics import auc_score, logloss

y_true = [0, 1, 0, 1]
y_score = [0.1, 0.9, 0.2, 0.8]

print("AUC:", round(auc_score(y_true, y_score), 6))
print("LogLoss:", round(logloss(y_true, y_score), 6))
