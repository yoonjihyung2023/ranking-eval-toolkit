from ranking_eval_toolkit import auc_score, logloss_score

y_true = [1, 0, 1, 0]
y_score = [0.9, 0.2, 0.8, 0.1]

print("AUC:", auc_score(y_true, y_score))
print("LogLoss:", logloss_score(y_true, y_score))
