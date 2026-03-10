from sklearn.metrics import roc_auc_score, log_loss

def auc_score(y_true, y_score):
    return float(roc_auc_score(y_true, y_score))

def logloss_score(y_true, y_score):
    return float(log_loss(y_true, y_score))
