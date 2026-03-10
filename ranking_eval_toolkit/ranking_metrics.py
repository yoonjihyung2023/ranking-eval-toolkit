import math

def precision_at_k(y_true, y_pred, k=10):
    topk = y_pred[:k]
    if k == 0:
        return 0.0
    hits = sum(1 for item in topk if item in y_true)
    return hits / k

def recall_at_k(y_true, y_pred, k=10):
    if not y_true:
        return 0.0
    topk = y_pred[:k]
    hits = sum(1 for item in topk if item in y_true)
    return hits / len(y_true)

def dcg_at_k(y_true, y_pred, k=10):
    topk = y_pred[:k]
    score = 0.0
    for i, item in enumerate(topk):
        if item in y_true:
            score += 1.0 / math.log2(i + 2)
    return score

def ndcg_at_k(y_true, y_pred, k=10):
    ideal_list = list(y_true)
    ideal = dcg_at_k(y_true, ideal_list, min(k, len(ideal_list)))
    if ideal == 0:
        return 0.0
    return dcg_at_k(y_true, y_pred, k) / ideal

def hitrate_at_k(y_true, y_pred, k=10):
    topk = y_pred[:k]
    return 1.0 if any(item in y_true for item in topk) else 0.0
