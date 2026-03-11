import math


def auc_score(y_true, y_score):
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC is undefined when only one class is present.")

    rank_sum = 0
    for i, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            rank_sum += i

    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def logloss(y_true, y_prob, eps=1e-15):
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")

    total = 0.0
    for y, p in zip(y_true, y_prob):
        p = min(max(p, eps), 1 - eps)
        total += y * math.log(p) + (1 - y) * math.log(1 - p)

    return -total / len(y_true)


def precision_at_k(y_true, y_score, k):
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    ranked = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    top_k = ranked[:k]
    hits = sum(label for _, label in top_k)
    return hits / len(top_k)


def recall_at_k(y_true, y_score, k):
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    total_relevant = sum(y_true)
    if total_relevant == 0:
        raise ValueError("Recall is undefined when there are no positive labels.")

    ranked = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    top_k = ranked[:k]
    hits = sum(label for _, label in top_k)
    return hits / total_relevant
