from ranking_eval.metrics import auc_score, logloss, precision_at_k, recall_at_k


def test_auc_score():
    y_true = [0, 1, 0, 1]
    y_score = [0.1, 0.9, 0.2, 0.8]
    assert round(auc_score(y_true, y_score), 6) == 1.0


def test_logloss():
    y_true = [0, 1, 0, 1]
    y_score = [0.1, 0.9, 0.2, 0.8]
    assert round(logloss(y_true, y_score), 6) > 0


def test_precision_at_k():
    y_true = [0, 1, 0, 1, 1]
    y_score = [0.1, 0.9, 0.2, 0.8, 0.7]
    assert round(precision_at_k(y_true, y_score, 3), 6) == 1.0


def test_recall_at_k():
    y_true = [0, 1, 0, 1, 1]
    y_score = [0.1, 0.9, 0.2, 0.8, 0.7]
    assert round(recall_at_k(y_true, y_score, 3), 6) == 1.0
