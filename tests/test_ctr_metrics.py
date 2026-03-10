from ranking_eval_toolkit import auc_score, logloss_score

def test_auc_score_runs():
    y_true = [1, 0, 1, 0]
    y_score = [0.9, 0.1, 0.8, 0.2]
    assert 0.0 <= auc_score(y_true, y_score) <= 1.0

def test_logloss_score_runs():
    y_true = [1, 0, 1, 0]
    y_score = [0.9, 0.1, 0.8, 0.2]
    assert logloss_score(y_true, y_score) >= 0.0
