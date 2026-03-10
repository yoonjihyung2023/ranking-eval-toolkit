from ranking_eval_toolkit import precision_at_k, recall_at_k, ndcg_at_k, hitrate_at_k

def test_ranking_metrics_run():
    y_true = {1, 3}
    y_pred = [1, 2, 3, 4, 5]

    assert 0.0 <= precision_at_k(y_true, y_pred, k=3) <= 1.0
    assert 0.0 <= recall_at_k(y_true, y_pred, k=3) <= 1.0
    assert 0.0 <= ndcg_at_k(y_true, y_pred, k=3) <= 1.0
    assert hitrate_at_k(y_true, y_pred, k=3) in [0.0, 1.0]
