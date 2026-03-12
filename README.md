ranking-eval-toolkit

Reusable ranking metrics toolkit for CTR/RecSys experiments
Metrics: AUC, LogLoss, Precision@K, Recall@K, NDCG@K
Focus: lightweight, testable, reproducible offline evaluation

One-line

A small Python toolkit for reusable offline ranking evaluation in CTR and recommendation experiments.

Why this repo matters

Reusable metric code for ranking / recommendation experiments

Lightweight and testable utilities for offline evaluation

Covers both classification-style and ranking-style metrics

Useful as a companion repo for CTR / RecSys pipelines

Metrics

AUC

LogLoss

Precision@K

Recall@K

NDCG@K

Intended use

CTR prediction experiments

recommendation ranking experiments

offline model comparison

evaluation utility extraction from larger ML projects

Example direction

from ranking_eval_toolkit import auc, logloss, precision_at_k, recall_at_k, ndcg_at_k

Why it fits this portfolio

This repository widens the story beyond one benchmark:

ctr-seqrec-avazu shows benchmark proof

ctr-api shows serving

ranking-eval-toolkit shows reusable evaluation thinking
