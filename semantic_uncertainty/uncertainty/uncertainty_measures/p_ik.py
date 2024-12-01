"""Predict model correctness from linear classifier."""

import logging
import torch
import wandb
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def get_p_ik(train_embeddings, is_false, eval_embeddings=None, eval_is_false=None):
    """Fit linear classifier to embeddings to predict model correctness."""
    logging.info("Accuracy of model on Task: %f.", 1 - torch.tensor(is_false).mean())

    # Process both sets first to determine sizes
    train_embeddings_tensor = torch.cat(train_embeddings, dim=0)
    eval_embeddings_tensor = torch.cat(eval_embeddings, dim=0)

    # Get the minimum size between training and evaluation
    num_train_samples = len(train_embeddings_tensor)
    num_eval_samples = len(eval_embeddings_tensor)
    num_samples = min(num_train_samples, num_eval_samples)

    logging.info(f"Total training samples: {num_train_samples}")
    logging.info(f"Total evaluation samples: {num_eval_samples}")
    logging.info(f"Using {num_samples} samples for analysis")

    # Trim both sets to the same size
    X_train = train_embeddings_tensor[:num_samples].cpu().numpy()
    y_train = np.array(is_false[:num_samples])
    X_eval = eval_embeddings_tensor[:num_samples].cpu().numpy()
    y_eval = np.array(eval_is_false[:num_samples])

    logging.info(f"Final training shape: {X_train.shape}")
    logging.info(f"Final evaluation shape: {X_eval.shape}")
    logging.info(f"Unique classes in training data: {np.unique(y_train)}")

    # Fit model and get predictions
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Compute metrics for all splits
    Xs = [X_train, X_test, X_eval]
    ys = [y_train, y_test, eval_is_false]
    suffixes = ["train_train", "train_test", "eval"]
    metrics, y_preds_proba = {}, {}

    for suffix, X, y_true in zip(suffixes, Xs, ys):
        if suffix == "eval":
            model = LogisticRegression()
            model.fit(X_train, y_train)  # Use trimmed training data
            convergence = {
                "n_iter": model.n_iter_[0],
                "converged": (model.n_iter_ < model.max_iter)[0],
            }
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        y_preds_proba[suffix] = y_pred_proba
        acc_p_ik_train = accuracy_score(y_true, y_pred)
        auroc_p_ik_train = roc_auc_score(y_true, y_pred_proba[:, 1])
        split_metrics = {
            f"acc_p_ik_{suffix}": acc_p_ik_train,
            f"auroc_p_ik_{suffix}": auroc_p_ik_train,
        }
        metrics.update(split_metrics)

    logging.info("Metrics for p_ik classifier: %s.", metrics)
    wandb.log({**metrics, **convergence})
    return y_preds_proba["eval"][:, 1]
