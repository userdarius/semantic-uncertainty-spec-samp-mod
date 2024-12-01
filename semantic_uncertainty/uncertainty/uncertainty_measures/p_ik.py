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
    """Fit linear classifier to embeddings to predict model correctness.

    Args:
        train_embeddings: List of embedding tensors
        is_false: List of 0.0/1.0 values indicating correctness
        eval_embeddings: Optional list of evaluation embedding tensors
        eval_is_false: Optional list of evaluation 0.0/1.0 values
    """
    # Convert list of tensors to numpy array
    train_embeddings_tensor = torch.cat(train_embeddings, dim=0)
    embeddings_array = train_embeddings_tensor.cpu().numpy()

    # Convert is_false to numpy array if it isn't already
    is_false = np.array(is_false)

    # Log shapes and stats
    logging.info("Total training samples: %d", embeddings_array.shape[0])
    logging.info("Final training shape: %s", str(embeddings_array.shape))
    logging.info("Training labels shape: %s", str(is_false.shape))
    logging.info("Accuracy of model on Task: %f.", 1 - np.mean(is_false))

    # Convert evaluation embeddings if provided
    if eval_embeddings is not None:
        X_eval = torch.cat(eval_embeddings, dim=0).cpu().numpy()
        logging.info("Final evaluation shape: %s", str(X_eval.shape))
        if eval_is_false is not None:
            eval_is_false = np.array(eval_is_false)
            logging.info("Eval labels shape: %s", str(eval_is_false.shape))

    # Split training data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings_array, is_false, test_size=0.2, random_state=42
    )

    # Initialize and fit model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Prepare data for evaluation
    if eval_embeddings is not None:
        X_eval = torch.cat(eval_embeddings, dim=0).cpu().numpy()
        y_eval = eval_is_false
    else:
        X_eval = X_test
        y_eval = y_test

    Xs = [X_train, X_test, X_eval]
    ys = [y_train, y_test, y_eval]
    suffixes = ["train_train", "train_test", "eval"]
    metrics, y_preds_proba = {}, {}

    # Evaluate on all splits
    for suffix, X, y_true in zip(suffixes, Xs, ys):
        if suffix == "eval" and eval_embeddings is not None:
            model = LogisticRegression(max_iter=1000)
            model.fit(embeddings_array, is_false)
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
