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

    # Process evaluation embeddings first
    eval_embeddings_tensor = torch.cat(eval_embeddings, dim=0)
    X_eval = eval_embeddings_tensor.cpu().numpy()
    num_eval_samples = len(X_eval)
    logging.info(f"Evaluation samples: {num_eval_samples}")

    # Process and trim training embeddings
    train_embeddings_tensor = torch.cat(train_embeddings[:num_eval_samples], dim=0)
    X_train = train_embeddings_tensor.cpu().numpy()
    y_train = np.array(is_false[:num_eval_samples])

    logging.info(f"Training samples after trimming: {len(X_train)}")
    logging.info(f"Training embeddings shape: {X_train.shape}")
    logging.info(f"Evaluation embeddings shape: {X_eval.shape}")
    logging.info(f"Unique classes in training data: {np.unique(y_train)}")

    # For very small datasets (less than 10 samples), skip train-test split
    if len(X_train) < 10:
        logging.warning(
            "Dataset too small for train-test split, using all data for training"
        )
        X_test = X_train.copy()  # Use same data for testing
        y_test = y_train.copy()
    else:
        # Original train-test split logic
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=test_size, random_state=42, stratify=y_train
        )

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
