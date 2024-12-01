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

    # Convert the list of tensors to a 2D tensor.
    train_embeddings_tensor = torch.cat(train_embeddings, dim=0)
    embeddings_array = train_embeddings_tensor.cpu().numpy()

    # Trim both embeddings_array and is_false to match eval size
    num_eval_samples = len(eval_embeddings)
    embeddings_array = embeddings_array[:num_eval_samples]
    is_false = np.array(is_false[:num_eval_samples])

    logging.info(f"Using {num_eval_samples} samples for both training and validation")
    logging.info(f"Embeddings array shape: {embeddings_array.shape}")
    logging.info(f"is_false length: {len(is_false)}")
    logging.info(f"Unique classes in data: {np.unique(is_false)}")

    # For very small datasets (less than 10 samples), skip train-test split
    if len(embeddings_array) < 10:
        logging.warning(
            "Dataset too small for train-test split, using all data for training"
        )
        X_train = embeddings_array
        y_train = is_false
        X_test = embeddings_array  # Use same data for testing
        y_test = is_false
    else:
        # Original train-test split logic
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings_array,
            is_false,
            test_size=test_size,
            random_state=42,
            stratify=is_false,
        )  # Added stratification

    # Fit a logistic regression model.
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict deterministically and probabilistically and compute accuracy and auroc for all splits.
    X_eval = (
        torch.cat(eval_embeddings, dim=0).cpu().numpy()
    )  # pylint: disable=no-member,invalid-name
    y_eval = eval_is_false

    Xs = [X_train, X_test, X_eval]  # pylint: disable=invalid-name
    ys = [y_train, y_test, y_eval]  # pylint: disable=invalid-name
    suffixes = ["train_train", "train_test", "eval"]

    metrics, y_preds_proba = {}, {}

    for suffix, X, y_true in zip(suffixes, Xs, ys):  # pylint: disable=invalid-name

        # If suffix is eval, we fit a new model on the entire training data set
        # rather than just a split of the training data set.
        if suffix == "eval":
            model = LogisticRegression()
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

    # Return model predictions on the eval set.
    return y_preds_proba["eval"][:, 1]
