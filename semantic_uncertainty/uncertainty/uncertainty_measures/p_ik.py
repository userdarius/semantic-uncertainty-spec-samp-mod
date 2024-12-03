"""Predict model correctness from linear classifier."""

import logging
import torch
import wandb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def get_p_ik(train_embeddings, is_false, eval_embeddings=None, eval_is_false=None):
    """Fit linear classifier to embeddings to predict model correctness."""
    logging.info("Accuracy of model on Task: %f.", 1 - torch.tensor(is_false).mean())

    # First, flatten each embedding separately
    flattened_train_embeddings = []
    for emb in train_embeddings:
        # Flatten the embedding while preserving sample identity
        flattened_emb = emb.reshape(
            1, -1
        )  # Reshape to (1, -1) to keep as single sample
        flattened_train_embeddings.append(flattened_emb)

    # Stack instead of concatenate
    train_embeddings_tensor = torch.vstack(flattened_train_embeddings)
    embeddings_array = train_embeddings_tensor.cpu().numpy()

    logging.info(
        f"Training embeddings shape after flattening: {embeddings_array.shape}"
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings_array, is_false, test_size=0.2, random_state=42
    )

    # Process eval embeddings the same way
    flattened_eval_embeddings = []
    for emb in eval_embeddings:
        flattened_eval = emb.reshape(1, -1)
        flattened_eval_embeddings.append(flattened_eval)

    X_eval = torch.vstack(flattened_eval_embeddings).cpu().numpy()
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
