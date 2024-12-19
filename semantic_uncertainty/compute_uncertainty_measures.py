"""Compute semantic uncertainty measures."""

import pickle
from collections import defaultdict
import logging
import numpy as np
from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from uncertainty.utils import utils

utils.setup_logger()


def main(args):
    # Load entailment model
    logging.info("Beginning loading for entailment model.")
    entailment_model = EntailmentDeberta()
    logging.info("Entailment model loading complete.")

    # Load generations
    with open("validation_generations.pkl", "rb") as f:
        validation_generations = pickle.load(f)

    result_dict = {}
    result_dict["semantic_ids"] = []
    entropies = defaultdict(list)

    # Loop over datapoints and compute entropies
    for idx, tid in enumerate(validation_generations):
        example = validation_generations[tid]
        question = example["question"]
        responses = [fr[0] for fr in example["responses"]]
        log_liks = [r[1] for r in example["responses"]]

        # Compute semantic ids
        semantic_ids = get_semantic_ids(
            responses,
            model=entailment_model,
            strict_entailment=args.strict_entailment,
            example=example,
        )
        result_dict["semantic_ids"].append(semantic_ids)

        # Compute cluster assignment entropy
        entropies["cluster_assignment_entropy"].append(
            cluster_assignment_entropy(semantic_ids)
        )

        # Length normalization of generation probabilities
        log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

        # Compute regular entropy
        entropies["regular_entropy"].append(predictive_entropy(log_liks_agg))

        # Compute semantic entropy
        log_likelihood_per_semantic_id = logsumexp_by_id(
            semantic_ids, log_liks_agg, agg="sum_normalized"
        )
        semantic_entropy = predictive_entropy_rao(log_likelihood_per_semantic_id)
        entropies["semantic_entropy"].append(semantic_entropy)

        # Logging
        logging.info(f"Example {idx}:")
        logging.info(f"Question: {question}")
        logging.info(f"Responses: {responses}")
        logging.info(f"Semantic IDs: {semantic_ids}")
        logging.info(f'Entropies: {[f"{k}:{v[-1]:.2f}" for k,v in entropies.items()]}')

    result_dict["uncertainty_measures"] = entropies

    # Save results
    utils.save(result_dict, "uncertainty_measures.pkl")
    entailment_model.save_prediction_cache()


if __name__ == "__main__":
    parser = utils.get_parser(stages=["compute"])
    args = parser.parse_args()
    main(args)
