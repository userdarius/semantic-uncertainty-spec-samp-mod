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
    logging.info("Starting semantic uncertainty computation")
    logging.info(f"Arguments: {args}")

    # Load entailment model
    logging.info("Beginning loading for entailment model.")
    entailment_model = EntailmentDeberta()
    logging.info("Entailment model loading complete.")

    # Load generations
    logging.info("Loading generated responses from validation_generations.pkl")
    try:
        with open("validation_generations.pkl", "rb") as f:
            validation_generations = pickle.load(f)
        logging.info(f"Loaded {len(validation_generations)} examples")
    except Exception as e:
        logging.error(f"Error loading generations: {str(e)}")
        return

    result_dict = {}
    result_dict["semantic_ids"] = []
    entropies = defaultdict(list)

    # Loop over datapoints and compute entropies
    logging.info("Starting entropy computations")
    total_examples = len(validation_generations)

    for idx, tid in enumerate(validation_generations):
        logging.info(f"\nProcessing example {idx+1}/{total_examples}")
        logging.info(f"Example ID: {tid}")

        example = validation_generations[tid]
        question = example["question"]
        responses = [fr[0] for fr in example["responses"]]
        log_liks = [r[1] for r in example["responses"]]

        logging.info(f"Question: {question}")
        logging.info(f"Number of responses: {len(responses)}")

        # Compute semantic ids
        logging.info("Computing semantic IDs...")
        try:
            semantic_ids = get_semantic_ids(
                responses,
                model=entailment_model,
                strict_entailment=args.strict_entailment,
                example=example,
            )
            result_dict["semantic_ids"].append(semantic_ids)
            logging.info(f"Semantic IDs computed: {semantic_ids}")
        except Exception as e:
            logging.error(f"Error computing semantic IDs: {str(e)}")
            continue

        # Compute entropies
        try:
            # Cluster assignment entropy
            cluster_entropy = cluster_assignment_entropy(semantic_ids)
            entropies["cluster_assignment_entropy"].append(cluster_entropy)
            logging.info(f"Cluster assignment entropy: {cluster_entropy:.4f}")

            # Length normalization of generation probabilities
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

            # Regular entropy
            reg_entropy = predictive_entropy(log_liks_agg)
            entropies["regular_entropy"].append(reg_entropy)
            logging.info(f"Regular entropy: {reg_entropy:.4f}")

            # Semantic entropy
            log_likelihood_per_semantic_id = logsumexp_by_id(
                semantic_ids, log_liks_agg, agg="sum_normalized"
            )
            semantic_entropy = predictive_entropy_rao(log_likelihood_per_semantic_id)
            entropies["semantic_entropy"].append(semantic_entropy)
            logging.info(f"Semantic entropy: {semantic_entropy:.4f}")

        except Exception as e:
            logging.error(f"Error computing entropies: {str(e)}")
            continue

        # Log response details
        for i, response in enumerate(responses):
            logging.info(
                f"Response {i+1}: {response[:100]}..."
                if len(response) > 100
                else f"Response {i+1}: {response}"
            )

    # Log final statistics
    logging.info("\nFinal Statistics:")
    for entropy_type, values in entropies.items():
        mean_entropy = np.mean(values)
        std_entropy = np.std(values)
        logging.info(f"{entropy_type}:")
        logging.info(f"  Mean: {mean_entropy:.4f}")
        logging.info(f"  Std:  {std_entropy:.4f}")

    result_dict["uncertainty_measures"] = entropies

    # Save results
    logging.info("Saving results to uncertainty_measures.pkl")
    try:
        utils.save(result_dict, "uncertainty_measures.pkl")
        logging.info("Results saved successfully")
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")

    logging.info("Saving entailment model prediction cache")
    try:
        entailment_model.save_prediction_cache()
        logging.info("Prediction cache saved successfully")
    except Exception as e:
        logging.error(f"Error saving prediction cache: {str(e)}")

    logging.info("Semantic uncertainty computation completed")


if __name__ == "__main__":
    parser = utils.get_parser(stages=["compute"])
    args = parser.parse_args()
    main(args)
