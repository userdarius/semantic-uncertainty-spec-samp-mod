"""Compute semantic uncertainty measures."""

from collections import defaultdict
import logging
import os
import pickle
import numpy as np
import wandb

from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from uncertainty.uncertainty_measures.semantic_entropy import context_entails_response
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT35
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4Turbo
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama
from uncertainty.utils import utils

utils.setup_logger()


def main(args):
    user = os.environ["USER"]
    scratch_dir = os.getenv("SCRATCH_DIR", ".")
    wandb_dir = f"{scratch_dir}/{user}/uncertainty"
    slurm_jobid = os.getenv("SLURM_JOB_ID", None)
    project = "semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug"

    wandb.init(
        entity=args.entity,
        project=project,
        dir=wandb_dir,
        notes=f"slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}",
        config=args.__dict__,
    )

    # Load entailment model
    logging.info("Beginning loading for entailment model.")
    if args.entailment_model == "deberta":
        entailment_model = EntailmentDeberta()
    elif args.entailment_model == "gpt-4":
        entailment_model = EntailmentGPT4(
            args.entailment_cache_id, args.entailment_cache_only
        )
    elif args.entailment_model == "gpt-3.5":
        entailment_model = EntailmentGPT35(
            args.entailment_cache_id, args.entailment_cache_only
        )
    elif args.entailment_model == "gpt-4-turbo":
        entailment_model = EntailmentGPT4Turbo(
            args.entailment_cache_id, args.entailment_cache_only
        )
    elif "llama" in args.entailment_model.lower():
        entailment_model = EntailmentLlama(
            args.entailment_cache_id, args.entailment_cache_only, args.entailment_model
        )
    else:
        raise ValueError("Invalid entailment model")
    logging.info("Entailment model loading complete.")

    # Load validation generations
    with open(f"{wandb.run.dir}/validation_generations.pkl", "rb") as infile:
        validation_generations = pickle.load(infile)

    entropies = defaultdict(list)
    result_dict = {"semantic_ids": []}
    count = 0

    # Loop over datapoints and compute entropies
    for idx, tid in enumerate(validation_generations):
        example = validation_generations[tid]
        question = example["question"]
        context = example["context"]
        full_responses = example["responses"]
        most_likely_answer = example["most_likely_answer"]

        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            responses = [fr[0] for fr in full_responses[: args.use_num_generations]]
            log_liks = [r[1] for r in full_responses[: args.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]
            log_liks = [r[1] for r in full_responses]

        if args.compute_context_entails_response:
            entropies["context_entails_response"].append(
                context_entails_response(context, responses, entailment_model)
            )

        if args.condition_on_question and args.entailment_model == "deberta":
            responses = [f"{question} {r}" for r in responses]

        # Compute semantic ids
        semantic_ids = get_semantic_ids(
            responses,
            model=entailment_model,
            strict_entailment=args.strict_entailment,
            example=example,
        )

        result_dict["semantic_ids"].append(semantic_ids)

        # Compute entropy from frequencies of cluster assignments
        entropies["cluster_assignment_entropy"].append(
            cluster_assignment_entropy(semantic_ids)
        )

        # Length normalization of generation probabilities
        log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

        # Compute naive entropy
        entropies["regular_entropy"].append(predictive_entropy(log_liks_agg))

        # Compute semantic entropy
        log_likelihood_per_semantic_id = logsumexp_by_id(
            semantic_ids, log_liks_agg, agg="sum_normalized"
        )
        pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
        entropies["semantic_entropy"].append(pe)

        # Logging
        entropies_fmt = ", ".join([f"{i}:{j[-1]:.2f}" for i, j in entropies.items()])
        logging.info(80 * "#")
        logging.info("NEW ITEM %d at id=`%s`.", idx, tid)
        logging.info("Context:\n%s", example["context"])
        logging.info("Question:\n%s", question)
        logging.info("True Answers:\n%s", example["reference"])
        logging.info("Low Temperature Generation:\n%s", most_likely_answer["response"])
        logging.info("High Temp Generation:\n%s", [r[0] for r in full_responses])
        logging.info(
            "semantic_ids: %s\navg_token_log_likelihoods: %s\nentropies: %s",
            semantic_ids,
            log_liks_agg,
            entropies_fmt,
        )

        count += 1
        if count >= args.num_eval_samples:
            logging.info("Breaking out of main loop.")
            break

    result_dict["uncertainty_measures"] = entropies
    utils.save(result_dict, "uncertainty_measures.pkl")
    entailment_model.save_prediction_cache()


if __name__ == "__main__":
    parser = utils.get_parser(stages=["compute"])
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f"Unknown args: {unknown}")
    logging.info("Args: %s", args)
    main(args)
