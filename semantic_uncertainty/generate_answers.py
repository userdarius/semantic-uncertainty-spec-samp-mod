"""Generate answers for semantic uncertainty analysis."""

import gc
import os
import logging
import random
from tqdm import tqdm

import numpy as np
import torch
import wandb

from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils
from uncertainty.models.cot_model import ChainOfThoughtHuggingfaceModel

utils.setup_logger()


def process_model_output(output, model):
    """Ensure consistent output format across model types."""
    if isinstance(model, ChainOfThoughtHuggingfaceModel):
        if isinstance(output, tuple) and len(output) == 3:
            return output
        else:
            return output, [], None
    return output


def main(args):
    # Setup wandb
    user = os.environ["USER"]
    slurm_jobid = os.getenv("SLURM_JOB_ID", None)
    scratch_dir = os.getenv("SCRATCH_DIR", ".")
    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")

    wandb.init(
        entity=args.entity,
        project=(
            "semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug"
        ),
        dir=f"{scratch_dir}/{user}/uncertainty",
        config=args,
        notes=f"slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}",
    )
    logging.info("Finished wandb init.")

    # Load dataset
    train_dataset, validation_dataset = load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed
    )

    # Get indices of answerable questions and construct prompt
    answerable_indices, _ = utils.split_dataset(train_dataset)
    prompt_indices = random.sample(answerable_indices, args.num_few_shot)

    # Create Few-Shot prompt
    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset,
        prompt_indices,
        BRIEF,
        args.brief_always if args.enable_brief else True,
        make_prompt,
    )
    logging.info("Prompt is: %s", prompt)

    # Initialize model
    model = utils.init_model(args)

    # Start answer generation for validation set
    logging.info(80 * "=")
    logging.info("Generating answers for validation set:")
    logging.info(80 * "=")

    accuracies = []
    generations = {}
    metric = utils.get_metric(args.metric)

    # Get indices for validation set
    indices = random.sample(
        range(len(validation_dataset)), min(args.num_samples, len(validation_dataset))
    )

    if args.num_samples > len(validation_dataset):
        logging.warning(
            "Not enough samples in dataset. Using all %d samples.",
            len(validation_dataset),
        )

    # Generate answers
    for it, index in enumerate(tqdm(indices)):
        if (it + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        # Get example
        example = validation_dataset[index]
        question, context = example["question"], example["context"]
        generations[example["id"]] = {
            "question": question,
            "context": context,
            "reference": utils.get_reference(example),
        }

        # Create prompt for this example
        current_input = make_prompt(
            context, question, None, BRIEF, args.brief_always and args.enable_brief
        )
        local_prompt = prompt + current_input
        logging.info("Current input: ".ljust(15) + current_input)

        # Generate answers
        full_responses = []
        num_generations = args.num_generations + 1

        for i in range(num_generations):
            # Low temperature for first generation
            temperature = 0.1 if i == 0 else args.temperature

            try:
                output = model.predict(
                    local_prompt,
                    temperature,
                    use_branching=args.use_chain_of_thought,
                    num_branches=(
                        args.num_branches if args.use_chain_of_thought else None
                    ),
                )
                predicted_answer, token_log_likelihoods, embedding = (
                    process_model_output(output, model)
                )

                if embedding is not None:
                    embedding = embedding.cpu()

            except Exception as e:
                logging.error(f"Error in generation: {str(e)}")
                predicted_answer = ""
                token_log_likelihoods = []
                embedding = None

            # Compute accuracy for answerable questions
            if example["answers"]["text"]:
                acc = metric(predicted_answer, example, model)
            else:
                acc = 0.0

            # Store first (low temperature) generation
            if i == 0:
                logging.info("Iteration " + str(it) + ":  " + 80 * "#")
                if args.use_context:
                    logging.info("context: ".ljust(15) + str(context))
                logging.info("question: ".ljust(15) + question)
                logging.info("low-t prediction: ".ljust(15) + predicted_answer)
                logging.info(
                    "correct answer: ".ljust(15) + str(example["answers"]["text"])
                )
                logging.info("accuracy: ".ljust(15) + str(acc))

                accuracies.append(acc)
                generations[example["id"]]["most_likely_answer"] = {
                    "response": predicted_answer,
                    "token_log_likelihoods": token_log_likelihoods,
                    "embedding": embedding,
                    "accuracy": acc,
                }
            else:
                logging.info(
                    "high-t prediction ".ljust(15) + str(i) + " : " + predicted_answer
                )
                full_responses.append(
                    (predicted_answer, token_log_likelihoods, embedding, acc)
                )

        generations[example["id"]]["responses"] = full_responses

    # Save generations
    utils.save(generations, "validation_generations.pkl")

    # Log overall accuracy
    accuracy = np.mean(accuracies)
    logging.info(f"Overall validation accuracy: {accuracy}")
    wandb.log({"validation_accuracy": accuracy})

    logging.info("Generation complete.")
    del model


if __name__ == "__main__":
    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()
    logging.info("Starting generation with args: %s", args)

    if unknown:
        raise ValueError(f"Unknown args: {unknown}")

    main(args)
