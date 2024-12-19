"""Sample answers from LLMs for semantic uncertainty analysis."""

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
    # Setup run
    random.seed(args.random_seed)
    user = os.environ["USER"]
    slurm_jobid = os.getenv("SLURM_JOB_ID", None)
    scratch_dir = os.getenv("SCRATCH_DIR", ".")

    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")

    # Initialize wandb
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

    # Get accuracy metric
    metric = utils.get_metric(args.metric)

    # Load dataset
    train_dataset, validation_dataset = load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed
    )

    # Get indices for few-shot prompts
    answerable_indices, _ = utils.split_dataset(train_dataset)
    prompt_indices = random.sample(answerable_indices, args.num_few_shot)

    # Create few-shot prompt
    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    brief_always = args.brief_always if args.enable_brief else True
    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, brief_always, make_prompt
    )
    logging.info("Prompt is: %s", prompt)

    # Initialize model
    model = utils.init_model(args)

    # Start answer generation
    logging.info(80 * "=")
    logging.info("Generating answers for semantic uncertainty analysis:")
    logging.info(80 * "=")

    # Process validation dataset
    accuracies = []
    generations = {}

    # Sample subset of validation dataset
    possible_indices = range(len(validation_dataset))
    indices = random.sample(
        possible_indices, min(args.num_samples, len(validation_dataset))
    )

    if args.num_samples > len(validation_dataset):
        logging.warning(
            "Not enough samples in dataset. Using all %d samples.",
            len(validation_dataset),
        )

    # Generate answers for each example
    for it, index in enumerate(tqdm(indices)):
        if (it + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        # Get example
        example = validation_dataset[index]
        question, context = example["question"], example["context"]
        generations[example["id"]] = {"question": question, "context": context}
        correct_answer = example["answers"]["text"]

        # Construct prompt
        current_input = make_prompt(context, question, None, BRIEF, brief_always)
        local_prompt = prompt + current_input
        logging.info("Current input: ".ljust(15) + current_input)

        full_responses = []

        # Generate answers with different temperatures
        for i in range(args.num_generations + 1):
            # First generation always uses low temperature
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
            acc = metric(predicted_answer, example, model) if correct_answer else 0.0

            # Handle first (low temperature) generation
            if i == 0:
                logging.info("Iteration " + str(it) + ":  " + 80 * "#")
                if args.use_context:
                    logging.info("context: ".ljust(15) + str(context))
                logging.info("question: ".ljust(15) + question)
                logging.info("low-t prediction: ".ljust(15) + predicted_answer)
                logging.info("correct answer: ".ljust(15) + str(correct_answer))
                logging.info("accuracy: ".ljust(15) + str(acc))

                accuracies.append(acc)
                most_likely_answer_dict = {
                    "response": predicted_answer,
                    "token_log_likelihoods": token_log_likelihoods,
                    "embedding": embedding,
                    "accuracy": acc,
                }
                generations[example["id"]].update(
                    {
                        "most_likely_answer": most_likely_answer_dict,
                        "reference": utils.get_reference(example),
                    }
                )
            else:
                logging.info(
                    "high-t prediction ".ljust(15) + str(i) + " : " + predicted_answer
                )
                full_responses.append(
                    (predicted_answer, token_log_likelihoods, embedding, acc)
                )

        # Store all responses for this example
        generations[example["id"]]["responses"] = full_responses

    # Save generations
    utils.save(generations, "validation_generations.pkl")

    # Log overall accuracy
    accuracy = np.mean(accuracies)
    print(f"Overall validation accuracy: {accuracy}")
    wandb.log({"validation_accuracy": accuracy})

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()
    logging.info("Starting new run with args: %s", args)

    if unknown:
        raise ValueError(f"Unknown args: {unknown}")

    main(args)
