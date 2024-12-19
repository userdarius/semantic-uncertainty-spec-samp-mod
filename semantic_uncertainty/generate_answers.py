"""Sample answers from LLMs for semantic uncertainty analysis."""

import logging
from tqdm import tqdm
import torch
import numpy as np

from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils

utils.setup_logger()


def main(args):
    # Load dataset
    _, validation_dataset = load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed
    )

    # Initialize model
    model = utils.init_model(args)

    # Generate answers
    generations = {}
    for idx in tqdm(range(min(args.num_samples, len(validation_dataset)))):
        example = validation_dataset[idx]
        question = example["question"]
        context = example["context"]

        generations[example["id"]] = {
            "question": question,
            "context": context,
            "reference": utils.get_reference(example),
        }

        # Generate multiple responses at high temperature
        responses = []
        for _ in range(args.num_generations):
            output = model.predict(question, temperature=args.temperature)
            response, token_log_liks, embedding = output
            responses.append((response, token_log_liks, embedding))

        generations[example["id"]]["responses"] = responses

    # Save generations
    utils.save(generations, "validation_generations.pkl")


if __name__ == "__main__":
    parser = utils.get_parser()
    args = parser.parse_args()
    main(args)
