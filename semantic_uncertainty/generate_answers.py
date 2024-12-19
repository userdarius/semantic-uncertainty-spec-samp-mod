"""Sample answers from LLMs for semantic uncertainty analysis."""

import logging
from tqdm import tqdm
import torch
import numpy as np

from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils

utils.setup_logger()


def main(args):
    logging.info("Starting answer generation process")
    logging.info(f"Arguments: {args}")

    # Load dataset
    logging.info(f"Loading dataset: {args.dataset}")
    _, validation_dataset = load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed
    )
    logging.info(f"Loaded validation dataset with {len(validation_dataset)} examples")

    # Initialize model
    logging.info("Initializing model...")
    model = utils.init_model(args)
    logging.info(f"Model initialized: {type(model).__name__}")

    # Generate answers
    generations = {}
    num_samples = min(args.num_samples, len(validation_dataset))
    logging.info(
        f"Starting generation for {num_samples} samples with {args.num_generations} generations each"
    )

    for idx in tqdm(range(num_samples)):
        example = validation_dataset[idx]
        question = example["question"]
        context = example["context"]
        example_id = example["id"]

        logging.info(f"\nProcessing example {idx+1}/{num_samples}")
        logging.info(f"Example ID: {example_id}")
        logging.info(f"Question: {question}")
        logging.info(
            f"Context: {context[:200]}..."
            if len(context) > 200
            else f"Context: {context}"
        )

        generations[example_id] = {
            "question": question,
            "context": context,
            "reference": utils.get_reference(example),
        }

        # Generate multiple responses at high temperature
        responses = []
        logging.info(f"Generating {args.num_generations} responses...")

        for gen_idx in range(args.num_generations):
            try:
                output = model.predict(question, temperature=args.temperature)
                response, token_log_liks, embedding = output
                responses.append((response, token_log_liks, embedding))

                logging.info(f"Generation {gen_idx+1}:")
                logging.info(f"Response: {response}")
                logging.info(
                    f"Token log likelihoods shape: {np.array(token_log_liks).shape if token_log_liks else 'None'}"
                )
                logging.info(
                    f"Embedding shape: {embedding.shape if embedding is not None else 'None'}"
                )

            except Exception as e:
                logging.error(f"Error in generation {gen_idx+1}: {str(e)}")
                continue

        generations[example_id]["responses"] = responses
        logging.info(f"Completed {len(responses)} generations for example {idx+1}")

    # Save generations
    logging.info("Saving generations to validation_generations.pkl")
    utils.save(generations, "validation_generations.pkl")
    logging.info("Generation process completed successfully")
