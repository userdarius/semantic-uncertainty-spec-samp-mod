from .huggingface_models import HuggingfaceModel
import torch
from .kvcache_model import KVCacheModel
from .utils import sample, max_fn
from dataclasses import dataclass
from typing import Optional, Tuple
import logging


@dataclass
class SpeculativeOutput:
    sequences: torch.Tensor
    hidden_states: tuple
    logits: torch.Tensor
    scores: Optional[list[torch.Tensor]] = None
    past_key_values: Optional[Tuple[torch.Tensor]] = None
    decoder_hidden_states: Optional[tuple] = None


class SpeculativeSamplingModel(HuggingfaceModel):
    def __init__(
        self,
        target_model_name,
        approx_model_name,
        stop_sequences="default",
        max_new_tokens=20,
    ):
        # Initial setup logging
        logging.info(f"\n{'='*50}")
        logging.info("Initializing SpeculativeSamplingModel:")
        logging.info(f"Target model: {target_model_name}")
        logging.info(f"Approximation model: {approx_model_name}")
        logging.info(f"Max new tokens: {max_new_tokens}")

        # Add 8-bit quantization suffix if not already present
        if not target_model_name.endswith("-8bit"):
            target_model_name += "-8bit"
            logging.info(
                f"Added 8-bit quantization to target model: {target_model_name}"
            )

        if not approx_model_name.endswith("-8bit"):
            approx_model_name += "-8bit"
            logging.info(
                f"Added 8-bit quantization to approximation model: {approx_model_name}"
            )

        logging.info("\nInitializing target model...")
        super().__init__(target_model_name, stop_sequences, max_new_tokens)
        logging.info("\nTarget Model Architecture:")
        logging.info(f"Model type: {type(self.model).__name__}")
        logging.info(
            f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )
        logging.info("Layer structure:")

        for name, module in self.model.named_children():
            logging.info(f"  {name}: {type(module).__name__}")
            if hasattr(module, "config"):
                config = module.config
                logging.info(f"    Hidden size: {config.hidden_size}")
                logging.info(f"    Number of layers: {config.num_hidden_layers}")
                logging.info(
                    f"    Number of attention heads: {config.num_attention_heads}"
                )
                logging.info(f"    Vocabulary size: {config.vocab_size}")

        logging.info("\nInitializing approximation model...")
        self.approx_model = HuggingfaceModel(
            approx_model_name, stop_sequences, max_new_tokens
        )
        logging.info("\nApproximation Model Architecture:")
        logging.info(f"Model type: {type(self.approx_model.model).__name__}")
        logging.info(
            f"Number of parameters: {sum(p.numel() for p in self.approx_model.model.parameters()):,}"
        )
        logging.info("Layer structure:")
        for name, module in self.approx_model.model.named_children():
            logging.info(f"  {name}: {type(module).__name__}")
            if hasattr(module, "config"):
                config = module.config
                logging.info(f"    Hidden size: {config.hidden_size}")
                logging.info(f"    Number of layers: {config.num_hidden_layers}")
                logging.info(
                    f"    Number of attention heads: {config.num_attention_heads}"
                )
                logging.info(f"    Vocabulary size: {config.vocab_size}")

        # Log memory usage
        if torch.cuda.is_available():
            logging.info("\nGPU Memory Usage:")
            logging.info(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            logging.info(f"Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

        self.gamma = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"\nUsing device: {self.device}")
        logging.info(f"Gamma (speculative tokens): {self.gamma}")
        logging.info(f"{'='*50}\n")

        # Log model comparison if possible
        if hasattr(self.model, "config") and hasattr(self.approx_model.model, "config"):
            target_config = self.model.config
            approx_config = self.approx_model.model.config
            logging.info("\nModel Size Comparison:")
            logging.info(f"{'Metric':<25} {'Target':>15} {'Approximation':>15}")
            logging.info(f"{'-'*60}")
            logging.info(
                f"{'Hidden size':<25} {target_config.hidden_size:>15} {approx_config.hidden_size:>15}"
            )
            logging.info(
                f"{'Number of layers':<25} {target_config.num_hidden_layers:>15} {approx_config.num_hidden_layers:>15}"
            )
            logging.info(
                f"{'Attention heads':<25} {target_config.num_attention_heads:>15} {approx_config.num_attention_heads:>15}"
            )
            logging.info(
                f"{'Vocabulary size':<25} {target_config.vocab_size:>15} {approx_config.vocab_size:>15}"
            )

    def compute_transition_scores(self, sequences, scores, normalize_logits=True):
        """Compute transition scores similarly to HuggingFace's compute_transition_scores."""
        log_probs = []
        for logits in scores:
            # Normalize logits to probabilities
            if normalize_logits:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                log_probs_step = torch.nn.functional.log_softmax(logits, dim=-1)
            else:
                probs = logits
                log_probs_step = torch.log(logits)

            # Get the log probability of the selected tokens
            selected_tokens = sequences[:, -len(scores) :]
            batch_size = selected_tokens.shape[0]
            selected_log_probs = log_probs_step[
                torch.arange(batch_size, device=self.device),
                selected_tokens[:, -len(scores)],
            ]
            log_probs.append(selected_log_probs)

        return torch.stack(log_probs).T

    def predict(self, input_data: str, temperature: float, return_full: bool = False):
        """Generate text using speculative sampling with same interface as HuggingFaceModel."""
        logging.info(f"\nStarting prediction with temperature {temperature}")
        logging.info(f"Input length: {len(input_data)} characters")

        # Tokenize input
        inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        n_input_tokens = input_ids.size(1)

        # Setup pad token id like HuggingFaceModel
        if (
            "llama" in self.model_name.lower()
            or "falcon" in self.model_name
            or "mistral" in self.model_name.lower()
        ):
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        logging.info("Initializing KV caches...")
        # Initialize KV caches
        approx_model_cache = KVCacheModel(
            self.approx_model.model, temperature, top_k=20, top_p=0.9
        )
        target_model_cache = KVCacheModel(self.model, temperature, top_k=20, top_p=0.9)
        logging.info("KV caches initialized")

        # Generate using speculative sampling
        outputs = SpeculativeOutput(
            sequences=input_ids.clone(),
            hidden_states=[],
            logits=torch.tensor([]),
            scores=[],
            decoder_hidden_states=[],
        )

        generation_step = 0
        while outputs.sequences.shape[1] < n_input_tokens + self.max_new_tokens:
            generation_step += 1

            logging.info(
                f"Generating tokens... Current length: {outputs.sequences.shape[1]}"
            )
            prefix_len = outputs.sequences.shape[1]
            logging.info(f"\nGeneration step {generation_step}:")
            logging.info(f"Current sequence length: {prefix_len}")

            # Generate from approx model
            logging.info("Generating from approximation model...")
            x = approx_model_cache.generate(outputs.sequences, self.gamma)
            logging.info(f"Approximation model generated {self.gamma} tokens")

            logging.info("Generating from target model...")
            _ = target_model_cache.generate(x, 1)

            n = prefix_len + self.gamma - 1

            # Accept/reject loop with logging
            for i in range(self.gamma):
                j = x[:, prefix_len + i]
                r = torch.rand(1, device=self.device)
                target_prob = target_model_cache._prob_history[:, prefix_len + i - 1, j]
                approx_prob = approx_model_cache._prob_history[:, prefix_len + i - 1, j]

                if r > target_prob / approx_prob:
                    logging.info(
                        f"Token {i+1} rejected (Target prob: {target_prob.item():.4f}, Approx prob: {approx_prob.item():.4f})"
                    )
                    n = prefix_len + i - 1
                    break

                accepted_tokens += 1
                logging.info(
                    f"Token {i+1} accepted (Target prob: {target_prob.item():.4f}, Approx prob: {approx_prob.item():.4f})"
                )

            outputs.sequences = x[:, : n + 1]
            approx_model_cache.rollback(n + 1)
            logging.info(
                f"Accepted {accepted_tokens} out of {self.gamma} proposed tokens"
            )

            if n < prefix_len + self.gamma - 1:
                # Rejection occurred, sample from target
                t = sample(
                    max_fn(
                        target_model_cache._prob_history[:, n, :]
                        - approx_model_cache._prob_history[:, n, :]
                    )
                )
                target_model_cache.rollback(n + 1)
            else:
                # All accepted, sample from target
                t = sample(target_model_cache._prob_history[:, -1, :])
                target_model_cache.rollback(n + 2)

            outputs.sequences = torch.cat((outputs.sequences, t), dim=1)

            # Store hidden states and scores
            outputs.hidden_states = outputs.hidden_states + [
                target_model_cache._past_key_values
            ]
            outputs.decoder_hidden_states = outputs.decoder_hidden_states + [
                target_model_cache._past_key_values
            ]
            outputs.scores = outputs.scores + [
                target_model_cache._prob_history[:, -1, :]
            ]

        # Check token limit
        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                f"Generation exceeding token limit {len(outputs.sequences[0])} > {self.token_limit}"
            )

        # Decode full answer
        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )

        if return_full:
            return full_answer

        # Process output exactly like HuggingFaceModel
        if full_answer.startswith(input_data):
            input_data_offset = len(input_data)
        else:
            raise ValueError("Have not tested this in a while.")

        # Remove input from answer
        answer = full_answer[input_data_offset:]

        # Remove stop_words from answer - Modified to match HuggingFaceModel exactly
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                # Check for stop sequence anywhere in the text
                stop_idx = answer.find(stop)
                if stop_idx != -1:
                    stop_at = stop_idx
                    sliced_answer = answer[:stop_at]
                    break
                # Check for stop sequence at the end
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break

            # Verify stop sequence removal
            if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                error_msg = "Error: Stop words not removed successfully!"
                error_msg += f"Answer: >{answer}< "
                error_msg += f"Sliced Answer: >{sliced_answer}<"
                if "falcon" not in self.model_name.lower():
                    logging.error(error_msg)
                    # For non-Falcon models, we continue and handle it gracefully
                else:
                    logging.error(error_msg)
                    # For Falcon models, return early with empty log likelihoods
                    return sliced_answer.strip(), [], None

        # Remove whitespaces
        sliced_answer = sliced_answer.strip()

        # Calculate token indices exactly like HuggingFaceModel
        token_stop_index = self.tokenizer(
            full_answer[: input_data_offset + stop_at], return_tensors="pt"
        )["input_ids"].shape[1]
        n_generated = token_stop_index - n_input_tokens

        if n_generated == 0:
            logging.warning(
                "Only stop_words were generated. For likelihoods and embeddings, taking stop word instead."
            )
            n_generated = 1

        # Handle hidden states exactly like HuggingFaceModel
        if hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states:
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        # Debug logging
        logging.debug(f"Hidden length: {len(hidden)}")
        if len(hidden) > 0:
            logging.debug(f"First hidden element type: {type(hidden[0])}")
            if isinstance(hidden[0], list):
                logging.debug(
                    f"First hidden element first item type: {type(hidden[0][0])}"
                )

        try:
            last_input = hidden[n_generated - 1]  # This gets us a list

            # Extract the actual tensor from the nested structure
            if isinstance(last_input, list) and len(last_input) > 0:
                # Get the first tuple from the list
                first_tuple = last_input[0]

                # Get the last tensor from the tuple (assuming it's the output tensor)
                if isinstance(first_tuple, tuple):
                    all_tensors = []
                    # Collect all tensors from the tuple
                    for item in first_tuple:
                        if isinstance(item, torch.Tensor):
                            all_tensors.append(item)
                    if all_tensors:
                        tensor = all_tensors[-1]  # Take the last tensor
                    else:
                        raise ValueError("No tensors found in tuple")
                else:
                    tensor = first_tuple

            else:
                tensor = last_input

            # Get embedding
            if isinstance(tensor, torch.Tensor):
                last_token_embedding = tensor[
                    0, -1, :
                ].cpu()  # Add batch dimension handling
            else:
                raise ValueError(f"Unexpected tensor type: {type(tensor)}")

        except Exception as e:
            logging.error(f"Error processing hidden states: {str(e)}")
            logging.error(f"Hidden type: {type(hidden)}")
            logging.error(f"Last input type: {type(last_input)}")
            if isinstance(last_input, list) and len(last_input) > 0:
                logging.error(f"First element type: {type(last_input[0])}")
                if isinstance(last_input[0], tuple):
                    logging.error(f"Tuple length: {len(last_input[0])}")
                    logging.error(
                        f"Tuple element types: {[type(x) for x in last_input[0]]}"
                    )
            raise

        # Compute transition scores exactly like HuggingFaceModel
        transition_scores = self.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            logging.warning("Taking first and only generation for log likelihood!")
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning("Generation interrupted by max_token limit.")

        if len(log_likelihoods) == 0:
            raise ValueError("No log likelihoods computed")

        return sliced_answer, log_likelihoods, last_token_embedding
