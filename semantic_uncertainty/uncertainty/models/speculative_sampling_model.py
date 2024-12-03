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
        logging.info("%s", f"\n{'='*50}")
        logging.info("Initializing SpeculativeSamplingModel:")
        logging.info("Target model: %s", target_model_name)
        logging.info("Approximation model: %s", approx_model_name)
        logging.info("Max new tokens: %s", max_new_tokens)

        # Add 8-bit quantization suffix if not already present
        if not target_model_name.endswith("-8bit"):
            target_model_name += "-8bit"
            logging.info(
                "Added 8-bit quantization to target model: %s", target_model_name
            )

        if not approx_model_name.endswith("-8bit"):
            approx_model_name += "-8bit"
            logging.info(
                "Added 8-bit quantization to approximation model: %s", approx_model_name
            )

        logging.info("\nInitializing target model...")
        super().__init__(target_model_name, stop_sequences, max_new_tokens)
        logging.info("\nTarget Model Architecture:")
        logging.info("Model type: %s", type(self.model).__name__)
        logging.info(
            "Number of parameters: %s", sum(p.numel() for p in self.model.parameters())
        )
        logging.info("Layer structure:")

        for name, module in self.model.named_children():
            logging.info("  %s: %s", name, type(module).__name__)
            if hasattr(module, "config"):
                config = module.config
                logging.info("    Hidden size: %s", config.hidden_size)
                logging.info("    Number of layers: %s", config.num_hidden_layers)
                logging.info(
                    "    Number of attention heads: %s", config.num_attention_heads
                )
                logging.info("    Vocabulary size: %s", config.vocab_size)

        logging.info("\nInitializing approximation model...")
        self.approx_model = HuggingfaceModel(
            approx_model_name, stop_sequences, max_new_tokens
        )
        logging.info("\nApproximation Model Architecture:")
        logging.info("Model type: %s", type(self.approx_model.model).__name__)
        logging.info(
            "Number of parameters: %s",
            "%s" % f"{sum(p.numel() for p in self.approx_model.model.parameters()):,}",
        )
        logging.info("Layer structure:")
        for name, module in self.approx_model.model.named_children():
            logging.info("  %s: %s", name, type(module).__name__)
            if hasattr(module, "config"):
                config = module.config
                logging.info("    Hidden size: %s", config.hidden_size)
                logging.info("    Number of layers: %s", config.num_hidden_layers)
                logging.info(
                    "    Number of attention heads: %s", config.num_attention_heads
                )
                logging.info("    Vocabulary size: %s", config.vocab_size)

        # Log memory usage
        if torch.cuda.is_available():
            logging.info("\nGPU Memory Usage:")
            logging.info("Allocated: %.2f MB", torch.cuda.memory_allocated() / 1024**2)
            logging.info("Cached: %.2f MB", torch.cuda.memory_reserved() / 1024**2)

        self.gamma = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Using device: %s", self.device)
        logging.info("Gamma (speculative tokens): %d", self.gamma)
        logging.info("%s", "=" * 50 + "\n")

        # Log model comparison if possible
        if hasattr(self.model, "config") and hasattr(self.approx_model.model, "config"):
            target_config = self.model.config
            approx_config = self.approx_model.model.config
            logging.info("\nModel Size Comparison:")
            logging.info("%-25s %15s %15s", "Metric", "Target", "Approximation")
            logging.info("%s", "-" * 60)
            logging.info(
                "%-25s %15s %15s",
                "Hidden size",
                target_config.hidden_size,
                approx_config.hidden_size,
            )
            logging.info(
                "%-25s %15s %15s",
                "Number of layers",
                target_config.num_hidden_layers,
                approx_config.num_hidden_layers,
            )
            logging.info(
                "%-25s %15s %15s",
                "Attention heads",
                target_config.num_attention_heads,
                approx_config.num_attention_heads,
            )
            logging.info(
                "%-25s %15s %15s",
                "Vocabulary size",
                target_config.vocab_size,
                approx_config.vocab_size,
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
        logging.info("Starting prediction with temperature %s", temperature)
        logging.info("Input length: %d characters", len(input_data))

        # Tokenize input
        inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        n_input_tokens = input_ids.size(1)
        logging.info("Input tokens: %d", n_input_tokens)

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

        # Initialize KV caches with logging
        logging.info("Initializing KV caches...")
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

        # Generation loop with logging
        generation_step = 0
        while outputs.sequences.shape[1] < n_input_tokens + self.max_new_tokens:
            generation_step += 1
            prefix_len = outputs.sequences.shape[1]
            logging.info("Generation step %d:", generation_step)
            logging.info("Current sequence length: %d", prefix_len)

            # Generate from approx model
            logging.info("Generating from approximation model...")
            x = approx_model_cache.generate(outputs.sequences, self.gamma)
            logging.info("Approximation model generated %d tokens", self.gamma)

            # Get target model probabilities
            logging.info("Getting target model probabilities...")
            _ = target_model_cache.generate(x, 1)

            n = prefix_len + self.gamma - 1
            accepted_tokens = 0

            # Accept/reject loop with logging
            for i in range(self.gamma):
                j = x[:, prefix_len + i]
                r = torch.rand(1, device=self.device)
                target_prob = target_model_cache._prob_history[:, prefix_len + i - 1, j]
                approx_prob = approx_model_cache._prob_history[:, prefix_len + i - 1, j]

                if r > target_prob / approx_prob:
                    logging.info(
                        "Token %d rejected (Target prob: %f, Approx prob: %f)",
                        i + 1,
                        target_prob.item(),
                        approx_prob.item(),
                    )
                    n = prefix_len + i - 1
                    break
                accepted_tokens += 1
                logging.info(
                    "Token %d accepted (Target prob: %f, Approx prob: %f)",
                    i + 1,
                    target_prob.item(),
                    approx_prob.item(),
                )

            outputs.sequences = x[:, : n + 1]
            approx_model_cache.rollback(n + 1)
            logging.info(
                "Accepted %d out of %d proposed tokens", accepted_tokens, self.gamma
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
                "Generation exceeding token limit %d > %d"
                % (len(outputs.sequences[0]), self.token_limit)
            )

        # Decode full answer

        full_answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        logging.info("Full generated text: %s", full_answer)

        if return_full:
            return full_answer

        # Process output and find correct offset
        if full_answer.startswith(input_data):
            input_data_offset = len(input_data)
            logging.info("Using direct input offset: %d", input_data_offset)
        else:
            # Try multiple ways to find the answer section
            content_start = full_answer.find("Answer:")
            if content_start != -1:
                input_data_offset = content_start
            else:
                for line in full_answer.split("\n"):
                    if line.strip().startswith("Answer:"):
                        input_data_offset = full_answer.find(line)
                        break
                else:
                    raise ValueError(f"Cannot find answer content in text: {full_answer}")
            logging.info("Found answer at offset: %d", input_data_offset)

        # Extract answer portion and handle stop sequences
        answer = full_answer[input_data_offset:]
        logging.info("Extracted answer portion: %s", answer)

        stop_at = len(answer)
        sliced_answer = answer

        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                stop_idx = answer.find(stop)
                if stop_idx != -1:
                    stop_at = stop_idx
                    sliced_answer = answer[:stop_at]
                    break
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break

        sliced_answer = sliced_answer.strip()
        logging.info("Processed answer: %s", sliced_answer)

        # Calculate token counts based on the actual text portions
        input_portion = full_answer[:input_data_offset]
        generated_portion = full_answer[input_data_offset : input_data_offset + stop_at]

        # Get token counts for input and generated portions
        input_tokens = self.tokenizer(
            input_portion, return_tensors="pt", return_attention_mask=False
        )["input_ids"]

        full_tokens = self.tokenizer(
            input_portion + generated_portion,
            return_tensors="pt",
            return_attention_mask=False,
        )["input_ids"]

        # Calculate the actual number of generated tokens
        n_generated = len(full_tokens[0]) - len(input_tokens[0])
        logging.info(
            "Token counts - Full: %d, Input: %d, Generated: %d",
            len(full_tokens[0]),
            len(input_tokens[0]),
            n_generated,
        )

        if n_generated <= 0:
            logging.error("Token counting error:")
            logging.error("Input text: %s", input_portion)
            logging.error("Generated text: %s", generated_portion)
            logging.error(
                "Token counts - Full: %d, Input: %d",
                len(full_tokens[0]),
                len(input_tokens[0]),
            )
            raise ValueError(f"Invalid token count: {n_generated} tokens generated")

        # Handle hidden states
        if hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states:
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        logging.info("Hidden states processing:")
        logging.info("n_generated: %d, hidden length: %d", n_generated, len(hidden))

        # Process hidden states and get embedding
        try:
            if n_generated - 1 >= len(hidden):
                logging.warning(
                    "Using last hidden state (index %d > length %d)",
                    n_generated - 1,
                    len(hidden),
                )
                last_input = hidden[-1]
            else:
                last_input = hidden[n_generated - 1]

            # Extract tensor from nested structure
            if isinstance(last_input, list) and len(last_input) > 0:
                first_tuple = last_input[0]
                if isinstance(first_tuple, tuple):
                    all_tensors = [
                        item for item in first_tuple if isinstance(item, torch.Tensor)
                    ]
                    if not all_tensors:
                        raise ValueError("No tensors found in tuple")
                    tensor = all_tensors[-1]
                else:
                    tensor = first_tuple
            else:
                tensor = last_input

            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor, got {type(tensor)}")

            last_token_embedding = tensor[0, -1, :].cpu()

        except Exception as e:
            logging.error("Error processing hidden states: %s", str(e))
            logging.error("Hidden type: %s", type(hidden))
            logging.error(
                "Hidden states info: length=%d, n_generated=%d", len(hidden), n_generated
            )
            raise

        # Compute transition scores
        transition_scores = self.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        # Get log likelihoods
        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            logging.warning("Only one log likelihood value available")
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == 0:
            raise ValueError("No log likelihoods computed")

        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning("Generation reached max_new_tokens limit")

        return sliced_answer, log_likelihoods, last_token_embedding
