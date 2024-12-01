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

        # Add 8-bit quantization suffix if not already present
        if not target_model_name.endswith("-8bit"):
            target_model_name += "-8bit"
        if not approx_model_name.endswith("-8bit"):
            approx_model_name += "-8bit"

        super().__init__(target_model_name, stop_sequences, max_new_tokens)
        self.approx_model = HuggingfaceModel(
            approx_model_name, stop_sequences, max_new_tokens
        )
        self.gamma = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Initialize KV caches
        approx_model_cache = KVCacheModel(
            self.approx_model.model, temperature, top_k=20, top_p=0.9
        )
        target_model_cache = KVCacheModel(self.model, temperature, top_k=20, top_p=0.9)

        # Generate using speculative sampling
        outputs = SpeculativeOutput(
            sequences=input_ids.clone(),
            hidden_states=[],
            logits=torch.tensor([]),
            scores=[],
            decoder_hidden_states=[],
        )

        while outputs.sequences.shape[1] < n_input_tokens + self.max_new_tokens:
            prefix_len = outputs.sequences.shape[1]

            # Generate from approx model
            x = approx_model_cache.generate(outputs.sequences, self.gamma)
            _ = target_model_cache.generate(x, 1)

            n = prefix_len + self.gamma - 1

            # Accept/reject loop
            for i in range(self.gamma):
                j = x[:, prefix_len + i]
                r = torch.rand(1, device=self.device)

                if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (
                    approx_model_cache._prob_history[:, prefix_len + i - 1, j]
                ):
                    n = prefix_len + i - 1
                    break

            outputs.sequences = x[:, : n + 1]
            approx_model_cache.rollback(n + 1)

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

        try:
            if len(hidden) == 1:
                logging.warning(
                    "Taking first and only generation for hidden!"
                )
                last_input = hidden[0]
            elif (n_generated - 1) >= len(hidden):
                logging.error(
                    "Taking last state because n_generated is too large"
                )
                last_input = hidden[-1]
            else:
                last_input = hidden[n_generated - 1]

            # Handle the list structure
            if isinstance(last_input, list) and len(last_input) > 0:
                # Get the first tensor from the list
                last_tensor = last_input[0]
                # Extract the last token embedding directly from the tensor
                last_token_embedding = last_tensor[:, -1, :].cpu()
            else:
                # Fallback for other cases
                last_layer = last_input if not isinstance(last_input, list) else last_input[0]
                last_token_embedding = last_layer[:, -1, :].cpu()

        except Exception as e:
            logging.error(f"Error processing hidden states: {str(e)}")
            logging.error(f"last_input type: {type(last_input)}")
            logging.error(f"last_input structure: {last_input}")
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