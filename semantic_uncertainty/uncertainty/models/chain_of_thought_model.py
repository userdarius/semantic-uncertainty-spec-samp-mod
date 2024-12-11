from .huggingface_models import HuggingfaceModel
import torch
import logging
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class CoTOutput:
    sequences: torch.Tensor
    hidden_states: tuple
    logits: torch.Tensor
    scores: Optional[list[torch.Tensor]] = None
    past_key_values: Optional[Tuple[torch.Tensor]] = None
    decoder_hidden_states: Optional[tuple] = None
    reasoning_steps: List[str] = None


class ChainOfThoughtModel(HuggingfaceModel):
    def __init__(
        self,
        model_name: str,
        stop_sequences="default",
        max_new_tokens=20,
    ):
        super().__init__(model_name, stop_sequences, max_new_tokens)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_logging(model_name, max_new_tokens)

    def _setup_logging(self, model_name, max_new_tokens):
        logging.info("%s", f"\n{'='*50}")
        logging.info("Initializing ChainOfThoughtModel:")
        logging.info("Model: %s", model_name)
        logging.info("Max new tokens: %s", max_new_tokens)
        logging.info("\nModel Architecture:")
        logging.info("Model type: %s", type(self.model).__name__)
        logging.info(
            "Number of parameters: %s", sum(p.numel() for p in self.model.parameters())
        )

    def _get_next_token_logit(self, query):
        """Get logits for next token prediction."""
        inputs = self.tokenizer([query], return_tensors="pt").to(self.device)
        gen_out = self.model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return gen_out.scores[-1]

    def _get_token_path_prob(self, gen_out, num_append=1):
        """Calculate token path probabilities."""
        logits = gen_out.scores
        num_output = len(logits)
        output_ids = gen_out.sequences[0][-num_output - num_append :]
        path_prob = torch.stack([score[0].max() for score in logits])
        path_prob = torch.nn.functional.softmax(path_prob, dim=0)
        return output_ids, path_prob

    def _get_path_prob(self, gen_out, init_token_prob=None):
        """Calculate word-level path probabilities."""
        if init_token_prob is None:
            token_ids, probs = self._get_token_path_prob(gen_out, num_append=0)
        else:
            token_ids, probs = self._get_token_path_prob(gen_out)
            # Convert init_token_prob to tensor and match device
            init_prob_tensor = torch.tensor([init_token_prob]).to(probs.device)
            probs = torch.cat([init_prob_tensor, probs])

        word_probs = []
        ids = []
        current_n_tokens = 0
        word_prob = 0
        current_n_words = 0

        for token_id, prob in zip(token_ids, probs):
            ids.append(token_id)
            decode_seq = self.tokenizer.decode(ids)
            words = re.split(r" |\n|\.\|:", decode_seq)
            word = words[-1]

            if len(words) == current_n_words:
                word_prob += prob.item()  # Convert tensor to scalar
                current_n_tokens += 1
                word_probs[-1] = (word, word_prob / current_n_tokens)
            else:
                word_prob = prob.item()  # Convert tensor to scalar
                current_n_tokens = 1
                word_probs.append((word, word_prob / current_n_tokens))
                current_n_words += 1

        return word_probs

    def _get_follow_up_output(self, gen_out, follow_up_template, max_new_tokens=40):
        """Get follow-up completion with template."""
        construct_input = lambda new_ids: {
            "input_ids": new_ids,
            "attention_mask": torch.ones_like(new_ids),
        }
        output_ids = gen_out.sequences
        follow_up_ids = self.tokenizer(follow_up_template, return_tensors="pt")[
            "input_ids"
        ].to(self.device)
        new_ids = torch.cat([output_ids, follow_up_ids], dim=1)
        inputs = construct_input(new_ids)
        follow_up_out = self.model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            output_hidden_states=True,
        )
        return follow_up_out

    def predict(self, input_data: str, temperature: float, return_full: bool = False):
        """Generate text using chain-of-thought reasoning with tree search decoding."""
        logging.info("Starting CoT prediction with temperature %s", temperature)
        logging.info("Input length: %d characters", len(input_data))

        # Get initial logits for branching
        logit = self._get_next_token_logit(input_data)

        # Get top-k tokens and their probabilities
        k = 5  # Number of branches to explore
        k_tokens = logit[0].argsort()[-k:]
        k_probs = torch.nn.functional.softmax(logit[0][k_tokens], dim=0)

        # Store full response information for each branch
        k_responses = []

        for token, init_prob in zip(k_tokens, k_probs):
            # Create new input with selected token
            new_query = input_data + self.tokenizer.decode(token)
            new_inputs = self.tokenizer(new_query, return_tensors="pt").to(self.device)

            # Generate initial completion
            gen_out = self.model.generate(
                **new_inputs,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                output_hidden_states=True,
            )

            # Get follow-up completion
            follow_up_out = self._get_follow_up_output(
                gen_out, follow_up_template=" Therefore, the answer is: "
            )

            # Calculate path probabilities
            path_probs = self._get_path_prob(follow_up_out, init_prob)
            k_responses.append(path_probs)

            logging.info("Branch probabilities:")
            for word, prob in path_probs:
                logging.info(f"{word}: {prob:.4f}")
            logging.info("----" * 5)

        # Select best response based on average word probability
        best_response_idx = max(
            range(len(k_responses)),
            key=lambda i: sum(p for _, p in k_responses[i]) / len(k_responses[i]),
        )
        best_response = k_responses[best_response_idx]

        # Extract final answer and confidence
        final_answer = (
            self.tokenizer.decode(follow_up_out.sequences[0], skip_special_tokens=True)
            .split("Therefore, the answer is:")[-1]
            .strip()
        )

        # Get final embedding
        if hasattr(follow_up_out, "hidden_states"):
            # Handle tuple of hidden states correctly
            hidden_states = follow_up_out.hidden_states
            try:
                if isinstance(hidden_states, tuple):
                    # For tuple of tuples structure
                    last_layer_hidden = hidden_states[-1][-1]  # Get last layer state
                    if isinstance(last_layer_hidden, tuple):
                        last_layer_hidden = last_layer_hidden[-1]  # Get final state if nested
                    last_token_embedding = last_layer_hidden[0, -1, :].cpu()
                else:
                    # If hidden_states is directly a tensor
                    last_token_embedding = hidden_states[0, -1, :].cpu()
            except Exception as e:
                logging.warning(f"Error extracting hidden states: {e}")
                last_token_embedding = torch.zeros(self.model.config.hidden_size)
        else:
            last_token_embedding = torch.zeros(self.model.config.hidden_size)

        # Calculate average probability for log likelihood
        avg_prob = sum(p for _, p in best_response) / len(best_response)
        log_likelihoods = [float(avg_prob)]

        # Store full path information
        self.last_path_probs = best_response

        return final_answer, log_likelihoods, last_token_embedding

    def get_last_path_probabilities(self):
        """Get word probabilities from the last prediction."""
        return getattr(self, "last_path_probs", None)
