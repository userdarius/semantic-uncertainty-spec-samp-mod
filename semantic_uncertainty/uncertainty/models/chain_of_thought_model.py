from .huggingface_models import HuggingfaceModel
import torch
import logging
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np


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

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from generated text."""
        # Try to find explicit answer marker
        answer_markers = [
            "Therefore, ",
            "Thus, ",
            "So, ",
            "The answer is ",
            "Final answer: ",
        ]
        for marker in answer_markers:
            if marker in text:
                return text.split(marker)[-1].strip()

        # Fallback to last sentence
        sentences = text.split(".")
        return sentences[-1].strip()

    def get_last_path_probabilities(self):
        """Get word probabilities from the last prediction."""
        return getattr(self, "last_path_probs", None)

    def compute_answer_confidence(
        self, logits: torch.Tensor, answer_tokens: torch.Tensor
    ) -> float:
        """Compute confidence score for answer tokens as described in paper Section 2.2."""
        if not isinstance(logits, (list, tuple)):
            logits = [logits]

        confidence_scores = []

        for pos, logit in enumerate(logits):
            # Get probabilities for this position
            probs = torch.nn.functional.softmax(logit, dim=-1)[
                0
            ]  # Add [0] to get first batch

            # Sort probabilities in descending order
            sorted_probs, _ = torch.sort(probs, descending=True)

            # Calculate confidence as difference between top two probabilities
            # If there's only one non-zero probability, use 1.0 as confidence
            if len(sorted_probs) > 1:
                prob_diff = (sorted_probs[0] - sorted_probs[1]).item()
            else:
                prob_diff = 1.0

            confidence_scores.append(prob_diff)

        # Average over all positions
        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        return 0.0

    def get_word_level_probs(
        self, token_ids: List[int], token_probs: List[float]
    ) -> List[Tuple[str, float]]:
        """Calculate word-level probabilities by aggregating token probabilities.

        Args:
            token_ids: List of token IDs in the sequence
            token_probs: Probability for each token

        Returns:
            List of (word, probability) tuples
        """
        word_probs = []
        current_word = []
        current_probs = []

        for token_id, prob in zip(token_ids, token_probs):
            token = self.tokenizer.decode([token_id])

            if token.startswith(" ") or not current_word:
                # New word boundary
                if current_word:
                    # Average probability for previous word
                    word = "".join(current_word)
                    avg_prob = sum(current_probs) / len(current_probs)
                    word_probs.append((word, avg_prob))
                    current_word = []
                    current_probs = []

            current_word.append(token.strip())
            current_probs.append(prob)

        # Handle last word
        if current_word:
            word = "".join(current_word)
            avg_prob = sum(current_probs) / len(current_probs)
            word_probs.append((word, avg_prob))

        return word_probs

    def _extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract reasoning steps from generated text.

        Looks for:
        1. Numbered steps (e.g., "1.", "Step 1:", etc.)
        2. Bullet points or dashes
        3. Sequential keywords ("First", "Second", "Finally")
        4. Logical connectors starting new lines

        Args:
            text: Generated text to analyze

        Returns:
            List of extracted reasoning steps
        """
        steps = []
        lines = text.split("\n")
        current_step = ""

        # Keywords that indicate reasoning steps
        step_starters = [
            r"^\d+\.",  # Numbered steps like "1."
            r"^step\s+\d+:?",  # "Step 1:" format
            r"^(first|second|third|finally|next|then)[\s:]",  # Sequential keywords
            r"^\s*[-•]\s+",  # Bullet points or dashes
            r"^(therefore|thus|because|so|hence)",  # Logical connectors
        ]

        step_pattern = "|".join(step_starters)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line starts new step
            if re.match(step_pattern, line.lower()):
                if current_step:  # Save previous step if exists
                    steps.append(current_step.strip())
                current_step = line
            elif current_step:  # Continue current step
                current_step += " " + line
            else:  # Start first step if no clear marker
                current_step = line

        # Add final step
        if current_step:
            steps.append(current_step.strip())

        # Post-process steps
        processed_steps = []
        for step in steps:
            # Remove step numbers/markers
            step = re.sub(r"^\d+\.\s*", "", step)
            step = re.sub(r"^step\s+\d+:?\s*", "", step)
            step = re.sub(r"^[-•]\s+", "", step)

            # Clean up common prefixes
            for prefix in [
                "first",
                "second",
                "third",
                "finally",
                "next",
                "then",
                "therefore",
                "thus",
                "because",
                "so",
                "hence",
            ]:
                if step.lower().startswith(prefix):
                    step = step[len(prefix) :].strip()
                    if step.startswith(",") or step.startswith(":"):
                        step = step[1:].strip()

            if step:  # Only add non-empty steps
                processed_steps.append(step)

        # Handle special case for numerical calculations
        if not processed_steps:
            calculation_steps = re.findall(r"\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+", text)
            if calculation_steps:
                processed_steps.extend(calculation_steps)

        return processed_steps

    def is_cot_path(self, text: str) -> bool:
        """Detect if a generation path contains chain-of-thought reasoning.

        Looks for:
        - Step-by-step reasoning indicators
        - Mathematical operations/calculations
        - Logical connectors (therefore, because, so)
        - Numbered steps

        Args:
            text: Generated text to analyze

        Returns:
            bool: True if path shows CoT reasoning
        """
        # Look for numbered steps
        has_numbered_steps = bool(re.search(r"\d+\s*\.|step\s*\d+", text.lower()))

        # Look for reasoning keywords
        reasoning_indicators = [
            "therefore",
            "because",
            "so",
            "thus",
            "hence",
            "first",
            "second",
            "third",
            "finally",
        ]
        has_reasoning = any(
            indicator in text.lower() for indicator in reasoning_indicators
        )

        # Look for calculations
        has_calculations = bool(re.search(r"\d+\s*[\+\-\*\/]\s*\d+", text))

        return has_numbered_steps or has_calculations or has_reasoning

    def predict(self, input_data: str, temperature: float, return_full: bool = False):
        """Generate text using chain-of-thought reasoning with tree search decoding."""
        logging.info("Starting CoT prediction with temperature %s", temperature)
        logging.info("Input length: %d characters", len(input_data))

        # Enhanced prompt while keeping the CoT format
        if "Question:" not in input_data and "Q:" not in input_data:
            cot_input = (
                f"Question: {input_data}\n"
                "Let's solve this step by step:\n"
                "1. First, let's understand what we know:\n"
            )
        else:
            cot_input = input_data

        inputs = self.tokenizer(cot_input, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = torch.ones_like(input_ids)
        inputs["attention_mask"] = attention_mask

        n_input_tokens = input_ids.size(1)
        logging.info("Input tokens: %d", n_input_tokens)

        # Enhanced base generation kwargs
        gen_kwargs = {
            "output_scores": True,
            "return_dict_in_generate": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_legacy_cache": True,
            "repetition_penalty": 1.3,
            "min_length": n_input_tokens + 20,  # Enforce minimum generation length
            "no_repeat_ngram_size": 3,
            "top_p": 0.95 if temperature > 0 else 1.0,
            "output_hidden_states": True,
            "temperature": max(0.7, temperature),  # Prevent too-low temperature
            "do_sample": True if temperature > 0 else False,
        }

        # Only add bad_words_ids if unk_token_id exists and is not None
        if (
            hasattr(self.tokenizer, "unk_token_id")
            and self.tokenizer.unk_token_id is not None
        ):
            gen_kwargs["bad_words_ids"] = [[self.tokenizer.unk_token_id]]

        # Get initial logits for branching
        initial_kwargs = {**gen_kwargs, "max_new_tokens": 1}
        gen_out = self.model.generate(**inputs, **initial_kwargs)
        initial_logits = gen_out.scores[-1]

        # Get top-k tokens and probabilities
        k = 5  # Number of branches
        k_tokens = initial_logits[0].argsort()[-k:]
        k_probs = torch.nn.functional.softmax(initial_logits[0][k_tokens], dim=0)

        # Store all branch outputs with their metrics
        branch_outputs = []

        # Branch generation kwargs - ensure longer outputs
        branch_kwargs = {
            **gen_kwargs,
            "max_new_tokens": self.max_new_tokens,
            "num_beams": 3,  # Use small beam search for coherence
            "early_stopping": False,  # Don't stop until max length
        }

        valid_branches = 0
        # Explore each branch
        for token, init_prob in zip(k_tokens, k_probs):
            if valid_branches >= 3:  # Ensure we get at least 3 valid branches
                break

            # Create new input with the selected token
            new_query = cot_input + self.tokenizer.decode(token)
            new_inputs = self.tokenizer(new_query, return_tensors="pt").to(self.device)

            try:
                # Generate completion for this branch
                branch_out = self.model.generate(**new_inputs, **branch_kwargs)

                # Decode full output
                full_output = self.tokenizer.decode(
                    branch_out.sequences[0], skip_special_tokens=True
                )

                # Skip if output is too short or contains only numbers/single words
                if len(full_output.split()) < 10 or not self.is_cot_path(full_output):
                    continue

                # Extract answer and check if it's a CoT path
                answer_text = self._extract_answer(full_output)
                is_cot = self.is_cot_path(full_output)

                # Skip if answer is too short
                if len(answer_text.strip()) < 2:
                    continue

                # Get word-level probabilities
                token_ids = branch_out.sequences[0][n_input_tokens:].tolist()
                token_probs = [
                    torch.nn.functional.softmax(score[0], dim=-1).max().item()
                    for score in branch_out.scores
                ]
                word_probs = self.get_word_level_probs(token_ids, token_probs)

                # Compute answer confidence
                answer_tokens = self.tokenizer(answer_text, return_tensors="pt")[
                    "input_ids"
                ][0]
                if len(branch_out.scores) >= len(answer_tokens):
                    answer_logits = branch_out.scores[-len(answer_tokens) :]
                else:
                    answer_logits = branch_out.scores
                confidence = self.compute_answer_confidence(
                    answer_logits, answer_tokens
                )

                # Get hidden states
                try:
                    if (
                        hasattr(branch_out, "decoder_hidden_states")
                        and branch_out.decoder_hidden_states
                    ):
                        hidden = branch_out.decoder_hidden_states[-1]
                    elif hasattr(branch_out, "hidden_states"):
                        hidden = branch_out.hidden_states[-1][-1]
                    else:
                        with torch.no_grad():
                            model_output = self.model(
                                branch_out.sequences, output_hidden_states=True
                            )
                            hidden = model_output.hidden_states[-1]
                    last_token_embedding = hidden[0, -1, :].cpu()
                except Exception as e:
                    logging.warning(f"Could not extract hidden states: {e}")
                    last_token_embedding = torch.zeros(self.model.config.hidden_size)

                branch_outputs.append(
                    {
                        "answer": answer_text,
                        "is_cot": is_cot,
                        "confidence": confidence,
                        "word_probs": word_probs,
                        "embedding": last_token_embedding,
                        "full_output": full_output,
                    }
                )
                valid_branches += 1

            except Exception as e:
                logging.warning(f"Branch generation failed: {e}")
                continue

        # If no valid branches, regenerate with more focused parameters
        if not branch_outputs:
            logging.warning("No valid branches found, attempting recovery generation")
            recovery_kwargs = {
                **branch_kwargs,
                "num_beams": 5,  # Increase beam search
                "temperature": 0.8,  # Stable temperature
                "top_p": 0.95,
                "repetition_penalty": 1.5,
            }
            recovery_out = self.model.generate(**inputs, **recovery_kwargs)
            full_output = self.tokenizer.decode(
                recovery_out[0], skip_special_tokens=True
            )
            answer_text = self._extract_answer(full_output)
            return answer_text, [0.0], torch.zeros(self.model.config.hidden_size)

        # Select best branch based on confidence and CoT presence
        cot_branches = [b for b in branch_outputs if b["is_cot"]]
        if cot_branches:
            # Prefer CoT paths with highest confidence
            best_branch = max(cot_branches, key=lambda x: x["confidence"])
        else:
            # Fallback to highest confidence non-CoT path
            best_branch = max(branch_outputs, key=lambda x: x["confidence"])

        # Store reasoning steps if available
        self.last_reasoning_steps = (
            self._extract_reasoning_steps(best_branch["full_output"])
            if best_branch["is_cot"]
            else None
        )

        # Calculate log likelihoods for compatibility
        log_likelihoods = [-sum([np.log(p) for w, p in best_branch["word_probs"]])]

        return best_branch["answer"], log_likelihoods, best_branch["embedding"]
