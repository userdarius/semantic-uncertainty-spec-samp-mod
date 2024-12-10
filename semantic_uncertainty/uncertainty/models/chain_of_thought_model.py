from .huggingface_models import HuggingfaceModel
import torch
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple
from .utils import sample


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
        cot_prompt_template: str = "Let's solve this step by step:\n1. ",
    ):
        # Initial setup logging
        logging.info("%s", f"\n{'='*50}")
        logging.info("Initializing ChainOfThoughtModel:")
        logging.info("Model: %s", model_name)
        logging.info("Max new tokens: %s", max_new_tokens)

        # Add 8-bit quantization suffix if not already present
        if not model_name.endswith("-8bit"):
            model_name += "-8bit"
            logging.info("Added 8-bit quantization to model: %s", model_name)

        logging.info("\nInitializing model...")
        super().__init__(model_name, stop_sequences, max_new_tokens)

        self.cot_prompt_template = cot_prompt_template
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info("\nModel Architecture:")
        logging.info("Model type: %s", type(self.model).__name__)
        logging.info(
            "Number of parameters: %s", sum(p.numel() for p in self.model.parameters())
        )

        # Log model configuration
        if hasattr(self.model, "config"):
            config = self.model.config
            logging.info("Model Configuration:")
            logging.info("  Hidden size: %s", config.hidden_size)
            logging.info("  Number of layers: %s", config.num_hidden_layers)
            logging.info("  Number of attention heads: %s", config.num_attention_heads)
            logging.info("  Vocabulary size: %s", config.vocab_size)

        # Log memory usage
        if torch.cuda.is_available():
            logging.info("\nGPU Memory Usage:")
            logging.info("Allocated: %.2f MB", torch.cuda.memory_allocated() / 1024**2)
            logging.info("Cached: %.2f MB", torch.cuda.memory_reserved() / 1024**2)

        logging.info("Using device: %s", self.device)
        logging.info("%s", "=" * 50 + "\n")

    def _extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract numbered reasoning steps from the generated text."""
        steps = []
        lines = text.split("\n")
        current_step = ""

        for line in lines:
            if any(line.strip().startswith(str(i) + ".") for i in range(1, 10)):
                if current_step:
                    steps.append(current_step.strip())
                current_step = line.strip()
            elif current_step:
                current_step += " " + line.strip()

        if current_step:
            steps.append(current_step.strip())

        return steps

    def compute_transition_scores(self, sequences, scores, normalize_logits=True):
        """Compute transition scores for the generated sequence."""
        log_probs = []
        for logits in scores:
            if normalize_logits:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                log_probs_step = torch.nn.functional.log_softmax(logits, dim=-1)
            else:
                probs = logits
                log_probs_step = torch.log(logits)

            selected_tokens = sequences[:, -len(scores) :]
            batch_size = selected_tokens.shape[0]
            selected_log_probs = log_probs_step[
                torch.arange(batch_size, device=self.device),
                selected_tokens[:, -len(scores)],
            ]
            log_probs.append(selected_log_probs)

        return torch.stack(log_probs).T

    def predict(self, input_data: str, temperature: float, return_full: bool = False):
        """Generate text using chain-of-thought reasoning with tree search decoding."""
        logging.info("Starting CoT prediction with temperature %s", temperature)
        logging.info("Input length: %d characters", len(input_data))

        # Prepare input with CoT prompt if not already present
        if "Let's solve this step by step" not in input_data:
            cot_input = input_data + "\n" + self.cot_prompt_template
        else:
            cot_input = input_data

        inputs = self.tokenizer(cot_input, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = torch.ones_like(input_ids)
        inputs["attention_mask"] = attention_mask

        n_input_tokens = input_ids.size(1)
        logging.info("Input tokens: %d", n_input_tokens)

        # Get initial logits for branching
        gen_out = self.model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        logit = gen_out.scores[-1]

        # Get top-k tokens and their probabilities
        k = 5  # Number of branches to explore
        k_tokens = logit[0].argsort()[-k:]
        k_probs = torch.nn.functional.softmax(logit[0][k_tokens], dim=0)

        # Store responses and their probabilities for each branch
        branch_outputs = []

        for token, prob in zip(k_tokens, k_probs):
            # Create new input with the selected token
            new_query = cot_input + self.tokenizer.decode(token)
            new_inputs = self.tokenizer(new_query, return_tensors="pt").to(self.device)

            # Generate completion for this branch
            gen_kwargs = {
                "input_ids": new_inputs["input_ids"],
                "attention_mask": torch.ones_like(new_inputs["input_ids"]),
                "max_new_tokens": self.max_new_tokens,
                "temperature": temperature,
                "do_sample": True,
                "output_scores": True,
                "return_dict_in_generate": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "output_hidden_states": True,
            }
            branch_out = self.model.generate(**gen_kwargs)

            # Get path probabilities for this branch
            full_output = self.tokenizer.decode(
                branch_out.sequences[0], skip_special_tokens=True
            )
            reasoning_steps = self._extract_reasoning_steps(
                full_output[len(input_data) :]
            )

            # Get follow-up completion for final answer
            follow_up_template = " Therefore, the answer is: "
            follow_up_ids = self.tokenizer(follow_up_template, return_tensors="pt")[
                "input_ids"
            ].to(self.device)
            follow_up_input_ids = torch.cat(
                [branch_out.sequences, follow_up_ids], dim=1
            )

            follow_up_out = self.model.generate(
                input_ids=follow_up_input_ids,
                attention_mask=torch.ones_like(follow_up_input_ids),
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=20,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Calculate path probabilities including initial token prob
            path_probs = []
            for score in follow_up_out.scores:
                step_probs = torch.nn.functional.softmax(score[0], dim=0)
                path_probs.append(step_probs.max().item())

            avg_path_prob = sum(path_probs) / len(path_probs) * prob.item()

            final_output = self.tokenizer.decode(
                follow_up_out.sequences[0], skip_special_tokens=True
            )
            final_answer = final_output.split("Therefore, the answer is:")[-1].strip()

            branch_outputs.append(
                {
                    "answer": final_answer,
                    "probability": avg_path_prob,
                    "reasoning": reasoning_steps,
                    "hidden_states": (
                        follow_up_out.hidden_states[-1]
                        if hasattr(follow_up_out, "hidden_states")
                        else None
                    ),
                }
            )

        # Select the most confident branch
        best_branch = max(branch_outputs, key=lambda x: x["probability"])

        # Get the final embedding
        if best_branch["hidden_states"] is not None:
            last_token_embedding = best_branch["hidden_states"][0, -1, :].cpu()
        else:
            last_token_embedding = torch.zeros(self.model.config.hidden_size)

        # Store reasoning steps
        self.last_reasoning_steps = best_branch["reasoning"]

        # Calculate log likelihoods for compatibility
        log_likelihoods = [float(best_branch["probability"])]

        return best_branch["answer"], log_likelihoods, last_token_embedding
