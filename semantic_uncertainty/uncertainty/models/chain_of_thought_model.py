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
        cot_prompt_template: str = "Let's solve this step by step:\n1. "
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
            "Number of parameters: %s",
            sum(p.numel() for p in self.model.parameters())
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
        lines = text.split('\n')
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

            selected_tokens = sequences[:, -len(scores):]
            batch_size = selected_tokens.shape[0]
            selected_log_probs = log_probs_step[
                torch.arange(batch_size, device=self.device),
                selected_tokens[:, -len(scores)],
            ]
            log_probs.append(selected_log_probs)

        return torch.stack(log_probs).T

    def predict(self, input_data: str, temperature: float, return_full: bool = False):
        """Generate text using chain-of-thought reasoning."""
        logging.info("Starting CoT prediction with temperature %s", temperature)
        logging.info("Input length: %d characters", len(input_data))

        # Prepare input with CoT prompt
        cot_input = input_data + "\n" + self.cot_prompt_template
        inputs = self.tokenizer(cot_input, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        n_input_tokens = input_ids.size(1)
        logging.info("Input tokens: %d", n_input_tokens)

        # Setup generation parameters
        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": self.max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "output_scores": True,
            "return_dict_in_generate": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        # Generate with chain-of-thought reasoning
        outputs = self.model.generate(**gen_kwargs)
        
        # Process outputs
        full_output = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        reasoning_steps = self._extract_reasoning_steps(full_output[len(input_data):])
        
        logging.info("Generated reasoning steps: %d", len(reasoning_steps))
        for i, step in enumerate(reasoning_steps, 1):
            logging.info("Step %d: %s", i, step)

        if return_full:
            return full_output

        # Extract final answer after reasoning
        final_answer = full_output[len(cot_input):]
        
        # Handle stop sequences
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                stop_idx = final_answer.find(stop)
                if stop_idx != -1:
                    final_answer = final_answer[:stop_idx]
                    break

        final_answer = final_answer.strip()
        logging.info("Final processed answer: %s", final_answer)

        # Calculate token counts
        generated_tokens = outputs.sequences[0][n_input_tokens:]
        n_generated = len(generated_tokens)

        # Process hidden states
        if hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states:
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else None

        # Get last token embedding
        if hidden is not None:
            last_token_embedding = hidden[0, -1, :].cpu()
        else:
            last_token_embedding = None

        # Compute transition scores
        transition_scores = self.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        # Get log likelihoods
        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) > n_generated:
            log_likelihoods = log_likelihoods[:n_generated]

        return final_answer, log_likelihoods, last_token_embedding, reasoning_steps