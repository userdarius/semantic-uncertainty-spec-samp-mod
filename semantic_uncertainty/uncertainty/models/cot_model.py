from .huggingface_models import HuggingfaceModel
import torch
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import logging
import random


class ChainOfThoughtHuggingfaceModel(HuggingfaceModel):
    """Hugging Face Model with improved Chain of Thought decoding logic."""

    def __init__(
        self,
        model_name: str,
        stop_sequences: Optional[List[str]] = "default",
        max_new_tokens: Optional[int] = 20,
    ):
        # Initial setup logging
        logging.info("%s", f"\n{'='*50}")
        logging.info("Initializing ChainOfThoughtHuggingfaceModel:")
        logging.info("Model name: %s", model_name)
        logging.info("Max new tokens: %s", max_new_tokens)

        # Add 8-bit quantization suffix if not already present
        if not model_name.endswith("-8bit"):
            model_name += "-8bit"
            logging.info("Added 8-bit quantization to model: %s", model_name)

        super().__init__(model_name, stop_sequences, max_new_tokens)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Using device: %s", self.device)

        # Log model architecture
        logging.info("\nModel Architecture:")
        logging.info("Model type: %s", type(self.model).__name__)
        logging.info(
            "Number of parameters: %s",
            f"{sum(p.numel() for p in self.model.parameters()):,}",
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

        # Log memory usage
        if torch.cuda.is_available():
            logging.info("\nGPU Memory Usage:")
            logging.info("Allocated: %.2f MB", torch.cuda.memory_allocated() / 1024**2)
            logging.info("Cached: %.2f MB", torch.cuda.memory_reserved() / 1024**2)

        logging.info("%s", "=" * 50 + "\n")

    def get_topk_tokens(
        self, inputs: Dict[str, torch.Tensor], num_branches: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the top k most likely next tokens and their probabilities."""
        # Move inputs to correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            # Apply softmax to convert logits to probabilities
            probabilities = torch.softmax(next_token_logits, dim=-1)

            # Get the top k tokens and their probabilities
            topk_values, topk_indices = torch.topk(probabilities, num_branches)

            return topk_values, topk_indices

    def generate_single_response_batch(
        self,
        inputs: Dict[str, torch.Tensor],
        num_branches: int = 5,
        max_length: int = 500,
    ) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        """Generate multiple responses in parallel using batching."""
        try:
            # Expand inputs to create branches in one batch
            batch_size = num_branches
            inputs = {
                k: (
                    v.repeat(batch_size, 1)
                    if v.dim() == 2
                    else v.repeat(batch_size, 1, 1)
                )
                for k, v in inputs.items()
            }
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Initialize storage
            response_probs = torch.zeros(
                batch_size, dtype=torch.float, device=self.device
            )
            final_hidden_states = None
            all_response_ids = inputs["input_ids"]

            # Generate tokens for all branches simultaneously
            for _ in range(max_length):
                with torch.no_grad():
                    outputs = self.model(
                        **inputs, output_hidden_states=True, return_dict=True
                    )

                # Store final hidden states
                final_hidden_states = outputs.hidden_states[-1]

                # Get next token probabilities for all branches
                next_token_logits = outputs.logits[:, -1, :]
                probabilities = torch.softmax(next_token_logits, dim=-1)

                # Sample next tokens for all branches
                if random.random() < 0.9:  # Add some randomness to branches
                    next_tokens = torch.multinomial(probabilities, num_samples=1)
                else:
                    # Sometimes take top-k for diversity
                    _, next_tokens = torch.topk(probabilities, k=1, dim=-1)

                # Update response probabilities
                token_probs = torch.gather(probabilities, 1, next_tokens)
                response_probs += token_probs.squeeze(-1)

                # Check for stopping conditions
                is_finished = next_tokens == self.tokenizer.eos_token_id
                if is_finished.all():
                    break

                # Update inputs for next iteration
                inputs["input_ids"] = torch.cat(
                    [inputs["input_ids"], next_tokens], dim=1
                )
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.cat(
                        [
                            inputs["attention_mask"],
                            torch.ones(
                                (batch_size, 1), dtype=torch.long, device=self.device
                            ),
                        ],
                        dim=1,
                    )

            # Average the probabilities over sequence length
            response_probs = response_probs / (
                inputs["input_ids"].size(1) - all_response_ids.size(1)
            )

            return inputs["input_ids"], response_probs.tolist(), final_hidden_states

        except Exception as e:
            logging.error("Error in generate_single_response_batch: %s", str(e))
            raise

    def predict(
        self,
        input_data: str,
        temperature: float,
        return_full: bool = False,
        use_branching: bool = True,
        num_branches: int = 10,
    ) -> tuple[str, List[float], torch.Tensor]:
        """Optimized predict method using batched generation."""
        try:
            logging.info("Starting prediction with temperature %s", temperature)
            logging.info("Input length: %d characters", len(input_data))

            if not input_data.strip():
                raise ValueError("Empty input data")

            if use_branching:
                torch.cuda.empty_cache()

                # Tokenize input
                inputs = self.tokenizer(input_data, return_tensors="pt")
                logging.info("Input tokens: %d", inputs["input_ids"].size(1))

                # Generate all branches in one pass
                response_ids, probs, hidden_states = (
                    self.generate_single_response_batch(
                        inputs,
                        num_branches=num_branches,
                        max_length=self.max_new_tokens,
                    )
                )

                # Find best response
                best_idx = max(range(len(probs)), key=lambda i: probs[i])
                best_response = response_ids[best_idx : best_idx + 1]
                best_probs = [probs[best_idx]]
                best_hidden_state = hidden_states[best_idx : best_idx + 1]

                # Decode response
                full_answer = self.tokenizer.decode(
                    best_response[0], skip_special_tokens=True
                )

                if return_full:
                    return full_answer

                # Process output
                if full_answer.startswith(input_data):
                    input_data_offset = len(input_data)
                else:
                    content_start = full_answer.find("Answer:")
                    if content_start != -1:
                        input_data_offset = content_start
                    else:
                        for line in full_answer.split("\n"):
                            if line.strip().startswith("Answer:"):
                                input_data_offset = full_answer.find(line)
                                break
                        else:
                            raise ValueError(
                                f"Cannot find answer content in text: {full_answer}"
                            )

                # Extract answer
                answer = full_answer[input_data_offset:].strip()
                if self.stop_sequences:
                    for stop in self.stop_sequences:
                        if stop in answer:
                            answer = answer[: answer.find(stop)].strip()

                last_token_embedding = best_hidden_state[0, -1, :].cpu()

                return answer, best_probs, last_token_embedding

            else:
                return super().predict(input_data, temperature, return_full)

        except Exception as e:
            logging.error("Error in predict: %s", str(e))
            torch.cuda.empty_cache()
            raise
        finally:
            torch.cuda.empty_cache()
