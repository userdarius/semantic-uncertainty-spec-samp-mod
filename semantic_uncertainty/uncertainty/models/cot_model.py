from .huggingface_models import HuggingfaceModel
import torch
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import logging


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

    def generate_single_response(
        self, inputs: Dict[str, torch.Tensor], max_length: int = 500
    ) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        """Generate a single response with probability tracking and hidden states."""
        try:
            # Ensure inputs are on correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Initialize storage for probabilities and hidden states
            response_probs = []
            hidden_states = []

            # Check token limit
            if max_length > self.token_limit:
                raise ValueError(
                    f"max_length {max_length} exceeds token limit {self.token_limit}"
                )

            for _ in range(max_length):
                # Generate with output_hidden_states=True to match speculative model
                with torch.no_grad():
                    outputs = self.model(
                        **inputs, output_hidden_states=True, return_dict=True
                    )

                # Store hidden states
                hidden_states.append(outputs.hidden_states[-1])

                # Get next token probabilities
                next_token_logits = outputs.logits[:, -1, :]
                probabilities = torch.softmax(next_token_logits, dim=-1)

                # Get top tokens
                topk_values, topk_indices = torch.topk(probabilities, k=2)

                # Calculate probability difference
                prob_diff = topk_values[:, 0] - topk_values[:, 1]
                response_probs.append(prob_diff.item())

                # Get next token
                next_token = topk_indices[:, 0].unsqueeze(-1)

                # Handle special tokens based on model type
                if (
                    "llama" in self.model_name.lower()
                    or "falcon" in self.model_name
                    or "mistral" in self.model_name.lower()
                ):
                    pad_token_id = self.tokenizer.eos_token_id
                    if next_token.item() == pad_token_id:
                        break
                elif next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Update inputs
                inputs["input_ids"] = torch.cat(
                    [inputs["input_ids"], next_token], dim=1
                )
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.cat(
                        [
                            inputs["attention_mask"],
                            torch.ones((1, 1), dtype=torch.long, device=self.device),
                        ],
                        dim=1,
                    )

            return inputs["input_ids"], response_probs, torch.stack(hidden_states)

        except Exception as e:
            logging.error("Error in generate_single_response: %s", str(e))
            raise

    def predict(
        self,
        input_data: str,
        temperature: float,
        return_full: bool = False,
        use_branching: bool = False,
        num_branches: int = 10,
    ) -> tuple[str, List[float], torch.Tensor]:
        """Enhanced predict method with better error handling and output processing."""
        try:
            logging.info("Starting prediction with temperature %s", temperature)
            logging.info("Input length: %d characters", len(input_data))

            if not input_data.strip():
                raise ValueError("Empty input data")

            if use_branching:
                # Clear GPU cache before branching
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Tokenize input
                inputs = self.tokenizer(input_data, return_tensors="pt")
                n_input_tokens = inputs["input_ids"].size(1)
                logging.info("Input tokens: %d", n_input_tokens)

                best_response = None
                best_score = float("-inf")
                best_hidden_states = None
                best_probs = None

                # Generate branches
                for k in tqdm(range(num_branches), desc="Generating branches"):
                    response_ids, probs, hidden = self.generate_single_response(
                        inputs.copy()
                    )
                    avg_prob = sum(probs) / len(probs) if probs else 0

                    if avg_prob > best_score:
                        best_score = avg_prob
                        best_response = response_ids
                        best_hidden_states = hidden
                        best_probs = probs

                # Decode response
                full_answer = self.tokenizer.decode(
                    best_response[0], skip_special_tokens=True
                )

                if return_full:
                    return full_answer

                # Process output similar to speculative model
                if full_answer.startswith(input_data):
                    input_data_offset = len(input_data)
                    logging.info("Using direct input offset: %d", input_data_offset)
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

                # Extract answer and handle stop sequences
                answer = full_answer[input_data_offset:].strip()
                if self.stop_sequences:
                    for stop in self.stop_sequences:
                        if stop in answer:
                            answer = answer[: answer.find(stop)].strip()

                # Get last token embedding
                last_token_embedding = best_hidden_states[-1][0, -1, :].cpu()

                return answer, best_probs, last_token_embedding

            else:
                # Use parent class implementation for non-branching generation
                return super().predict(input_data, temperature, return_full)

        except Exception as e:
            logging.error("Error in predict: %s", str(e))
            torch.cuda.empty_cache()  # Cleanup on error
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
