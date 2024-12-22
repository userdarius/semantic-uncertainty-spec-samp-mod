from .huggingface_models import HuggingfaceModel
import logging


class ChainOfThoughtModel(HuggingfaceModel):
    """Extends HuggingfaceModel to add Chain of Thought reasoning capabilities."""

    def __init__(
        self, model_name, stop_sequences=None, max_new_tokens=None, cot_prompt=None
    ):
        """Initialize the Chain of Thought model.

        Args:
            model_name (str): Name of the Huggingface model to use
            stop_sequences (list, optional): List of sequences to stop generation
            max_new_tokens (int): Maximum number of tokens to generate
            cot_prompt (str, optional): Custom chain of thought prompt template.
                                      If None, uses default template.
        """
        super().__init__(model_name, stop_sequences, max_new_tokens)

        # Default chain of thought prompt if none provided
        self.cot_prompt = cot_prompt or (
            "Let's approach this step by step:\n"
            "1. First, let's understand what we're being asked\n"
            "2. Then, let's break down the key components\n"
            "3. Next, let's analyze each part carefully\n"
            "4. Finally, we'll combine our findings to reach a conclusion\n\n"
            "Question: {input}\n\n"
            "Let's answer this question:\n"
        )

    def predict(self, input_data, temperature, return_full=False):
        """Override predict method to include chain of thought reasoning.

        Args:
            input_data (str): The input prompt/question
            temperature (float): Sampling temperature for generation
            return_full (bool): Whether to return the full response including the prompt

        Returns:
            Same as parent class - (answer, log_likelihoods, last_token_embedding)
            or full_answer if return_full=True
        """
        # Format input with CoT prompt
        try:
            cot_input = self.cot_prompt.format(input=input_data)
        except KeyError as e:
            logging.error(f"Failed to format CoT prompt: {e}")
            logging.warning("Falling back to default formatting")
            cot_input = f"{self.cot_prompt}{input_data}"

        # Call parent class predict with CoT-enhanced input
        result = super().predict(cot_input, temperature, return_full)

        if return_full:
            return result

        answer, log_likelihoods, last_token_embedding = result

        # Extract final answer if needed (could add post-processing here)
        return answer, log_likelihoods, last_token_embedding
