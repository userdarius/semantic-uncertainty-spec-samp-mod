"""Implement HuggingfaceModel models."""

import copy
import logging
from collections import Counter
import torch

import accelerate

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download


from uncertainty.models.base_model import BaseModel
from uncertainty.models.base_model import STOP_SEQUENCES


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""

    def __init__(self, stops, tokenizer, match_on="text", initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == "tokens":
            self.stops = [
                torch.tensor(self.tokenizer.encode(i)).to("cuda") for i in self.stops
            ]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` arg is required by StoppingCriteria but unused by us.
        for stop in self.stops:
            if self.match_on == "text":
                generation = self.tokenizer.decode(
                    input_ids[0][self.initial_length :], skip_special_tokens=False
                )
                match = stop in generation
            elif self.match_on == "tokens":
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop) :]
            else:
                raise
            if match:
                return True
        return False


def remove_split_layer(device_map_in):
    """Modify device maps s.t. individual layers are not spread across devices."""

    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())

    counts = Counter([".".join(i.split(".")[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            # Only triggers if we find more than one split layer.
            raise ValueError(
                "More than one split layer.\n"
                f"Currently at layer {layer}.\n"
                f"In map: {device_map_in}\n"
                f"Out map: {device_map}\n"
            )

        logging.info(f"Split layer is {layer}.")

        # Remove split for that layer.
        for name in list(device_map.keys()):
            if name.startswith(layer):
                print(f"pop {name}")
                device = device_map.pop(name)

        device_map[layer] = device
        found_split = True

    return device_map


class HuggingfaceModel(BaseModel):
    """Hugging Face Model."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
        if max_new_tokens is None:
            raise ValueError("max_new_tokens cannot be None")
        self.max_new_tokens = max_new_tokens

        if stop_sequences == "default":
            stop_sequences = STOP_SEQUENCES

        if "llama" in model_name.lower():

            if model_name.endswith("-8bit"):
                kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
                model_name = model_name[: -len("-8bit")]
                eightbit = True
            else:
                kwargs = {}
                eightbit = False

            if "Llama-3.2" in model_name:
                base = "meta-llama"
            elif "Llama-2" in model_name:
                base = "meta-llama"
                model_name = model_name + "-hf"
            else:
                base = "huggyllama"

            self.tokenizer = AutoTokenizer.from_pretrained(
                f"{base}/{model_name}", device_map="auto", token_type_ids=None
            )

            if "3.2" in model_name:
                kwargs = {}
                self.model = AutoModelForCausalLM.from_pretrained(
                    f"{base}/{model_name}",
                    device_map="auto",
                    max_memory={0: "80GIB"},
                    **kwargs,
                )
                self.token_limit = 128000  # 128k tokens supported
            elif ("7b" in model_name or "13b" in model_name) or eightbit:
                self.model = AutoModelForCausalLM.from_pretrained(
                    f"{base}/{model_name}",
                    device_map="auto",
                    max_memory={0: "80GIB"},
                    **kwargs,
                )
                self.token_limit = 4096
            else:
                raise ValueError

        elif "mistral" in model_name.lower():
            if model_name.endswith("-8bit"):
                kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
                model_name = model_name[: -len("-8bit")]
            else:
                kwargs = {}

            model_id = f"mistralai/{model_name}"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                device_map="auto",
                token_type_ids=None,
                clean_up_tokenization_spaces=False,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                max_memory={0: "80GIB"},
                **kwargs,
            )
            self.token_limit = 2048

        elif "falcon" in model_name:
            model_id = f"tiiuae/{model_name}"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                device_map="auto",
                token_type_ids=None,
                clean_up_tokenization_spaces=False,
            )

            kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map="auto",
                **kwargs,
            )
            self.token_limit = 2048
        else:
            raise ValueError

        self.model_name = model_name
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]

    def predict(self, input_data, temperature, return_full=False):
        # Tokenize the input
        inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")

        if (
            "llama" in self.model_name.lower()
            or "falcon" in self.model_name
            or "mistral" in self.model_name.lower()
        ):
            if "token_type_ids" in inputs:  # Some HF models include this
                del inputs["token_type_ids"]
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        # Set stopping criteria
        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList(
                [
                    StoppingCriteriaSub(
                        stops=self.stop_sequences,
                        initial_length=len(inputs["input_ids"][0]),
                        tokenizer=self.tokenizer,
                    )
                ]
            )
        else:
            stopping_criteria = None

        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                do_sample=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
            )

        # Check for token limit
        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                f"Generation exceeding token limit {len(outputs.sequences[0])} > {self.token_limit}"
            )

        # Decode the full answer
        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )

        if return_full:
            return full_answer

        # Handle cases where full_answer does not start with input_data
        if full_answer.startswith(input_data):
            input_data_offset = len(input_data)
        else:
            logging.warning(
                "Mismatch between input_data and full_answer. Proceeding with offset reset."
            )
            input_data_offset = 0

        # Remove input_data from the answer
        answer = full_answer[input_data_offset:].strip()

        # Remove stop sequences
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
            if not all(stop not in sliced_answer for stop in self.stop_sequences):
                logging.error(
                    f"Stop sequences not fully removed. Answer: {answer}, Sliced: {sliced_answer}"
                )

        # Strip extra spaces
        sliced_answer = sliced_answer.strip()

        # Calculate the number of tokens in the generated output
        token_stop_index = self.tokenizer(
            full_answer[: input_data_offset + stop_at], return_tensors="pt"
        )["input_ids"].shape[1]
        n_input_token = len(inputs["input_ids"][0])
        n_generated = token_stop_index - n_input_token

        if n_generated == 0:
            logging.warning(
                "Only stop_words were generated. Using stop word instead for embeddings and likelihoods."
            )
            n_generated = 1

        # Extract the last hidden state and token embeddings
        hidden_states = (
            outputs.decoder_hidden_states
            if "decoder_hidden_states" in outputs
            else outputs.hidden_states
        )

        if len(hidden_states) == 1:
            logging.warning("Using first and only hidden state for embedding.")
            last_hidden_state = hidden_states[0]
        elif (n_generated - 1) >= len(hidden_states):
            logging.error(
                f"Access index out of range for hidden states. Using last hidden state. "
                f"n_generated: {n_generated}, hidden states length: {len(hidden_states)}"
            )
            last_hidden_state = hidden_states[-1]
        else:
            last_hidden_state = hidden_states[n_generated - 1]

        # Get last token embedding
        last_layer = last_hidden_state[-1]
        last_token_embedding = last_layer[:, -1, :].cpu()

        # Calculate log-likelihoods
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        log_likelihoods = [score.item() for score in transition_scores[0]]
        log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == 0:
            raise ValueError("Log likelihoods are empty.")

        return sliced_answer, log_likelihoods, last_token_embedding

    def get_p_true(self, input_data):
        """Get the probability of the model anwering A (True) for the given input."""

        input_data += " A"
        tokenized_prompt_true = self.tokenizer(input_data, return_tensors="pt").to(
            "cuda"
        )["input_ids"]
        # The computation of the negative log likelihoods follows:
        # https://huggingface.co/docs/transformers/perplexity.

        target_ids_true = tokenized_prompt_true.clone()
        # Set all target_ids except the last one to -100.
        target_ids_true[0, :-1] = -100

        with torch.no_grad():
            model_output_true = self.model(
                tokenized_prompt_true, labels=target_ids_true
            )

        loss_true = model_output_true.loss

        return -loss_true.item()
