## Installation Guide


To install Python with all necessary dependencies, we recommend the use of conda, and we refer to [https://conda.io/](https://conda.io/) for an installation guide.


After installing conda, you can set up and activate a new conda environment with all required packages by executing the following commands from the root folder of this repository in a shell:


```
conda-env update -f environment.yaml
conda activate semantic_uncertainty
```

The installation should take around 15 minutes.

## Run

Execute with speculative sampling

```
python3 hf_login.py # for logging into huggingface quickly (currently uses my api key)
python3 semantic_uncertainty/generate_answers.py     --use_speculative_sampling     --target_model_name "Llama-3.2-3b"     --approx_model_name "Llama-3.2-1b"     --model_name "Llama-3.2-3b"
```

Execute with CoT reasoning

```
python3 semantic_uncertainty/generate_answers.py --use_chain_of_thought --model_name "Llama-3.2-3b" 
```
