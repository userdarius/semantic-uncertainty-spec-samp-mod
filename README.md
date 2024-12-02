## Installation Guide


To install Python with all necessary dependencies, we recommend the use of conda, and we refer to [https://conda.io/](https://conda.io/) for an installation guide.


After installing conda, you can set up and activate a new conda environment with all required packages by executing the following commands from the root folder of this repository in a shell:


```
conda-env update -f environment.yaml
conda activate semantic_uncertainty
```

The installation should take around 15 minutes.

## Run

Execute

```
python3 semantic_uncertainty/generate_answers.py     --use_speculative_sampling     --target_model_name "Llama-2-7b-chat"     --approx_model_name "Llama-2-7b-chat"     --model_name "Llama-2-7b-chat"```
