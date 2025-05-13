
# Self-Taught Evaluators

This repo contains an implementation of the key ideas in [Self-Taught Evaluators](https://arxiv.org/abs/2408.02666).

Specifically, the implementation contains code (mostly in [self_taught_evaluator.py](self_taught_evaluator.py)) to:
(a) Generate response pairs for a set of instructions using a base model (see `generate_response()` and `generate_bad_response()`)

(b) Generate judgements for the response pairs using a judge model (see `generate_judgements()`)

(c) Evaluate the base model using the judgements (see `evaluate_model_as_judge()`)

(d) Iteratively fine-tune a model to be a better evaluator, and utilities related to this.

We use Together AI (https://together.ai) for inference and fine-tuning.

### Setup

```bash
conda create -n self-taught-evaluators python=3.11
conda activate self-taught-evaluators
pip install -r requirements.txt
```

```bash
export TOGETHER_API_KEY=...
```

### Usage

Running the following command will train the (self-taught) evaluator (see [self_taught_evaluator.py](self_taught_evaluator.py) for all args):
```bash
python self_taught_evaluator.py --data_path path/to/raw/dataset --output_dir path/to/output/directory
```

NOTE: Together AI's models, especially after finetuning, do not support serverless inference. However, it is possible to deploy them to a dedicated endpoint in order to use them for inference. 
Due to the fact that this requires some manual intervention (changing deployed model name in the code, etc), I made it more convenient to run scripts by calling these functions from a Jupyter notebook.

These can be found in [smaller-set.ipynb](smaller-set.ipynb) and [larger-set.ipynb](larger-set.ipynb). The differences are described in the results section.



### Results

In [smaller-set.ipynb](smaller-set.ipynb), I used a smaller dataset for training as well as eval-- this was both because inference and fine-tuning took quite a while,
and also because I wanted to see if we could get signal with a smaller dataset. I used LoRA (Low-rank adapatation) for fine-tuning.

| Param/config | Value |
|-----------|-------|
| Training Set Size | 200 samples |
| Evaluation Set Size | 50 samples |
| Model (both response generation and finetuning) | `mistralai/Mixtral-8x7B-Instruct-v0.1` |
| Number of Iterations | 4 |



We see that the performance increases for the first iteration, but then decreases for the other iterations. Some hypotheses: it is possible the model
is "overfitting" to the small training subset we had, or that the evaluation set was too small. Additionally, it is also possible that we needed a more capable base instruct model.



**Performance Across Iterations:**

| Iteration | Accuracy |
|-----------|----------|
| Base | 74% |
| 1st | 86% |
| 2nd | 74% |
| 3rd | 72% |
| 4th | 72% |

#### Large Dataset Experiment ([larger-set.ipynb](larger-set.ipynb))

In [larger-set.ipynb](larger-set.ipynb) (*currently running, so no results yet*), I used a larger dataset for fine-tuning (600 training samples) and the rest (381 samples) for evaluation.
I also opted to use `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF` as the model for generating response pairs as well as for finetuning.

Currently running with the following parameters:

| Parameter | Value |
|-----------|-------|
| Training Set Size | 600 samples |
| Evaluation Set Size | 381 samples |
| Base Model | `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF` |


[Results TBD]






