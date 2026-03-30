# HW1_314706007

Medical question answering for PathoQA using `meta-llama/Llama-3.2-1B-Instruct` with LoRA fine-tuning.

This project follows the homework constraints:

- Only `meta-llama/Llama-3.2-1B-Instruct` is used.
- Inference is generation-based.
- Predictions are extracted from generated text.
- Model ensembling uses hard voting only.
- No logit-based inference or soft voting is used.

## Repository Layout

```text
HW1_314706007/
├── code/
│   ├── train_eval.py
│   ├── inference.py
│   └── utils.py
├── dataset/
│   ├── dataset.csv
│   └── benchmark.csv
├── saved_models/
│   ├── checkpoint/
│   └── splits/
├── requirements.txt
└── README.md
```

## Environment Setup

Recommended Python version: `3.10+`

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The model `meta-llama/Llama-3.2-1B-Instruct` is gated on Hugging Face, so you must have access to it and log in before training or inference:

```bash
huggingface-cli login
```

## Dataset

Put the homework files here:

- `dataset/dataset.csv`
- `dataset/benchmark.csv`

The scripts assume the current working directory is `code/`, because the default paths in the config are written relative to that directory.

## Method Summary

The pipeline has three main parts:

1. Prompt engineering
   The model receives an instruct-style prompt with two pathology few-shot examples and must answer in the format:
   `Reasoning: ... Final Answer: <A/B/C/D>`

2. LoRA fine-tuning
   The training script fine-tunes `Llama-3.2-1B-Instruct` using LoRA on the PathoQA training split. The supervised target is generation-style text rather than logits.

3. Generation-based inference
   The benchmark questions are answered by text generation. The final choice is parsed from generated output. If multiple models are supplied, the final answer is decided by hard voting. Optional option-order hard voting is also supported.

## Default Training Configuration

Current defaults in `code/train_eval.py`:

- `model_name=meta-llama/Llama-3.2-1B-Instruct`
- `batch_size=8`
- `eval_batch_size=4`
- `learning_rate=2e-4`
- `num_epochs=15`
- `weight_decay=0.02`
- `warmup_ratio=0.1`
- `label_smoothing=0.1`
- `shuffle_option_augmentation=True`
- `option_order_ensemble=True`
- `num_option_order_permutations=4`
- `generation_max_new_tokens=40`
- `early_stopping_patience=5`
- `max_length=768`
- `grad_accum_steps=4`
- `val_ratio=0.1`
- `seed=42`
- `use_lora=True`
- `lora_r=32`
- `lora_alpha=32`
- `lora_dropout=0.05`
- `lora_target_modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

All fields in `TrainingConfig` can be overridden from the command line.

## Training

Change into the `code/` directory first:

```bash
cd code
```

### Single Run

```bash
python train_eval.py
```

This creates a timestamped directory under `../saved_models/checkpoint/` and saves:

- model checkpoint
- tokenizer
- `train_config.json`
- `history.json`
- `metrics.json`
- `training_curve.png`

### Example: Override Hyperparameters

```bash
python train_eval.py \
  --seed 123 \
  --learning_rate 1e-4 \
  --num_epochs 20 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --max_length 768
```

### Optional: K-Fold Training

```bash
python train_eval.py --use_kfold --num_folds 5
```

This will train one model per fold and save them inside a timestamped run directory.

## Reproducing the Final Submission Workflow

One practical workflow is to train multiple seeds and ensemble them at inference time.

```bash
cd code

python train_eval.py --seed 42
python train_eval.py --seed 123
python train_eval.py --seed 456
```

After training finishes, note the timestamped checkpoint directories and run:

```bash
python inference.py \
  --model_dirs \
    ../saved_models/checkpoint/<run_dir_1> \
    ../saved_models/checkpoint/<run_dir_2> \
    ../saved_models/checkpoint/<run_dir_3> \
  --output_csv ../saved_models/checkpoint/ensemble_submission.csv
```

## Inference

### Single-Model Inference

```bash
cd code

python inference.py \
  --model_dirs ../saved_models/checkpoint/<run_dir> \
  --output_csv ../saved_models/checkpoint/submission.csv
```

### Multi-Model Hard Voting

```bash
cd code

python inference.py \
  --model_dirs \
    ../saved_models/checkpoint/<run_dir_1> \
    ../saved_models/checkpoint/<run_dir_2> \
    ../saved_models/checkpoint/<run_dir_3> \
  --option_order_ensemble \
  --num_option_order_permutations 4 \
  --max_new_tokens 40 \
  --output_csv ../saved_models/checkpoint/ensemble_submission.csv
```

The output CSV format is:

```text
question_id,ans
1,2
2,0
...
```

## Useful Notes

- Training automatically uses GPU if available, otherwise CPU.
- `saved_models/splits/` stores reusable train/validation or k-fold split definitions.
- Validation accuracy is based on generated answers, not logits.
- Early stopping is based on validation accuracy.
- For faster iteration, you can reduce `num_option_order_permutations`, `generation_max_new_tokens`, or disable option-order ensemble during training-time validation.

## Requirements

Packages listed in `requirements.txt`:

```text
matplotlib
pandas
peft
scikit-learn
torch
transformers
tqdm
```
