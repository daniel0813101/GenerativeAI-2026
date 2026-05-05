# HW3 — Hallucination Type Classification in Peer Reviews

Student ID: 314706007

## Environment

Recommended: Kaggle (dual T4, 30 GB VRAM). Google Colab Pro also works.

```bash
pip install -r requirements.txt
```

## Data placement

Expected layout relative to this directory:

```
../dataset/
├── train.csv
├── dev.csv
├── test.csv
├── classes.json
└── paper_evidence/
    ├── <paper_id>.pdf
    └── ...
```

`python inference.py` uses this `../dataset/` path by default. On Kaggle, either place/copy the competition dataset at `../dataset/`, or pass `--data_dir` explicitly.

## Training

```bash
python train.py --data_dir /kaggle/input/genai-hw3 --adapter_dir adapter_checkpoint
```

Key flags (all have defaults):

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `../dataset` | Path to the dataset folder |
| `--adapter_dir` | `adapter_checkpoint` | Where to save the LoRA adapter |
| `--epochs` | `3` | Training epochs |
| `--lora_r` | `32` | LoRA rank |
| `--max_multiplier` | `10` | Oversample cap for minority classes |

PDF text is cached to `paper_cache/` after first parse — re-runs skip parsing.

## Inference

```bash
python inference.py
```

With no arguments, this runs **test only**: it reads `../dataset/test.csv`, loads the latest timestamped adapter under `adapter_checkpoint/`, and writes predictions to `hw3_314706007.csv`.

Useful validation modes:

```bash
# Dev-only validation
python inference.py --dev_only --no-test_only

# Evaluate dev first, then write test predictions
python inference.py --no-test_only

# Three-seed ensemble validation
python inference.py --dev_only --no-test_only --ensemble_dirs adapter_seed42,adapter_seed43,adapter_seed44

# Final single-adapter test-only run
python inference.py

# Final three-seed ensemble test-only run
python inference.py --ensemble_dirs adapter_seed42,adapter_seed43,adapter_seed44
```

Train the three seed adapters separately:

```bash
python train.py --data_dir ../dataset --adapter_dir adapter_seed42 --seed 42
python train.py --data_dir ../dataset --adapter_dir adapter_seed43 --seed 43
python train.py --data_dir ../dataset --adapter_dir adapter_seed44 --seed 44
```

## Reproducing results

```bash
python train.py
python inference.py
```
