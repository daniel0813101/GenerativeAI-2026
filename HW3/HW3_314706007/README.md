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
../data/
├── train.csv
├── dev.csv
├── test.csv
├── classes.json
└── paper_evidence/
    ├── <paper_id>.pdf
    └── ...
```

On Kaggle, mount the competition dataset and set `--data_dir` to its path (e.g. `/kaggle/input/genai-hw3/`).

## Training

```bash
python train.py --data_dir /kaggle/input/genai-hw3 --adapter_dir adapter_checkpoint
```

Key flags (all have defaults):

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `../data` | Path to the dataset folder |
| `--adapter_dir` | `adapter_checkpoint` | Where to save the LoRA adapter |
| `--epochs` | `3` | Training epochs |
| `--lora_r` | `16` | LoRA rank |
| `--max_multiplier` | `5` | Oversample cap for minority classes |

PDF text is cached to `paper_cache/` after first parse — re-runs skip parsing.

## Inference

```bash
python inference.py
```

Reads `test.csv`, runs inference using the adapter in `adapter_checkpoint/`, and writes predictions to `hw3_314706007.csv`.

If `dev.csv` is present it also prints Dev Macro F1 before the test run.

## Reproducing results

```bash
python train.py
python inference.py
```
