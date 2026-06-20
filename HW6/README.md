# HW6 Brainrot Conditional Image Generation

This implementation is designed around the HW6 restrictions:

- The main image generator is a from-scratch conditional DDPM/DDIM model.
- No pretrained generator, pretrained UNet, pretrained Transformer, pretrained diffusion model, or `diffusers` pipeline is used.
- The model architecture, noise schedule, loss, training loop, EMA, and sampler are implemented in `src/`.
- Pretrained CLIP is not used to generate images. It is only an optional scorer dependency through the official scoring program.

## Layout

```text
HW6/
├── data/
│   ├── trainset/             # extracted from dataset.zip
│   ├── train.csv
│   └── generate.csv
├── scoring_program/
│   ├── metadata              # extracted from scoring_program.zip
│   ├── score.py              # extracted from scoring_program.zip
│   └── input/
│       ├── ref/
│       │   ├── test/
│       │   ├── test.json
│       │   ├── test_mu.npy
│       │   └── test_sigma.npy
│       └── res/              # generated PNGs for local scoring
├── scripts/
├── src/
├── checkpoints/
└── submission/
```

`scoring_program/input/ref` and `scoring_program/input/res` follow the official local scorer folder names. These directories are created by `scripts/setup_scoring.py` when the official archives are extracted. Final E3/CodaBench packaging uses `submission/HW6_{student_id}/generated_images`.

## Server Setup

From inside `HW6/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_data.py
python scripts/setup_scoring.py
```

The setup scripts only unpack the provided files:

```bash
unzip scoring_program.zip -d scoring_program
mkdir -p scoring_program/input
unzip hw6_reference.zip -d scoring_program/input/ref
mkdir -p scoring_program/input/res
```

## Training

Recommended first 4090 run:

```bash
python scripts/train.py \
  --image_dir data/trainset \
  --metadata_path data/train.csv \
  --output_dir checkpoints/ddpm_unet_4090 \
  --batch_size 128 \
  --grad_accum_steps 1 \
  --max_steps 200000 \
  --base_channels 128 \
  --channel_mults 1,2,2,4 \
  --num_res_blocks 2 \
  --attention_resolutions 16,8 \
  --lr 2e-4 \
  --condition_drop_prob 0.1 \
  --mixed_precision
```

The final EMA checkpoint is copied to:

```text
checkpoints/model_ema.pth
```

## Generate For Local Scoring

Generate the assignment images from `generate.csv` into the official local scoring result folder:

```bash
python scripts/generate.py \
  --checkpoint checkpoints/model_ema.pth \
  --conditions data/generate.csv \
  --output_dir scoring_program/input/res \
  --batch_size 64 \
  --num_steps 100 \
  --guidance_scale 2.5
```

Validate the output:

```bash
python scripts/validate_submission.py \
  --image_dir scoring_program/input/res \
  --conditions data/generate.csv
```

Run the official local scorer:

```bash
cd scoring_program
python score.py \
  --input_dir ./input \
  --output_dir ./ \
  --image_size 64 \
  --num_images 3000 \
  --test_json test.json \
  --score fid clip_t clip_i \
  --verbose
```

## Build Submission

After generating the final images:

```bash
python scripts/make_submission_zip.py \
  --student_id 314706007 \
  --generated_dir scoring_program/input/res \
  --conditions data/generate.csv \
  --checkpoint checkpoints/model_ema.pth
```

This creates:

```text
submission/HW6_314706007.zip
```

Expected package contents:

```text
HW6_314706007/
├── generated_images/
├── scripts/
├── src/
├── model.pth
├── README.md
└── requirements.txt
```

## Score Targets

Aim for:

```text
FID <= 49.2545
CLIP-T >= 0.2703
```

The model supports classifier-free guidance through condition dropout during training and `--guidance_scale` during sampling. Tune `--guidance_scale` around `1.5`, `2.0`, `2.5`, `3.0`, and `3.5`; high CLIP-T can hurt FID if images become less realistic or less diverse.
