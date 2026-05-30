# HW5 Latent Diffusion Baseline

This implementation follows the assignment restriction: the only pretrained model used by training and inference is `stabilityai/sd-vae-ft-mse` for VAE encode/decode. The denoising U-Net is initialized from scratch.

## Train

```bash
python HW5/code/train.py --mixed_precision
```

Useful lower-memory variant:

```bash
python HW5/code/train.py --batch_size 16 --grad_accum_steps 4 --mixed_precision
```

The final EMA checkpoint is saved to:

```text
HW5/model/baseline_latent_ddpm/unet_ema_final
```

## Generate

```bash
python HW5/code/inference.py \
  --checkpoint_dir HW5/model/baseline_latent_ddpm/unet_ema_final \
  --output_dir HW5/scoring_program/input/res \
  --num_samples 3000 \
  --sampler ddim \
  --num_inference_steps 250
```

By default, inference writes to `HW5/scoring_program/input/res`, which is the official scoring directory. It also removes old PNG files from that directory before generation so the scorer sees exactly 3000 generated images.

For the final submission, try several seeds and samplers locally. Keep exactly 3000 PNG files at 256x256.

```bash
cd HW5/scoring_program
python score.py --input_dir ./input --output_dir ./ --image_size 256 --num_images 3000 --verbose
```

The official scorer writes `HW5/scoring_program/scores.json`.
