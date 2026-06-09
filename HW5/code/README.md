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

Flip-pair latent cache experiment:

```bash
python HW5/code/train.py \
  --mixed_precision \
  --rebuild_cache \
  --cache_flip_pairs \
  --latent_cache HW5/model/cache/latents_flip_pairs.pt \
  --max_steps 150000 \
  --save_every 15000 \
  --output_dir HW5/model/flip_150k
```

This caches both original and horizontally flipped VAE latents, doubling the training latent count while keeping the baseline cache separate.

Large U-Net experiment without flip augmentation:

```bash
python HW5/code/train.py \
  --mixed_precision \
  --unet_channels 128,256,512,768 \
  --max_steps 150000 \
  --save_every 15000 \
  --output_dir HW5/model/unet_large_150k
```

Mild augmentation fine-tuning experiment:

```bash
python HW5/code/train.py \
  --mixed_precision \
  --unet_channels 192,384,512,768 \
  --init_unet_checkpoint HW5/model/unet_large_150k/unet_ema_step_0150000 \
  --rebuild_cache \
  --cache_mild_augments \
  --latent_cache HW5/model/cache/latents_mild_aug.pt \
  --lr 5e-5 \
  --max_steps 75000 \
  --save_every 15000 \
  --output_dir HW5/model/unet_large_mild_aug_ft
```

This caches original, mild color-jittered, and mild affine VAE latents for each image, then fine-tunes from the best large U-Net checkpoint.

Large U-Net v-prediction experiment:

```bash
python HW5/code/train.py \
  --mixed_precision \
  --unet_channels 192,384,512,768 \
  --prediction_type v_prediction \
  --latent_cache HW5/model/cache/latents.pt \
  --max_steps 150000 \
  --save_every 15000 \
  --output_dir HW5/model/unet_large_vpred_150k
```

When generating from a v-prediction checkpoint, pass the matching prediction type:

```bash
python HW5/code/inference.py \
  --checkpoint_dir HW5/model/unet_large_vpred_150k/unet_ema_step_0150000 \
  --output_dir HW5/scoring_program/input/res \
  --num_samples 3000 \
  --sampler ddim \
  --num_inference_steps 250 \
  --prediction_type v_prediction
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
