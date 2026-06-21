# Reproducing UDIS (inference only)

This repo can be reproduced in **inference-only** mode using the released
pretrained models, so no training is required. The two-stage pipeline is run
with the pretrained Stage-1 homography model and Stage-2 reconstruction model,
and evaluated on:

- **UDIS-D testing** (1106 pairs) -- sanity check against the paper's PSNR/SSIM table.
- **StitchBench / General** -- overlap-region PSNR/SSIM (UDIS paper protocol) plus generated panoramas.

All generated panoramas, warped images, masks, metrics, and logs are written
under `outputs/`, which is git-ignored.

## Pretrained models

Bundled inside this repository (override with CLI flags if elsewhere):

| Stage | Checkpoint |
|-------|------------|
| 1 (homography)     | `ImageAlignment/Codes/checkpoints_homo/model.ckpt-1000000` |
| 2 (reconstruction) | `ImageReconstruction/Codes/checkpoints/model.ckpt-200000` |

TensorLayer / VGG19 weights are **not** needed for inference (they are only used
by the Stage-2 training loss; the import is made lazy).

## Environment

The original code targets TensorFlow 1.x. On Windows the cleanest route is conda
(it bundles the CUDA 10 runtime, so the host driver version does not matter):

```powershell
conda create -y -n udis python=3.7
conda install -y -n udis -c conda-forge tensorflow-gpu=1.15
conda install -y -n udis -c conda-forge "scikit-image=0.16.2" opencv "tensorflow-estimator=1.15"
```

See `requirements-udis.txt` for the exact verified versions.

## Run: UDIS-D testing

```powershell
conda run -n udis python reproduce\run_udis.py `
  --name udis_d_test `
  --input-dir D:\UDIS-D\testing `
  --gpu 0
```

## Run: StitchBench / General

First stage the scene pairs into the `input1/`/`input2/` layout the code expects
(both images of a pair are resized to a common 512x512 to match UDIS-D), then run:

```powershell
conda run -n udis python reproduce\prepare_stitchbench.py `
  --dataset D:\StitchBench\General `
  --out outputs\stitchbench_general\_input

conda run -n udis python reproduce\run_udis.py `
  --name stitchbench_general `
  --input-dir outputs\stitchbench_general\_input `
  --manifest outputs\stitchbench_general\_input\manifest.csv `
  --gpu 0
```

> Note: the test set is used only for evaluation. No fine-tuning/training is done
> on StitchBench, in keeping with the requirement to not train on the target test set.

## Run: HD3D Dataset

Uses the existing manifest at ``D:\HD3D_Result\_work\manifest.csv`` (13 scenes × 6 pairs = 78 runs).
Results are written to ``D:\HD3D_Result\<scene>\pair_<id>\UDIS\`` and merged into the top-level
``per_pair_metrics.csv``, ``summary_all.csv``, and ``report.md`` (method name ``UDIS``).

```powershell
python reproduce\run_hd3d.py --gpu 0
```

Retry a single pair (e.g. after a TF error on a difficult scene):

```powershell
python reproduce\run_hd3d.py --scene Outdoor_005 --pair 13 --max-side 1280 --force --gpu 0
```

Evaluation uses the same GT-alignment protocol as ``NIS_depths`` (MDR / NIQE / PSNR / SSIM / LPIPS
within the valid mask; ``raw.png`` downsampled to 0.7 MP before metrics).

## Smoke test

Add `--limit N` to process only the first N pairs end-to-end, e.g. `--limit 3`.

## Output layout (per run)

```
outputs/<name>/
  <scene>/
    panorama.png      stitched result
    reference.png     resized input1 (reference)
    target.png        resized input2 (warped onto reference)
    metrics.json      overlap PSNR/SSIM for this pair
  metrics.csv         one row per scene
  summary.json        averages + top 30% / 30-60% / 60-100% PSNR & SSIM
  _warp/              intermediate warp1/warp2/mask1/mask2
  _panorama/          flat panorama outputs (00000i.jpg)
  _align_metrics.json raw per-pair metrics from Stage 1
  _logs/              per-stage stdout logs
```

## Caveats

- `ImageReconstruction/ImageReconstruction.md` notes the released reconstruction
  model is a retrain, so UDIS-D numbers should be close to (but may not exactly
  match) the paper table.
- StitchBench pairs often have large parallax / very different source resolutions.
  UDIS (homography + reconstruction, trained on UDIS-D) will produce imperfect
  panoramas on some of those scenes; metrics are reported as-is.
