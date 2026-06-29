"""Stitch one image pair with the pretrained UDIS two-stage pipeline.

Stages left/right images into ``input1/000001.jpg`` and ``input2/000001.jpg``,
runs Stage-1 warp/mask generation and Stage-2 reconstruction, and writes the
result to ``--out`` (typically ``raw.png``).
"""
import argparse
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
ALIGN_CODES = os.path.join(REPO_ROOT, 'ImageAlignment', 'Codes')
RECON_CODES = os.path.join(REPO_ROOT, 'ImageReconstruction', 'Codes')

from reproduce.paths import HOMO_CKPT_DIR, RECON_CKPT_DIR, HOMO_CKPT_STEP, RECON_CKPT_STEP

DEFAULT_HOMO_CKPT_DIR = HOMO_CKPT_DIR
DEFAULT_HOMO_CKPT_STEP = HOMO_CKPT_STEP
DEFAULT_RECON_CKPT_DIR = RECON_CKPT_DIR
DEFAULT_RECON_CKPT_STEP = RECON_CKPT_STEP


def _read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('failed to read image: {}'.format(path))
    return image


def _read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError('failed to read mask: {}'.format(path))
    return mask.astype(np.float32) / 255.0


def render_natural_canvas_from_warp(warp_dir, out_path, name='000001.jpg'):
    """Blend UDIS Stage-1 natural-size warp outputs into one inspectable canvas."""
    warp1 = _read_image(os.path.join(warp_dir, 'warp1', name)).astype(np.float32)
    warp2 = _read_image(os.path.join(warp_dir, 'warp2', name)).astype(np.float32)
    mask1 = _read_mask(os.path.join(warp_dir, 'mask1', name))
    mask2 = _read_mask(os.path.join(warp_dir, 'mask2', name))
    if warp1.shape[:2] != warp2.shape[:2] or warp1.shape[:2] != mask1.shape[:2] or warp1.shape[:2] != mask2.shape[:2]:
        raise RuntimeError('UDIS Stage-1 warp/mask sizes do not match in {}'.format(warp_dir))

    mask1 = np.clip(mask1, 0.0, 1.0)
    mask2 = np.clip(mask2, 0.0, 1.0)
    denom = mask1 + mask2
    out = np.zeros_like(warp1, dtype=np.float32)
    valid = denom > 1e-3
    out[valid] = (
        warp1[valid] * mask1[valid, None] + warp2[valid] * mask2[valid, None]
    ) / denom[valid, None]
    out[~valid] = 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, np.clip(out, 0, 255).astype(np.uint8))
    return {
        'renderer': 'udis_stage1_natural_canvas',
        'width': int(out.shape[1]),
        'height': int(out.shape[0]),
    }


def run_stage(script_dir, script, env_overrides):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    proc = subprocess.run(
        [sys.executable, '-u', script],
        cwd=script_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            '{} failed (exit {}):\n{}'.format(script, proc.returncode, proc.stdout[-4000:])
        )
    return proc.stdout


def stage_inputs(left_path, right_path, input_dir, max_side=1280):
    input1_dir = os.path.join(input_dir, 'input1')
    input2_dir = os.path.join(input_dir, 'input2')
    os.makedirs(input1_dir, exist_ok=True)
    os.makedirs(input2_dir, exist_ok=True)

    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    if left is None:
        raise RuntimeError('failed to read left image: {}'.format(left_path))
    if right is None:
        raise RuntimeError('failed to read right image: {}'.format(right_path))
    if left.shape[:2] != right.shape[:2]:
        height = max(left.shape[0], right.shape[0])
        width = max(left.shape[1], right.shape[1])
        left = cv2.resize(left, (width, height))
        right = cv2.resize(right, (width, height))

    height, width = left.shape[:2]
    if max(height, width) > max_side:
        scale = max_side / float(max(height, width))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        left = cv2.resize(left, (new_width, new_height), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (new_width, new_height), interpolation=cv2.INTER_AREA)

    name = '000001.jpg'
    cv2.imwrite(os.path.join(input1_dir, name), left)
    cv2.imwrite(os.path.join(input2_dir, name), right)
    return input_dir


def stitch_pair(left_path, right_path, out_path, work_dir, gpu, homo_ckpt_dir, homo_ckpt_step,
                recon_ckpt_dir, recon_ckpt_step, max_side=0):
    os.makedirs(work_dir, exist_ok=True)
    input_dir = os.path.join(work_dir, 'input')
    warp_dir = os.path.join(work_dir, 'warp')
    pano_dir = os.path.join(work_dir, 'panorama')

    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    if os.path.exists(warp_dir):
        shutil.rmtree(warp_dir)
    if os.path.exists(pano_dir):
        shutil.rmtree(pano_dir)

    stage_inputs(left_path, right_path, input_dir, max_side=max_side or 999999)

    common = {
        'UDIS_GPU': gpu,
        'UDIS_LIMIT': '1',
        'UDIS_TEST_FOLDER': os.path.abspath(input_dir),
        'UDIS_HOMO_CKPT_DIR': homo_ckpt_dir,
        'UDIS_HOMO_CKPT_STEP': homo_ckpt_step,
    }

    run_stage(ALIGN_CODES, 'output_inference.py', dict(common, **{
        'UDIS_WARP_OUT': os.path.abspath(warp_dir),
    }))

    run_stage(RECON_CODES, 'inference.py', dict(common, **{
        'UDIS_RECON_TEST_FOLDER': os.path.abspath(warp_dir),
        'UDIS_RECON_CKPT_DIR': recon_ckpt_dir,
        'UDIS_RECON_CKPT_STEP': recon_ckpt_step,
        'UDIS_RESULT_OUT': os.path.abspath(pano_dir),
    }))

    src_pano = os.path.join(pano_dir, '000001.jpg')
    if not os.path.exists(src_pano):
        raise RuntimeError('UDIS did not produce panorama: {}'.format(src_pano))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copy2(src_pano, os.path.join(work_dir, 'udis_reconstruction_1024.jpg'))
    render_natural_canvas_from_warp(warp_dir, out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', required=True)
    parser.add_argument('--right', required=True)
    parser.add_argument('--out', required=True, help='output panorama path, e.g. raw.png')
    parser.add_argument('--work-dir', required=True, help='temporary working directory')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--homo-ckpt-dir', default=DEFAULT_HOMO_CKPT_DIR)
    parser.add_argument('--homo-ckpt-step', default=DEFAULT_HOMO_CKPT_STEP)
    parser.add_argument('--recon-ckpt-dir', default=DEFAULT_RECON_CKPT_DIR)
    parser.add_argument('--recon-ckpt-step', default=DEFAULT_RECON_CKPT_STEP)
    parser.add_argument('--max-side', type=int, default=0,
                        help='optional max image side before stitching (0 = no downscale)')
    args = parser.parse_args()

    stitch_pair(
        args.left, args.right, os.path.abspath(args.out), os.path.abspath(args.work_dir),
        args.gpu, args.homo_ckpt_dir, args.homo_ckpt_step,
        args.recon_ckpt_dir, args.recon_ckpt_step,
        max_side=args.max_side,
    )
    print('Wrote', args.out)


if __name__ == '__main__':
    main()
