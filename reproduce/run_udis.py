"""End-to-end UDIS inference orchestrator (NIS_depths-style output layout).

For one dataset (a folder containing ``input1/`` and ``input2/`` sub-folders) this
runs the full two-stage UDIS pipeline using the pretrained models and writes
results under ``--out-root/<name>/``:

  <name>/
    <scene>/
      panorama.png          stitched result
      reference.png         resized input1 (the reference image)
      target.png            resized input2 (the image warped onto the reference)
      metrics.json          overlap PSNR/SSIM for this pair
    metrics.csv             one row per scene
    summary.json            averages + paper-style top 30 / 30-60 / 60-100 splits

Stage scripts are invoked as subprocesses (same Python interpreter / conda env)
with their working directory set to the corresponding ``Codes`` folder, and are
configured entirely through environment variables (see the two constant.py files).
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import cv2


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


def run_stage(script_dir, script, env_overrides, log_path):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    print('\n>>> running {} (cwd={})'.format(script, script_dir))
    for k, v in env_overrides.items():
        print('    {}={}'.format(k, v))
    with open(log_path, 'w', encoding='utf-8') as log:
        proc = subprocess.Popen(
            [sys.executable, '-u', script],
            cwd=script_dir, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError('{} failed with exit code {} (see {})'.format(script, proc.returncode, log_path))


def load_manifest(manifest_path):
    """index (1-based) -> dict(scene, img1, img2)."""
    mapping = {}
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, newline='') as f:
            for row in csv.DictReader(f):
                mapping[int(row['index'])] = row
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True, help='run name, e.g. udis_d_test or stitchbench_general')
    parser.add_argument('--input-dir', required=True, help='folder containing input1/ and input2/')
    parser.add_argument('--out-root', default=os.path.join(REPO_ROOT, 'outputs'),
                        help='root output dir (default: <repo>/outputs)')
    parser.add_argument('--manifest', default='', help='optional manifest.csv for scene names (StitchBench)')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--limit', type=int, default=0, help='cap number of pairs (0 = all), for smoke tests')
    parser.add_argument('--homo-ckpt-dir', default=DEFAULT_HOMO_CKPT_DIR)
    parser.add_argument('--homo-ckpt-step', default=DEFAULT_HOMO_CKPT_STEP)
    parser.add_argument('--recon-ckpt-dir', default=DEFAULT_RECON_CKPT_DIR)
    parser.add_argument('--recon-ckpt-step', default=DEFAULT_RECON_CKPT_STEP)
    parser.add_argument('--skip-stage1', action='store_true', help='reuse existing warp/mask + align metrics')
    parser.add_argument('--skip-stage2', action='store_true', help='reuse existing panoramas')
    args = parser.parse_args()

    # Stage scripts run with a different cwd (their Codes folder), so every path
    # handed to them must be absolute.
    args.input_dir = os.path.abspath(args.input_dir)
    args.out_root = os.path.abspath(args.out_root)
    if args.manifest:
        args.manifest = os.path.abspath(args.manifest)

    work_dir = os.path.join(args.out_root, args.name)
    warp_dir = os.path.join(work_dir, '_warp')
    panorama_dir = os.path.join(work_dir, '_panorama')
    align_metrics_path = os.path.join(work_dir, '_align_metrics.json')
    log_dir = os.path.join(work_dir, '_logs')
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    common_env = {
        'UDIS_GPU': args.gpu,
        'UDIS_LIMIT': args.limit,
    }

    if not args.skip_stage1:
        # Stage 1a: alignment metrics (overlap PSNR/SSIM)
        run_stage(ALIGN_CODES, 'inference.py', dict(common_env, **{
            'UDIS_TEST_FOLDER': args.input_dir,
            'UDIS_HOMO_CKPT_DIR': args.homo_ckpt_dir,
            'UDIS_HOMO_CKPT_STEP': args.homo_ckpt_step,
            'UDIS_METRICS_OUT': align_metrics_path,
        }), os.path.join(log_dir, 'stage1_metrics.log'))

        # Stage 1b: generate warp1/warp2/mask1/mask2
        run_stage(ALIGN_CODES, 'output_inference.py', dict(common_env, **{
            'UDIS_TEST_FOLDER': args.input_dir,
            'UDIS_HOMO_CKPT_DIR': args.homo_ckpt_dir,
            'UDIS_HOMO_CKPT_STEP': args.homo_ckpt_step,
            'UDIS_WARP_OUT': warp_dir,
        }), os.path.join(log_dir, 'stage1_warp.log'))

    if not args.skip_stage2:
        # Stage 2: reconstruction -> panoramas
        run_stage(RECON_CODES, 'inference.py', dict(common_env, **{
            'UDIS_RECON_TEST_FOLDER': warp_dir,
            'UDIS_RECON_CKPT_DIR': args.recon_ckpt_dir,
            'UDIS_RECON_CKPT_STEP': args.recon_ckpt_step,
            'UDIS_RESULT_OUT': panorama_dir,
        }), os.path.join(log_dir, 'stage2_recon.log'))

    # ---- assemble per-scene results -------------------------------------
    print('\n>>> assembling per-scene outputs')
    with open(align_metrics_path) as f:
        align = json.load(f)
    per_pair = {p['index']: p for p in align['per_pair']}
    manifest = load_manifest(args.manifest)

    rows = []
    for index in sorted(per_pair.keys()):
        pair = per_pair[index]
        if index in manifest:
            scene = manifest[index]['scene']
            img1 = manifest[index].get('img1', '')
            img2 = manifest[index].get('img2', '')
        else:
            scene = os.path.splitext(pair['name'])[0]
            img1 = pair['name']
            img2 = pair['name']

        scene_dir = os.path.join(work_dir, scene)
        os.makedirs(scene_dir, exist_ok=True)

        # panorama
        src_pano = os.path.join(panorama_dir, str(index).zfill(6) + '.jpg')
        pano_dst = os.path.join(scene_dir, 'panorama.png')
        pano_ok = os.path.exists(src_pano)
        if pano_ok:
            cv2.imwrite(pano_dst, cv2.imread(src_pano))

        # copy the reference/target inputs for visual comparison. input1 and
        # input2 share the same filename (true for both UDIS-D and the staged
        # StitchBench pairs), so the Stage-1 input1 basename locates both.
        for src_name, dst_name in (('input1', 'reference.png'), ('input2', 'target.png')):
            src_img = os.path.join(args.input_dir, src_name, pair['name'])
            if os.path.exists(src_img):
                cv2.imwrite(os.path.join(scene_dir, dst_name), cv2.imread(src_img))

        metrics = {
            'index': index,
            'scene': scene,
            'img1': img1,
            'img2': img2,
            'overlap_psnr': pair['psnr'],
            'overlap_ssim': pair['ssim'],
            'panorama': 'panorama.png' if pano_ok else None,
        }
        with open(os.path.join(scene_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        rows.append(metrics)

    # metrics.csv
    csv_path = os.path.join(work_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'scene', 'img1', 'img2',
                                               'overlap_psnr', 'overlap_ssim', 'panorama'])
        writer.writeheader()
        writer.writerows(rows)

    # summary.json
    summary = dict(align['summary'])
    summary['name'] = args.name
    summary['input_dir'] = args.input_dir
    summary['homo_ckpt'] = os.path.join(args.homo_ckpt_dir, 'model.ckpt-' + args.homo_ckpt_step)
    summary['recon_ckpt'] = os.path.join(args.recon_ckpt_dir, 'model.ckpt-' + args.recon_ckpt_step)
    summary['num_scenes'] = len(rows)
    summary['num_panoramas'] = sum(1 for r in rows if r['panorama'])
    with open(os.path.join(work_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\nDone. Results in {}'.format(work_dir))
    print('  scenes      : {}'.format(len(rows)))
    print('  panoramas   : {}'.format(summary['num_panoramas']))
    print('  avg PSNR    : {:.4f}'.format(summary['psnr_average']))
    print('  avg SSIM    : {:.4f}'.format(summary['ssim_average']))
    print('  metrics.csv : {}'.format(csv_path))


if __name__ == '__main__':
    sys.exit(main())
