"""Run UDIS on HD3D two-view pairs and write D:\\HD3D_Result-style outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduce.hd3d_eval import (
    downsample_to_megapixels,
    evaluate_raw,
    finite,
    load_lpips_metric,
    load_niqe_metric,
)
from reproduce.paths import HOMO_CKPT_DIR, RECON_CKPT_DIR, HOMO_CKPT_STEP, RECON_CKPT_STEP

METHOD = 'UDIS'
DEFAULT_MANIFEST = r'D:\HD3D_Result\_work\manifest.csv'
DEFAULT_RESULT_ROOT = r'D:\HD3D_Result'
DEFAULT_WORK_ROOT = r'D:\HD3D_Result\_work'
DEFAULT_HOMO_CKPT_DIR = HOMO_CKPT_DIR
DEFAULT_HOMO_CKPT_STEP = HOMO_CKPT_STEP
DEFAULT_RECON_CKPT_DIR = RECON_CKPT_DIR
DEFAULT_RECON_CKPT_STEP = RECON_CKPT_STEP

PER_PAIR_FIELDS = [
    'scene', 'pair_id', 'pair_name', 'method', 'status', 'failure_reason',
    'mdr', 'niqe', 'psnr', 'ssim', 'lpips', 'rmse', 'runtime_seconds',
    'valid_ratio', 'alignment_matcher', 'alignment_matches', 'alignment_inliers',
    'valid_mask_strategy', 'lpips_max_side', 'raw_path', 'aligned_path',
    'valid_mask_path', 'gt_path', 'cpp_mdr', 'cpp_warping_residual_avg',
    'cpp_warping_residual_sd', 'gt_width', 'gt_height',
]
SUMMARY_FIELDS = [
    'method', 'total_runs', 'successes', 'failures', 'failure_rate',
    'mean_mdr', 'median_mdr', 'mean_niqe', 'median_niqe',
    'mean_psnr', 'median_psnr', 'mean_ssim', 'median_ssim',
    'mean_lpips', 'median_lpips', 'mean_rmse', 'median_rmse',
    'mean_runtime', 'median_runtime',
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any) -> str:
    if value is None:
        return ''
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return '' if not math.isfinite(number) else f'{number:.5f}'


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, ensure_ascii=False), encoding='utf-8')


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except json.JSONDecodeError:
        return {}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) if isinstance(value, float) else value for key, value in row.items()})


def backup_once(path: Path, suffix: str = '.before_udis') -> None:
    if not path.exists():
        return
    backup = path.with_name('{}{}{}'.format(path.stem, suffix, path.suffix))
    if not backup.exists():
        shutil.copy2(path, backup)


def load_existing_per_pair(path: Path, method: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as handle:
        return [row for row in csv.DictReader(handle) if row.get('method') != method]


def base_row(row: dict[str, str], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        'scene': row['scene'],
        'pair_id': row['pair_id'],
        'pair_name': row['pair_name'],
        'method': METHOD,
        'status': 'failed',
        'failure_reason': '',
        'mdr': math.nan,
        'niqe': math.nan,
        'psnr': math.nan,
        'ssim': math.nan,
        'lpips': math.nan,
        'rmse': math.nan,
        'runtime_seconds': math.nan,
        'valid_ratio': math.nan,
        'alignment_matcher': '',
        'alignment_matches': '',
        'alignment_inliers': '',
        'valid_mask_strategy': '',
        'lpips_max_side': args.lpips_max_side,
        'raw_path': str(out_dir / 'raw.png'),
        'aligned_path': '',
        'valid_mask_path': '',
        'gt_path': row['gt_path'],
        'cpp_mdr': math.nan,
        'cpp_warping_residual_avg': math.nan,
        'cpp_warping_residual_sd': math.nan,
        'gt_width': '',
        'gt_height': '',
    }


def run_udis_stitch(left: str, right: str, raw_path: Path, work_dir: Path, args: argparse.Namespace) -> None:
    script = str(REPO_ROOT / 'reproduce' / 'udis_stitch_pair.py')
    if args.udis_python == 'conda':
        cmd = [
            'conda', 'run', '-n', args.udis_env, 'python', script,
            '--left', left,
            '--right', right,
            '--out', str(raw_path),
            '--work-dir', str(work_dir),
            '--gpu', args.gpu,
            '--homo-ckpt-dir', args.homo_ckpt_dir,
            '--homo-ckpt-step', args.homo_ckpt_step,
            '--recon-ckpt-dir', args.recon_ckpt_dir,
            '--recon-ckpt-step', args.recon_ckpt_step,
        ]
    else:
        cmd = [
            args.udis_python, script,
            '--left', left,
            '--right', right,
            '--out', str(raw_path),
            '--work-dir', str(work_dir),
            '--gpu', args.gpu,
            '--homo-ckpt-dir', args.homo_ckpt_dir,
            '--homo-ckpt-step', args.homo_ckpt_step,
            '--recon-ckpt-dir', args.recon_ckpt_dir,
            '--recon-ckpt-step', args.recon_ckpt_step,
        ]
    if args.max_side > 0:
        cmd.extend(['--max-side', str(args.max_side)])
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if proc.returncode != 0:
        raise RuntimeError('UDIS stitch failed:\n{}'.format(proc.stdout[-4000:]))


def cache_hit(out_dir: Path, args: argparse.Namespace) -> bool:
    status = load_json(out_dir / 'method_status.json')
    return (
        not args.force
        and status.get('success')
        and (out_dir / 'raw.png').exists()
        and (out_dir / 'metrics.json').exists()
    )


def process_pair(row: dict[str, str], result_root: Path, work_root: Path,
                 niqe_metric, lpips_metric, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = result_root / row['scene'] / 'pair_{}'.format(row['pair_id']) / METHOD
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / 'raw.png'
    metrics_path = out_dir / 'metrics.json'
    status_path = out_dir / 'method_status.json'
    pair_work = work_root / METHOD / row['pair_name']

    if cache_hit(out_dir, args):
        metrics = load_json(metrics_path)
        return {key: metrics.get(key, '') for key in PER_PAIR_FIELDS}

    result = base_row(row, out_dir, args)
    status = {
        'method': METHOD,
        'pair_name': row['pair_name'],
        'success': False,
        'runtime_seconds': None,
        'failure_reason': '',
    }
    try:
        if pair_work.exists():
            shutil.rmtree(pair_work)
        run_udis_stitch(row['left_source'], row['right_source'], raw_path, pair_work, args)

        full_raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if full_raw is None:
            raise RuntimeError('failed to read stitched output: {}'.format(raw_path))
        downsampled, raw_info = downsample_to_megapixels(full_raw, args.raw_target_megapixels)
        cv2.imwrite(str(raw_path), downsampled)

        eval_info = evaluate_raw(
            raw_path, Path(row['gt_path']), out_dir, niqe_metric, lpips_metric,
            feature_max_side=args.feature_max_side,
            min_alignment_inliers=args.min_alignment_inliers,
            min_valid_ratio=args.min_valid_ratio,
            min_niqe_side=args.min_niqe_side,
            valid_black_threshold=args.valid_black_threshold,
            lpips_max_side=args.lpips_max_side,
        )
        runtime = time.perf_counter() - started
        result.update(eval_info)
        result.update(raw_info)
        result.update({
            'status': 'success',
            'failure_reason': '',
            'runtime_seconds': runtime,
            'raw_path': str(raw_path),
            'cpp_mdr': math.nan,
            'cpp_warping_residual_avg': math.nan,
            'cpp_warping_residual_sd': math.nan,
        })
        status.update({'success': True, 'runtime_seconds': runtime, 'raw_path': str(raw_path), **raw_info})
    except Exception as exc:
        result['failure_reason'] = str(exc)
        result['runtime_seconds'] = time.perf_counter() - started
        status.update({
            'success': False,
            'failure_reason': str(exc),
            'failure_traceback': traceback.format_exc(),
            'runtime_seconds': result['runtime_seconds'],
        })

    write_json(status_path, status)
    write_json(metrics_path, result)
    return result


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    methods = []
    for row in rows:
        if row.get('method') not in methods:
            methods.append(row.get('method'))
    required = ('mdr', 'niqe', 'psnr', 'ssim', 'lpips', 'rmse', 'runtime_seconds')
    for method in methods:
        method_rows = [row for row in rows if row.get('method') == method]
        successes = [row for row in method_rows if row.get('status') == 'success'
                     and all(finite(row.get(key)) for key in required)]
        failures = len(method_rows) - len(successes)

        def values(key: str) -> list[float]:
            return [float(row[key]) for row in successes if finite(row.get(key))]

        row = {
            'method': method,
            'total_runs': len(method_rows),
            'successes': len(successes),
            'failures': failures,
            'failure_rate': failures / len(method_rows) if method_rows else math.nan,
        }
        for metric in ('mdr', 'niqe', 'psnr', 'ssim', 'lpips', 'rmse', 'runtime'):
            key = 'runtime_seconds' if metric == 'runtime' else metric
            metric_values = values(key)
            row['mean_{}'.format(metric)] = mean(metric_values) if metric_values else math.nan
            row['median_{}'.format(metric)] = median(metric_values) if metric_values else math.nan
        summary.append(row)
    return summary


def write_report(path: Path, summary_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    lines = [
        '# HD3D Two-View Stitching Report',
        '',
        'All scenes are aggregated together. MDR is the GT-alignment RANSAC reprojection RMSE in pixels. '
        'PSNR, SSIM, LPIPS, and image RMSE are computed between aligned output and GT within the valid mask.',
        '',
        '| Method | Success/Total | Failure Rate | Mean MDR | Mean NIQE | Mean PSNR | Mean SSIM | Mean LPIPS | Mean RMSE | Mean Runtime (s) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in summary_rows:
        lines.append(
            '| {method} | {successes}/{total_runs} | {failure_rate} | {mean_mdr} | {mean_niqe} | '
            '{mean_psnr} | {mean_ssim} | {mean_lpips} | {mean_rmse} | {mean_runtime} |'.format(
                method=row['method'],
                successes=row['successes'],
                total_runs=row['total_runs'],
                failure_rate=fmt(row['failure_rate']),
                mean_mdr=fmt(row['mean_mdr']),
                mean_niqe=fmt(row['mean_niqe']),
                mean_psnr=fmt(row['mean_psnr']),
                mean_ssim=fmt(row['mean_ssim']),
                mean_lpips=fmt(row['mean_lpips']),
                mean_rmse=fmt(row['mean_rmse']),
                mean_runtime=fmt(row['mean_runtime']),
            )
        )
    failures = [row for row in rows if row.get('status') != 'success']
    lines.extend(['', '## Failures', ''])
    if failures:
        lines.append('| Scene | Pair | Method | Reason |')
        lines.append('|---|---|---|---|')
        for row in failures:
            reason = str(row.get('failure_reason', '')).replace('|', '/')
            lines.append('| {} | {} | {} | {} |'.format(
                row.get('scene'), row.get('pair_id'), row.get('method'), reason))
    else:
        lines.append('None.')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def update_top_level_reports(result_root: Path, method: str, method_rows: list[dict[str, Any]]) -> None:
    per_pair_path = result_root / 'per_pair_metrics.csv'
    summary_path = result_root / 'summary_all.csv'
    report_path = result_root / 'report.md'
    for path in (per_pair_path, summary_path, report_path):
        backup_once(path)
    existing_rows = load_existing_per_pair(per_pair_path, method)
    rows = existing_rows + method_rows
    write_csv(per_pair_path, rows, PER_PAIR_FIELDS)
    summary_rows = summarize(rows)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    write_report(report_path, summary_rows, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--result-root', default=DEFAULT_RESULT_ROOT)
    parser.add_argument('--work-root', default=DEFAULT_WORK_ROOT)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--udis-env', default='udis')
    parser.add_argument('--udis-python', default='conda')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--scene', action='append')
    parser.add_argument('--pair', action='append')
    parser.add_argument('--homo-ckpt-dir', default=DEFAULT_HOMO_CKPT_DIR)
    parser.add_argument('--homo-ckpt-step', default=DEFAULT_HOMO_CKPT_STEP)
    parser.add_argument('--recon-ckpt-dir', default=DEFAULT_RECON_CKPT_DIR)
    parser.add_argument('--recon-ckpt-step', default=DEFAULT_RECON_CKPT_STEP)
    parser.add_argument('--feature-max-side', type=int, default=1800)
    parser.add_argument('--min-alignment-inliers', type=int, default=12)
    parser.add_argument('--min-valid-ratio', type=float, default=0.05)
    parser.add_argument('--min-niqe-side', type=int, default=96)
    parser.add_argument('--valid-black-threshold', type=int, default=5)
    parser.add_argument('--lpips-max-side', type=int, default=1024)
    parser.add_argument('--raw-target-megapixels', type=float, default=0.70)
    parser.add_argument('--max-side', type=int, default=0,
                        help='optional max image side before UDIS stitching (0 = native resolution)')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = read_manifest(Path(args.manifest))
    if args.scene:
        wanted = set(args.scene)
        manifest = [row for row in manifest if row['scene'] in wanted]
    if args.pair:
        wanted_pairs = set(args.pair)
        manifest = [row for row in manifest if row['pair_id'] in wanted_pairs]

    result_root = Path(args.result_root)
    work_root = Path(args.work_root)
    niqe_metric, metric_device = load_niqe_metric(args.device)
    lpips_metric, _ = load_lpips_metric(metric_device)

    rows = []
    for i, row in enumerate(manifest, 1):
        print('[{}/{}] {} {}'.format(i, len(manifest), row['pair_name'], METHOD))
        result = process_pair(row, result_root, work_root, niqe_metric, lpips_metric, args)
        rows.append(result)
        print('  -> {}'.format(result['status']))

    update_top_level_reports(result_root, METHOD, rows)
    successes = sum(1 for row in rows if row.get('status') == 'success')
    print('\nDone. {}/{} succeeded. Updated {}'.format(successes, len(rows), result_root))
    return 0 if successes == len(rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
