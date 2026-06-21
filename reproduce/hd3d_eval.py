"""Evaluate a stitched panorama against HD3D GT (same protocol as NIS_depths run_hd3d)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def downsample_to_megapixels(image: np.ndarray, target_megapixels: float) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image.shape[:2]
    current_pixels = float(height * width)
    target_pixels = float(target_megapixels) * 1_000_000.0
    if target_pixels <= 0.0 or current_pixels <= target_pixels:
        return image, {
            'raw_target_megapixels': target_megapixels,
            'raw_original_width': width,
            'raw_original_height': height,
            'raw_width': width,
            'raw_height': height,
            'raw_scale': 1.0,
        }
    scale = math.sqrt(target_pixels / current_pixels)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, {
        'raw_target_megapixels': target_megapixels,
        'raw_original_width': width,
        'raw_original_height': height,
        'raw_width': new_width,
        'raw_height': new_height,
        'raw_scale': scale,
    }


def canvas_valid_mask(image: np.ndarray, black_threshold: int) -> np.ndarray:
    near_black = (np.max(image, axis=2) <= black_threshold).astype(np.uint8)
    flood = near_black.copy()
    height, width = flood.shape
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    def fill_if_background(x: int, y: int) -> None:
        if flood[y, x] == 1:
            cv2.floodFill(flood, flood_mask, (x, y), 2)

    for x in range(width):
        fill_if_background(x, 0)
        fill_if_background(x, height - 1)
    for y in range(height):
        fill_if_background(0, y)
        fill_if_background(width - 1, y)
    return (~(flood == 2)).astype(np.uint8) * 255


def feature_image(image: np.ndarray, mask: np.ndarray | None, max_side: int):
    height, width = image.shape[:2]
    scale = 1.0
    if max(height, width) > max_side:
        scale = max_side / float(max(height, width))
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return image, mask, scale


def make_detector(name: str):
    if name == 'sift' and hasattr(cv2, 'SIFT_create'):
        return cv2.SIFT_create(nfeatures=10000), cv2.NORM_L2, 0.75
    if name == 'orb':
        return cv2.ORB_create(nfeatures=12000, fastThreshold=7), cv2.NORM_HAMMING, 0.80
    raise RuntimeError('Unsupported detector: {}'.format(name))


def estimate_output_to_gt(raw: np.ndarray, gt: np.ndarray, max_side: int, min_inliers: int, black_threshold: int):
    raw_mask = canvas_valid_mask(raw, black_threshold)
    gt_mask = np.full(gt.shape[:2], 255, dtype=np.uint8)
    raw_small, raw_mask_small, raw_scale = feature_image(raw, raw_mask, max_side)
    gt_small, gt_mask_small, gt_scale = feature_image(gt, gt_mask, max_side)
    raw_gray = cv2.cvtColor(raw_small, cv2.COLOR_BGR2GRAY)
    gt_gray = cv2.cvtColor(gt_small, cv2.COLOR_BGR2GRAY)
    errors = []

    for name in ('sift', 'orb'):
        try:
            detector, norm, ratio = make_detector(name)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        raw_keypoints, raw_desc = detector.detectAndCompute(raw_gray, raw_mask_small)
        gt_keypoints, gt_desc = detector.detectAndCompute(gt_gray, gt_mask_small)
        if raw_desc is None or gt_desc is None or len(raw_keypoints) < 4 or len(gt_keypoints) < 4:
            errors.append('{}: insufficient keypoints'.format(name))
            continue
        matcher = cv2.BFMatcher(norm)
        raw_matches = matcher.knnMatch(raw_desc, gt_desc, k=2)
        good = []
        for item in raw_matches:
            if len(item) < 2:
                continue
            first, second = item
            if first.distance < ratio * second.distance:
                good.append(first)
        if len(good) < 4:
            errors.append('{}: insufficient matches ({})'.format(name, len(good)))
            continue
        raw_pts = np.float32([raw_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) / raw_scale
        gt_pts = np.float32([gt_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2) / gt_scale
        homography, inlier_mask = cv2.findHomography(raw_pts, gt_pts, cv2.RANSAC, 5.0, maxIters=8000, confidence=0.995)
        if homography is None or inlier_mask is None:
            errors.append('{}: homography failed'.format(name))
            continue
        inliers = inlier_mask.ravel().astype(bool)
        inlier_count = int(inliers.sum())
        if inlier_count < min_inliers:
            errors.append('{}: insufficient inliers ({})'.format(name, inlier_count))
            continue
        projected = cv2.perspectiveTransform(raw_pts[inliers], homography)
        residuals = np.linalg.norm(projected.reshape(-1, 2) - gt_pts[inliers].reshape(-1, 2), axis=1)
        mdr = float(math.sqrt(float(np.mean(residuals * residuals))))
        return homography, {
            'alignment_matcher': name,
            'alignment_matches': len(good),
            'alignment_inliers': inlier_count,
            'mdr': mdr,
        }
    raise RuntimeError('; '.join(errors) if errors else 'GT alignment failed')


def valid_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def masked_crop_pair(aligned, gt, mask):
    bbox = valid_bbox(mask)
    if bbox is None:
        raise RuntimeError('empty valid mask')
    x0, y0, x1, y1 = bbox
    crop_mask = mask[y0:y1, x0:x1] > 0
    aligned_crop = aligned[y0:y1, x0:x1].copy()
    gt_crop = gt[y0:y1, x0:x1].copy()
    aligned_crop[~crop_mask] = gt_crop[~crop_mask]
    return aligned_crop, gt_crop, crop_mask


def masked_niqe_crop(aligned: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bbox = valid_bbox(mask)
    if bbox is None:
        raise RuntimeError('empty valid mask')
    x0, y0, x1, y1 = bbox
    crop = aligned[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1] > 0
    if np.any(crop_mask) and np.any(~crop_mask):
        fill = np.median(crop[crop_mask], axis=0).astype(np.uint8)
        crop[~crop_mask] = fill
    return crop


def resize_max_side(image: np.ndarray, max_side: int, interpolation: int) -> np.ndarray:
    if max_side <= 0:
        return image
    height, width = image.shape[:2]
    if max(height, width) <= max_side:
        return image
    scale = max_side / float(max(height, width))
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)


def compute_reference_metrics(aligned, gt, mask, lpips_metric, out_dir: Path, lpips_max_side: int):
    valid = mask > 0
    if not np.any(valid):
        raise RuntimeError('empty valid mask')
    diff = aligned.astype(np.float32) - gt.astype(np.float32)
    mse = float(np.mean((diff[valid]) ** 2))
    rmse = float(math.sqrt(mse))
    psnr = float('inf') if mse <= 0 else float(20.0 * math.log10(255.0 / rmse))

    aligned_crop, gt_crop, crop_mask = masked_crop_pair(aligned, gt, mask)
    min_side = min(aligned_crop.shape[:2])
    if min_side < 3:
        raise RuntimeError('SSIM crop too small: {}x{}'.format(aligned_crop.shape[1], aligned_crop.shape[0]))
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    win_size = max(3, win_size)
    gt_rgb = cv2.cvtColor(gt_crop, cv2.COLOR_BGR2RGB)
    aligned_rgb = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2RGB)
    try:
        _, ssim_map = structural_similarity(
            gt_rgb, aligned_rgb, channel_axis=2, data_range=255, win_size=win_size, full=True,
        )
    except TypeError:
        _, ssim_map = structural_similarity(
            gt_rgb, aligned_rgb, multichannel=True, data_range=255, win_size=win_size, full=True,
        )
    if ssim_map.ndim == 3:
        ssim_map = np.mean(ssim_map, axis=2)
    eval_mask = cv2.erode(crop_mask.astype(np.uint8), np.ones((win_size, win_size), dtype=np.uint8), iterations=1) > 0
    if not np.any(eval_mask):
        eval_mask = crop_mask
    ssim = float(np.mean(ssim_map[eval_mask]))

    lpips_aligned = resize_max_side(aligned_crop, lpips_max_side, cv2.INTER_AREA)
    lpips_gt = resize_max_side(gt_crop, lpips_max_side, cv2.INTER_AREA)
    lpips_aligned_path = out_dir / '_lpips_aligned_tmp.png'
    lpips_gt_path = out_dir / '_lpips_gt_tmp.png'
    cv2.imwrite(str(lpips_aligned_path), lpips_aligned)
    cv2.imwrite(str(lpips_gt_path), lpips_gt)
    try:
        score = lpips_metric(str(lpips_aligned_path), str(lpips_gt_path))
        lpips_score = float(score.detach().cpu().item()) if hasattr(score, 'detach') else float(score)
    finally:
        for tmp_path in (lpips_aligned_path, lpips_gt_path):
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return {'psnr': psnr, 'ssim': ssim, 'lpips': lpips_score, 'rmse': rmse}


def evaluate_raw(raw_path: Path, gt_path: Path, out_dir: Path, niqe_metric, lpips_metric,
                 feature_max_side=1800, min_alignment_inliers=12, min_valid_ratio=0.05,
                 min_niqe_side=96, valid_black_threshold=5, lpips_max_side=1024):
    raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
    if raw is None:
        raise RuntimeError('failed to read raw image: {}'.format(raw_path))
    if gt is None:
        raise RuntimeError('failed to read gt image: {}'.format(gt_path))

    raw_mask = canvas_valid_mask(raw, valid_black_threshold)
    homography, alignment = estimate_output_to_gt(
        raw, gt, feature_max_side, min_alignment_inliers, valid_black_threshold)
    aligned = cv2.warpPerspective(raw, homography, (gt.shape[1], gt.shape[0]))
    mask = cv2.warpPerspective(raw_mask, homography, (gt.shape[1], gt.shape[0]), flags=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8) * 255
    valid_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if valid_ratio < min_valid_ratio:
        raise RuntimeError('valid area too small: {:.5f}'.format(valid_ratio))

    niqe_crop = masked_niqe_crop(aligned, mask)
    if min(niqe_crop.shape[:2]) < min_niqe_side:
        raise RuntimeError('NIQE crop too small')
    niqe_crop_path = out_dir / '_niqe_crop_tmp.png'
    cv2.imwrite(str(niqe_crop_path), niqe_crop)
    try:
        score = niqe_metric(str(niqe_crop_path))
        niqe = float(score.detach().cpu().item()) if hasattr(score, 'detach') else float(score)
    finally:
        try:
            niqe_crop_path.unlink()
        except OSError:
            pass

    aligned_path = out_dir / 'aligned_to_gt.png'
    mask_path = out_dir / 'valid_mask.png'
    cv2.imwrite(str(aligned_path), aligned)
    cv2.imwrite(str(mask_path), mask)
    reference_metrics = compute_reference_metrics(
        aligned, gt, mask, lpips_metric, out_dir, lpips_max_side)
    return {
        **alignment,
        **reference_metrics,
        'niqe': niqe,
        'valid_ratio': valid_ratio,
        'aligned_path': str(aligned_path),
        'valid_mask_path': str(mask_path),
        'valid_mask_strategy': 'edge_connected_black_canvas',
        'lpips_max_side': lpips_max_side,
        'gt_width': int(gt.shape[1]),
        'gt_height': int(gt.shape[0]),
    }


def load_niqe_metric(device: str):
    import pyiqa
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    return pyiqa.create_metric('niqe', device=device), device


def load_lpips_metric(device: str):
    import pyiqa
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    return pyiqa.create_metric('lpips', device=device), device
