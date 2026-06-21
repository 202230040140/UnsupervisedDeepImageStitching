"""Stage StitchBench scenes into the input1/ input2/ layout expected by UDIS.

Each StitchBench scene folder contains two (or more) source images, possibly of
different resolutions and file extensions. The UDIS DataLoader expects two
parallel folders ``input1`` and ``input2`` with matching, sorted ``*.jpg`` names,
and it concatenates the two images along the channel axis -- which requires the
two images of a pair to share the same height/width. We therefore resize both
images of every scene to a common square size (default 512x512, matching the
UDIS-D training distribution) and re-encode them as JPEG.

A ``manifest.csv`` records the mapping index -> scene -> source files so the
orchestrator can reorganise results back into per-scene folders.
"""
import argparse
import csv
import os
import re
import sys

import cv2


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def list_scene_images(scene_dir):
    files = [f for f in os.listdir(scene_dir)
             if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    files.sort(key=natural_key)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True,
                        help='StitchBench category root, e.g. D:\\StitchBench\\General')
    parser.add_argument('--out', required=True,
                        help='staging output dir; input1/ input2/ and manifest.csv are written here')
    parser.add_argument('--size', type=int, default=512,
                        help='square resize side for both images of a pair (default 512)')
    args = parser.parse_args()

    input1_dir = os.path.join(args.out, 'input1')
    input2_dir = os.path.join(args.out, 'input2')
    os.makedirs(input1_dir, exist_ok=True)
    os.makedirs(input2_dir, exist_ok=True)

    scenes = [d for d in sorted(os.listdir(args.dataset), key=natural_key)
              if os.path.isdir(os.path.join(args.dataset, d))]

    manifest_rows = []
    index = 0
    skipped = []
    for scene in scenes:
        scene_dir = os.path.join(args.dataset, scene)
        imgs = list_scene_images(scene_dir)
        if len(imgs) < 2:
            skipped.append((scene, 'fewer than 2 images'))
            continue

        img1_name, img2_name = imgs[0], imgs[1]
        img1 = cv2.imread(os.path.join(scene_dir, img1_name))
        img2 = cv2.imread(os.path.join(scene_dir, img2_name))
        if img1 is None or img2 is None:
            skipped.append((scene, 'failed to read images'))
            continue

        img1 = cv2.resize(img1, (args.size, args.size))
        img2 = cv2.resize(img2, (args.size, args.size))

        index += 1
        out_name = str(index).zfill(6) + '.jpg'
        cv2.imwrite(os.path.join(input1_dir, out_name), img1)
        cv2.imwrite(os.path.join(input2_dir, out_name), img2)

        manifest_rows.append({
            'index': index,
            'staged_name': out_name,
            'scene': scene,
            'img1': img1_name,
            'img2': img2_name,
        })
        print('[{:>4}] {} -> {}  ({}, {})'.format(index, scene, out_name, img1_name, img2_name))

    manifest_path = os.path.join(args.out, 'manifest.csv')
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'staged_name', 'scene', 'img1', 'img2'])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print('\nStaged {} scene pairs into {}'.format(len(manifest_rows), args.out))
    print('Manifest: {}'.format(manifest_path))
    if skipped:
        print('Skipped {} scenes:'.format(len(skipped)))
        for s, why in skipped:
            print('  - {}: {}'.format(s, why))


if __name__ == '__main__':
    sys.exit(main())
