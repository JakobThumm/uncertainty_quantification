#!/usr/bin/env python3
"""
Convert existing preprocessed H36M float32 images to uint8 in-place.

Processes one file at a time to avoid needing double the disk space:
  1. Load float32 .npy file
  2. Denormalize to uint8
  3. Save uint8 .npy file over the original
  4. Move on to the next file

Normalization applied during preprocessing:
    float_image = uint8_image / 255.0 + NORMALIZATION_OFFSET
Inverse:
    uint8_image = round((float_image - NORMALIZATION_OFFSET) * 255)
                = round((float_image + [0.406, 0.457, 0.480]) * 255)
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from human_pose_pipeline.pose_estimation.h36m_settings import NORMALIZATION_OFFSET

# Positive mean: NORMALIZATION_OFFSET = [-0.406, -0.457, -0.480], so mean = [0.406, 0.457, 0.480]
NORM_MEAN = np.array([-o for o in NORMALIZATION_OFFSET], dtype=np.float32).reshape(1, 3, 1, 1)


def convert_file(npy_path: str, dry_run: bool = False) -> dict:
    """
    Convert a single float32 .npy image file to uint8 in-place.

    Returns a dict with 'original_gb', 'new_gb', 'skipped' keys.
    """
    # mmap_mode='r' reads only the header — no data loaded yet
    try:
        data = np.load(npy_path, mmap_mode='r')
    except Exception as e:
        print(e)
        return {'original_gb': 0, 'new_gb': 0, 'skipped': True}

    if data.dtype == np.uint8:
        return {'original_gb': 0, 'new_gb': 0, 'skipped': True}

    if data.dtype not in (np.float32, np.float16):
        print(f"  WARNING: unexpected dtype {data.dtype} in {npy_path}, skipping")
        return {'original_gb': 0, 'new_gb': 0, 'skipped': True}

    original_gb = data.nbytes / 1e9
    new_gb = data.size / 1e9  # uint8: 1 byte per element

    if dry_run:
        return {'original_gb': original_gb, 'new_gb': new_gb, 'skipped': False}

    # Load fully into RAM for conversion (mmap can't be written over itself)
    data = np.array(data, dtype=np.float32)
    uint8_data = np.clip(np.round((data + NORM_MEAN) * 255), 0, 255).astype(np.uint8)
    np.save(npy_path, uint8_data)

    return {'original_gb': original_gb, 'new_gb': new_gb, 'skipped': False}


def find_image_files(preprocessed_dir: str):
    """Yield all .npy paths inside PreprocessedImages subdirectories."""
    for root, dirs, files in os.walk(preprocessed_dir):
        if os.path.basename(root) == 'PreprocessedImages':
            for fname in sorted(files):
                if fname.endswith('.npy'):
                    yield os.path.join(root, fname)


def main():
    parser = argparse.ArgumentParser(description='Convert preprocessed H36M images from float32 to uint8 in-place')
    parser.add_argument('--preprocessed_dir', type=str,
                        default='datasets/H36M/pre_processed',
                        help='Path to preprocessed dataset directory')
    parser.add_argument('--dry_run', action='store_true',
                        help='Report what would be done without modifying files')
    args = parser.parse_args()

    preprocessed_dir = args.preprocessed_dir
    if not os.path.isabs(preprocessed_dir):
        preprocessed_dir = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
            preprocessed_dir
        )

    if not os.path.exists(preprocessed_dir):
        print(f"ERROR: directory not found: {preprocessed_dir}")
        sys.exit(1)

    npy_files = list(find_image_files(preprocessed_dir))
    if not npy_files:
        print("No .npy image files found.")
        sys.exit(0)

    print(f"Found {len(npy_files)} image files in {preprocessed_dir}")
    if args.dry_run:
        print("DRY RUN — no files will be modified\n")

    total_original_gb = 0.0
    total_new_gb = 0.0
    skipped = 0
    converted = 0

    for npy_path in tqdm(npy_files, desc='Converting', unit='file'):
        result = convert_file(npy_path, dry_run=args.dry_run)
        if result['skipped']:
            skipped += 1
        else:
            total_original_gb += result['original_gb']
            total_new_gb += result['new_gb']
            converted += 1

    print(f"\nDone.")
    print(f"  Converted : {converted} files")
    print(f"  Skipped   : {skipped} files (already uint8 or unexpected dtype)")
    print(f"  Space before : {total_original_gb:.1f} GB")
    print(f"  Space after  : {total_new_gb:.1f} GB")
    print(f"  Saved        : {total_original_gb - total_new_gb:.1f} GB")


if __name__ == '__main__':
    main()
