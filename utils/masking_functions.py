import random

import numpy as np

from constants import RANDOM_MASK_RATIO, SPAN_MASK_RATIO, MAX_SPAN_LENGTH


def apply_random_mask_on_coords(coords, mask_ratio=RANDOM_MASK_RATIO):
    """Applies independent random masking to backbone coordinates."""
    num_residues = len(coords)
    num_to_mask = max(1, int(num_residues * mask_ratio))
    mask_indices = random.sample(range(num_residues), num_to_mask)

    for idx in mask_indices:
        coords[idx] = [[[np.nan] * 3] * 4]  # Mask all 3 backbone coordinates

    return coords

def apply_span_mask_on_coords(coords, mask_ratio=SPAN_MASK_RATIO):
    """Applies span masking to backbone coordinates by replacing residues with np.nan."""
    num_residues = len(coords)
    num_to_mask = max(1, int(num_residues * mask_ratio))
    masked_indices = set()

    while len(masked_indices) < num_to_mask:
        span_length = random.randint(1, min(MAX_SPAN_LENGTH, num_residues // 2))
        start_idx = random.randint(0, num_residues - span_length)
        for i in range(start_idx, start_idx + span_length):
            if len(masked_indices) < num_to_mask:
                masked_indices.add(i)

    for idx in masked_indices:
        coords[idx] = [[[np.nan] * 3] * 4]  # Mask all 3 backbone coordinates

    return coords