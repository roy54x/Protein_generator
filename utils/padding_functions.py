import numpy as np
import torch

from utils.constants import AMINO_ACID_TO_INDEX


def padd_sequence(sequence, padding_size):
    tokens = [AMINO_ACID_TO_INDEX.get(aa, 0) for aa in sequence]  # 0 for unknown amino acids
    mask = [1] * len(tokens) + [0] * (padding_size - len(tokens))
    if len(tokens) < padding_size:
        tokens += [0] * (padding_size - len(tokens))
    tokens = torch.tensor(tokens, dtype=torch.int)
    mask_tensor = torch.tensor(mask, dtype=torch.int)
    return tokens, mask_tensor

def padd_contact_map(contact_map, padding_size):
    if contact_map.shape[0] < padding_size or contact_map.shape[1] < padding_size:
        padded_contact_map = np.zeros((padding_size, padding_size))
        padded_contact_map[:contact_map.shape[0], :contact_map.shape[1]] = contact_map
        contact_map = padded_contact_map
    contact_map = torch.tensor(contact_map, dtype=torch.float)
    return contact_map