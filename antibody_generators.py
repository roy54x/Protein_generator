import os
import random

import torch

from constants import MAIN_DIR, PRETRAINED_MODEL_PATH
from evodiff.pretrained import OA_DM_38M, D3PM_BLOSUM_38M
from evodiff.generate import generate_oaardm, generate_d3pm
from strategies.sequence_diffusion import SequenceDiffusion


def generate_EvoDiff(min_len, max_len, finetuned=True, batch_size=1, device="cpu"):
    seq_len = random.randint(min_len, max_len)
    if finetuned:
        strategy = SequenceDiffusion()
        model_path = os.path.join(MAIN_DIR, PRETRAINED_MODEL_PATH)
        strategy.load_state_dict(torch.load(model_path))
        strategy.eval()
        model = strategy.model.to(device)
        tokenizer = strategy.tokenizer
    else:
        checkpoint = OA_DM_38M()
        model, _, tokenizer, _ = checkpoint

    _, generated_sequence = generate_oaardm(model, tokenizer, seq_len, batch_size=batch_size, device=device)
    return generated_sequence[0]
