import random

from evodiff.pretrained import OA_DM_38M, D3PM_BLOSUM_38M
from evodiff.generate import generate_oaardm, generate_d3pm

def generate_EvoDiff(min_len, max_len, batch_size=1, device="cpu"):
    seq_len = random.randint(min_len, max_len)
    checkpoint = OA_DM_38M()
    model, _, tokenizer, _ = checkpoint
    _, generated_sequence = generate_oaardm(model, tokenizer, seq_len, batch_size=batch_size, device=device)
    return generated_sequence[0]
