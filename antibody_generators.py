from evodiff.pretrained import OA_DM_38M
from evodiff.generate import generate_oaardm

def generate_single_sequence(seq_len=120, batch_size=1, device="cpu"):
    checkpoint = OA_DM_38M()
    model, _, tokenizer, _ = checkpoint
    _, generated_sequence = generate_oaardm(model, tokenizer, seq_len, batch_size=batch_size, device=device)
    return generated_sequence[0]