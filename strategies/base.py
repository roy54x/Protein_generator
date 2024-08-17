import random
import torch.nn as nn

from constants import MIN_SIZE, MAX_SIZE


class BaseStrategy(nn.Module):
    def __init__(self):
        super(BaseStrategy, self).__init__()

    @staticmethod
    def get_augmentation_indices(seq_len):
        length = random.randint(MIN_SIZE, seq_len)
        start = random.randint(0, max(0, seq_len - length))
        return start, start+length

    def load_inputs_and_ground_truth(self, data):
        raise NotImplementedError("Each strategy must implement this method.")

    def compute_loss(self, outputs, ground_truth):
        raise NotImplementedError("Each strategy must implement this method.")

    def forward(self, x):
        raise NotImplementedError("Each strategy must implement this method.")

