import random

import torch
import torch.nn as nn

from constants import MIN_SIZE, MAX_TRAINING_SIZE


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()

    def load_inputs_and_ground_truth(self, data):
        raise NotImplementedError("Each strategy must implement this method.")

    def compute_loss(self, outputs, ground_truth):
        raise NotImplementedError("Each strategy must implement this method.")

    def forward(self, x):
        raise NotImplementedError("Each strategy must implement this method.")

    def evaluate(self, data):
        raise NotImplementedError("Each strategy must implement this method.")

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def get_augmentation_indices(seq_len):
        length = random.randint(MIN_SIZE, min(seq_len, MAX_TRAINING_SIZE))
        start = random.randint(0, max(0, seq_len - length))
        return start, start+length

