import random
import torch.nn as nn

from constants import MIN_SIZE, MAX_SIZE


class BaseStrategy(nn.Module):
    def __init__(self):
        super(BaseStrategy, self).__init__()

    def load_inputs(self, data):
        raise NotImplementedError("Each strategy must implement this method.")

    def get_ground_truth(self, data):
        raise NotImplementedError("Each strategy must implement this method.")

    def cut_chain(self, sequence):
        raise NotImplementedError("Each strategy must implement this method.")

    def compute_loss(self, outputs, ground_truth):
        raise NotImplementedError("Each strategy must implement this method.")

    def forward(self, x):
        raise NotImplementedError("Each strategy must implement this method.")

