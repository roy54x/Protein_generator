import json
import os.path

import numpy as np
import pandas as pd
import torch
from numpy import average
from scipy.stats import pearsonr

import os
import torch
import pandas as pd
from scipy.stats import pearsonr

from constants import MAIN_DIR, PRETRAINED_MODEL_PATH, BATCH_SIZE
from strategies.coords_to_latent_space import CoordsToLatentSpace
from strategies.coords_to_sequence import CoordsToSequence
from trainer import get_dataloader


if __name__ == '__main__':
    directory = os.path.join(MAIN_DIR, "cath_data/test_set")
    strategy = CoordsToSequence()
    model_path = os.path.join(MAIN_DIR, PRETRAINED_MODEL_PATH)

    strategy.load_state_dict(torch.load(model_path))
    strategy.eval()

    results = []
    file_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith('.json')]
    for file_path in file_paths:

        test_loader = get_dataloader(file_path, strategy, BATCH_SIZE, mode="test")

        for batch_data in test_loader:
            metric_output = strategy.evaluate(batch_data)
            if metric_output is not None:
                results.append(metric_output)

    avg_result = np.mean(results)

    print(f"Overall average result: {avg_result:.4f}")

