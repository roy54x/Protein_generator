import json
import os.path

import pandas as pd
import torch

from constants import MAIN_DIR, PRETRAINED_MODEL_PATH
from strategies.contact_map_to_sequence import ContactMapToSequence
from strategies.coords_to_latent_space import CoordsToLatentSpace
from strategies.coords_to_sequence import CoordsToSequence
from strategies.sequence_to_distogram import SequenceToDistogram

if __name__ == '__main__':
    strategy = CoordsToSequence()

    model_path = os.path.join(MAIN_DIR, PRETRAINED_MODEL_PATH)
    state_dict = torch.load(model_path)
    strategy.load_state_dict(state_dict)
    strategy.eval()

    cath_json_file = os.path.join(MAIN_DIR, "cath_data", "cath_data.json")
    cath_df = pd.read_json(cath_json_file)
    cath_df = cath_df[cath_df["dataset"] == "test"]
    print(f"Number of sequences in the test set is: {len(cath_df)}")

    total_recovery_rate = 0
    total_sequences = 0
    for i, data in cath_df.iterrows():
        recovery_rate = strategy.evaluate(data)

        if recovery_rate:
            total_recovery_rate += recovery_rate
            total_sequences += 1

    # Calculate the average recovery rate over all sequences
    average_recovery_rate = total_recovery_rate / total_sequences if total_sequences > 0 else 0
    print(f"Overall average recovery rate: {average_recovery_rate:.4f}")
