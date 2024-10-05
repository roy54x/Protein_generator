import pandas as pd
import torch

from strategies.contact_map_to_sequence import ContactMapToSequence
from strategies.sequence_to_distogram import SequenceToDistogram

if __name__ == '__main__':
    strategy = ContactMapToSequence()

    model_path = r"C:\Users\RoyIlani\Desktop\proteins\models\ContactMapToSequence\20241005\best_model.pth"
    state_dict = torch.load(model_path)
    strategy.load_state_dict(state_dict)
    strategy.eval()

    data_path = r"C:\Users\RoyIlani\Desktop\proteins\pdb_data_130000\pdb_df_9.json"
    pdb_df = pd.read_json(data_path, lines=True)
    pdb_id = "5P2T"
    chain_id = "A"
    data = pdb_df.loc[(pdb_df['pdb_id'] == pdb_id) & (pdb_df['chain_id'] == chain_id)].iloc[0]

    strategy.evaluate(data)
