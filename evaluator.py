import pandas as pd
import torch

from strategies.contact_map_to_sequence import ContactMapToSequence
from strategies.coords_to_sequence import CoordsToSequence
from strategies.sequence_to_distogram import SequenceToDistogram

if __name__ == '__main__':
    strategy = CoordsToSequence()

    model_path = r"D:\python project\data\Proteins\models\CoordsToSequence\20241126\best_model.pth"
    state_dict = torch.load(model_path)
    strategy.load_state_dict(state_dict)
    strategy.eval()

    data_path = r"C:\Users\RoyIlani\Desktop\personal\proteins\pdb_data\pdb_df_5.json"
    pdb_df = pd.read_json(data_path, lines=True)
    pdb_id = "7MQS"
    chain_id = "A"
    data = pdb_df.loc[(pdb_df['pdb_id'] == pdb_id) & (pdb_df['chain_id'] == chain_id)].iloc[0]

    strategy.evaluate(data)
