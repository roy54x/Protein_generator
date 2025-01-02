import os
import esm
import pandas as pd
import torch

from constants import MAIN_DIR, MAX_TRAINING_SIZE, BATCH_SIZE


class ESMRepresentationGenerator:
    def __init__(self, model_name="esm2_t33_650M_UR50D"):
        self.pretrained_llm, self.llm_alphabet = esm.pretrained.__dict__[model_name]()
        self.pretrained_llm.eval().cuda()
        self.batch_converter = self.llm_alphabet.get_batch_converter()
        self.last_layer_idx = self.pretrained_llm.num_layers
        self.device = next(self.pretrained_llm.parameters()).device

    def pad_sequence(self, sequence):
        padding_token = self.llm_alphabet.all_toks[self.llm_alphabet.padding_idx]
        return sequence + padding_token * (MAX_TRAINING_SIZE - len(sequence))

    def get_representations(self, batch_data):
        batch_converter_input = []
        for i, row in batch_data.iterrows():
            sequence = row['sequence']
            padded_sequence = self.pad_sequence(sequence)
            batch_converter_input.append((row['chain_id'], padded_sequence))

        batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_converter_input)
        result = self.pretrained_llm(batch_tokens.to(self.device), repr_layers=[self.last_layer_idx])
        representations = result["representations"][self.last_layer_idx]
        logits = self.pretrained_llm.lm_head(representations)
        indices = torch.argmax(logits, dim=-1)
        predicted_sequences = ''.join([self.llm_alphabet.get_tok(i) for
                                       i in indices.squeeze().tolist()])
        return representations, predicted_sequences


if __name__ == '__main__':
    esm_generator = ESMRepresentationGenerator()

    # Load the CATH dataset
    cath_json_file = os.path.join(MAIN_DIR, "cath_data", "cath_data.json")
    cath_df = pd.read_json(cath_json_file)
    valid_cath_df = cath_df[cath_df['sequence'].str.len() <= MAX_TRAINING_SIZE].copy()
    print(f"Filtered {len(cath_df) - len(valid_cath_df)} rows with sequences longer than {MAX_TRAINING_SIZE}.")

    # Prepare and process in batches
    representations_list = []
    predicted_sequences_list = []

    for i in range(0, len(valid_cath_df), BATCH_SIZE):
        batch_rows = valid_cath_df.iloc[i:i + BATCH_SIZE]
        representations, predicted_sequences = esm_generator.get_representations(batch_rows)
        representations_list.extend(representations.cpu().tolist())
        predicted_sequences_list.extend(predicted_sequences)

    # Add results to the dataframe
    valid_cath_df["representations"] = representations_list
    valid_cath_df["predicted_sequence"] = predicted_sequences_list

    # Save the updated dataframe to a JSON file
    output_file = os.path.join(MAIN_DIR, "cath_data", "cath_with_representations.json")
    valid_cath_df.to_json(output_file, orient="records", indent=4)

    print(f"Updated data saved to {output_file}")
