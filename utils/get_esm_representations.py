import os

import esm
import pandas as pd
import torch

from constants import MAIN_DIR, MAX_TRAINING_SIZE, NUM_SAMPLES_IN_DATAFRAME


class ESMRepresentationGenerator:
    def __init__(self, model_name="esm2_t33_650M_UR50D"):
        self.pretrained_llm, self.llm_alphabet = esm.pretrained.__dict__[model_name]()
        self.pretrained_llm.eval().cuda()
        self.batch_converter = self.llm_alphabet.get_batch_converter()
        self.last_layer_idx = self.pretrained_llm.num_layers
        self.device = next(self.pretrained_llm.parameters()).device

    def get_representations(self, row):
        sequence = row['sequence']
        batch_labels, batch_strs, batch_tokens = self.batch_converter([(row['chain_id'], sequence)])
        result = self.pretrained_llm(batch_tokens.to(self.device), repr_layers=[self.last_layer_idx])
        representations = result["representations"][self.last_layer_idx]
        logits = self.pretrained_llm.lm_head(representations)
        indices = torch.argmax(logits, dim=-1)
        predicted_sequence = ''.join([self.llm_alphabet.get_tok(i) for
                                       i in indices[0].tolist()])
        return representations[0], predicted_sequence[5:-5]


if __name__ == '__main__':
    esm_generator = ESMRepresentationGenerator()

    # Load the CATH dataset
    cath_json_file = os.path.join(MAIN_DIR, "cath_data", "cath_data.json")
    cath_df = pd.read_json(cath_json_file)
    valid_cath_df = cath_df[cath_df['sequence'].str.len() <= MAX_TRAINING_SIZE].copy()
    print(f"Filtered {len(cath_df) - len(valid_cath_df)} rows with sequences longer than {MAX_TRAINING_SIZE}.")

    # Split into train and test sets
    train_cath_df = valid_cath_df[valid_cath_df["dataset"] == "train"]
    test_cath_df = valid_cath_df[valid_cath_df["dataset"] == "test"]

    # Prepare and process in batches for train and test
    for dataset_name, dataset_df in [('train', train_cath_df), ('test', test_cath_df)]:
        representations_list = []
        predicted_sequences_list = []

        chunk_data = []

        for i, (_, row) in enumerate(dataset_df.iterrows()):
            # Process the row
            representation, predicted_sequence = esm_generator.get_representations(row)
            representations_list.append(representation.cpu().detach().numpy())
            predicted_sequences_list.append(predicted_sequence)

            # Add processed row to chunk_data
            row_data = row.copy()
            row_data["representations"] = representations_list[-1]
            row_data["predicted_sequence"] = predicted_sequences_list[-1]
            chunk_data.append(row_data)

            # Save the chunk if it reaches the specified chunk size
            if len(chunk_data) >= NUM_SAMPLES_IN_DATAFRAME:
                chunk_df = pd.DataFrame(chunk_data)
                output_file = os.path.join(MAIN_DIR, "cath_data", dataset_name + "_set",
                                           f"cath_df_{i // NUM_SAMPLES_IN_DATAFRAME}.json")
                chunk_df.to_json(output_file, orient="records", indent=4)
                print(f"Saved chunk {i // NUM_SAMPLES_IN_DATAFRAME} for {dataset_name} set to {output_file}")
                chunk_data = []

        # Save any remaining data in chunk_data that didn't reach full chunk size
        if chunk_data:
            chunk_df = pd.DataFrame(chunk_data)
            output_file = os.path.join(MAIN_DIR, "cath_data", dataset_name + "_set",
                                       f"cath_df_{len(dataset_df) // NUM_SAMPLES_IN_DATAFRAME}.json")
            chunk_df.to_json(output_file, orient="records", indent=4)
            print(f"Saved final chunk for {dataset_name} set to {output_file}")

    print("Processing complete!")
