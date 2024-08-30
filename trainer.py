import os
import time
from datetime import datetime

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, Dataset

from strategies.sequence_to_binding_sequence import SequenceDiffusionModel
from utils.constants import MIN_SIZE, MAIN_DIR, AMINO_ACIDS, MAX_SIZE, NUM_SAMPLES_IN_DATAFRAME
from strategies.sequence_to_distogram import SequenceToContactMap


class CustomDataset(Dataset):
    def __init__(self, dataframe, strategy):
        self.dataframe = dataframe
        self.strategy = strategy

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        inputs, ground_truth = self.strategy.load_inputs_and_ground_truth(row)
        return inputs, ground_truth


class Trainer:
    def __init__(self, directory, strategy, batch_size=32, test_size=0.2, device="cuda:0"):
        self.directory = directory
        self.strategy = strategy.to(device)
        self.batch_size = batch_size
        self.test_size = test_size
        self.device = device
        self.optimizer = torch.optim.Adam(strategy.parameters(), lr=0.001)
        self.best_test_loss = float('inf')

        # Collect all file paths from the directory
        self.file_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith('.json')]
        self.train_files, self.test_files = train_test_split(self.file_paths, test_size=test_size, random_state=42,
                                                             shuffle=False)
        print(f"number of samples in the train set are: {len(self.train_files)*NUM_SAMPLES_IN_DATAFRAME}")
        print(f"number of samples in the test set are: {len(self.test_files) * NUM_SAMPLES_IN_DATAFRAME}")
        print(f'Number of trainable parameters: '
              f'{sum(p.numel() for p in self.strategy.parameters() if p.requires_grad)}')

    def get_dataloader(self, file_path, mode):
        dataframe = pd.read_json(file_path)
        dataframe = dataframe[dataframe['sequence'].apply(lambda seq: len(seq) >= MIN_SIZE)]
        dataframe = dataframe[dataframe['sequence'].apply(lambda seq: len(seq) <= MAX_SIZE)]
        dataframe = dataframe[dataframe['sequence'].apply(lambda seq: all(char in AMINO_ACIDS for char in seq))]
        return DataLoader(CustomDataset(dataframe, self.strategy), batch_size=self.batch_size,
                          shuffle=(mode == "train"))

    def train(self, epochs=10):
        for epoch in range(epochs):

            # Process each training file one by one
            self.strategy.train()
            for train_file in self.train_files:
                train_loader = self.get_dataloader([train_file], mode="train")
                total_train_loss = 0
                total_train_samples = 0
                start_time = time.time()

                for inputs, ground_truth in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.strategy((x.to(self.device) for x in inputs))
                    loss = self.strategy.compute_loss(outputs, ground_truth.to(self.device))
                    loss.backward()
                    self.optimizer.step()

                    total_train_loss += loss.item()
                    total_train_samples += len(inputs)

                avg_train_loss = total_train_loss / total_train_samples
                print(f'Epoch {epoch + 1}, Train File {train_file}, Training Loss: {avg_train_loss:.4f}. '
                      f'Time taken: {time.time()-start_time:.4f} seconds.')

            # Evaluate on test data
            self.strategy.eval()
            total_test_loss = 0
            total_test_samples = 0
            with torch.no_grad():
                for test_file in self.test_files:
                    test_loader = self.get_dataloader([test_file], mode="test")
                    for inputs, ground_truth in test_loader:
                        outputs = self.strategy((x.to(self.device) for x in inputs))
                        loss = self.strategy.compute_loss(outputs, ground_truth.to(self.device))
                        total_test_loss += loss.item()
                        total_test_samples += len(inputs)

            average_test_loss = total_test_loss / total_test_samples
            print(f'Epoch {epoch + 1}, Test Loss: {average_test_loss:.4f}')

            # Save the model if the test loss is the best seen so far
            if average_test_loss < self.best_test_loss:
                self.best_test_loss = average_test_loss
                self.save_model()

    def save_model(self):
        # Create the directory name from strategy and date
        directory = os.path.join(MAIN_DIR, "models", self.strategy.__class__.__name__,
                                 datetime.now().strftime("%Y%m%d"))

        if not os.path.exists(directory):
            os.makedirs(directory)

        model_path = os.path.join(directory, 'best_model.pth')
        torch.save(self.strategy.state_dict(), model_path)
        print(f'Model saved at {model_path}')


if __name__ == '__main__':
    dataframe = pd.read_json(os.path.join(MAIN_DIR,"PDB\pdb_df_400.json"))
    strategy = SequenceToContactMap()
    #dataframe = pd.read_csv(os.path.join(MAIN_DIR,"UniProt\\uniprot_df.csv"))
    #strategy = SequenceDiffusionModel()
    trainer = Trainer(dataframe, strategy, batch_size=16, test_size=0.2)
    trainer.train(epochs=100)
