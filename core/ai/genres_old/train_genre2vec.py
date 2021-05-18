import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from core.ai.models.enc2spotify_model import Enc2SpotifyModel, train_epocs


# class EncodingToSpotifyGenresDataset(Dataset):
#     def __init__(self, inputs, outputs):
#         self.inputs = inputs
#         self.outputs = outputs
#         if len(self.inputs) != len(self.outputs):
#             raise Exception(f"Mismatched input and output data. "
#                             f"Found {len(self.inputs)} inputs and {len(self.outputs)} outputs")
#
#     def __len__(self):
#         return len(self.inputs)
#
#     def __getitem__(self, idx):
#         return self.inputs[idx], self.outputs[idx]


def main():
    start = time.time()
    data = pd.read_csv('../data/genre_similarity.csv')



    print(f"Loaded data from files in {time.time() - start}s")
    # inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs_full, outputs_full, test_size=0.10)
    # train_dataset = EncodingToSpotifyGenresDataset(inputs_train, outputs_train)
    # test_dataset = EncodingToSpotifyGenresDataset(inputs_test, outputs_test)

    # train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

    # model = Enc2SpotifyModel()
    #
    # final_loss = train_epocs(model, train_loader, test_loader, epochs=100, lr=0.001, wd=1e-6)
    # print(f"FINAL LOSS: {final_loss}")


if __name__ == '__main__':
    main()
