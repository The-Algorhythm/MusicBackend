import time
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import load

from core.ai.models.genre2vec_model import Genre2Vec, train_epocs


class Genre2VecDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        if len(self.inputs) != len(self.outputs):
            raise Exception(f"Mismatched input and output data. "
                            f"Found {len(self.inputs)} inputs and {len(self.outputs)} outputs")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


def main():
    start = time.time()
    data = pd.read_csv('../data/genre2vec/genre2vec_training_data_weighted.csv')

    inputs_full = data[['center_genre', 'context_genre']].to_numpy()
    outputs_full = data[['similarity']].to_numpy()

    num_genres = len(data.center_genre.unique())

    print(f"Loaded data from files in {time.time() - start}s")
    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs_full, outputs_full, test_size=0.10)
    train_dataset = Genre2VecDataset(inputs_train, outputs_train)
    test_dataset = Genre2VecDataset(inputs_test, outputs_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=512)

    # Uncomment if loading from saved checkpoint
    # filename = 'checkpoint.pth.tar'
    # checkpoint = load(filename)

    model = Genre2Vec(num_genres, enc_size=128)

    # final_loss = train_epocs(model, train_loader, test_loader, epochs=100, lr=0.001, wd=1e-6, checkpoint=checkpoint)
    final_loss = train_epocs(model, train_loader, test_loader, epochs=100, lr=0.001, wd=1e-6)
    print(f"Loss after 100 epochs: {final_loss}")
    final_loss = train_epocs(model, train_loader, test_loader, epochs=100, lr=0.001, wd=1e-6)
    print(f"FINAL LOSS: {final_loss}")


if __name__ == '__main__':
    main()
