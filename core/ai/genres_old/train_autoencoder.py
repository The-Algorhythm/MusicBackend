from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import time
import pickle
from torch import load

from core.ai.models.ae_model import Autoencoder, train_epocs


def main():
    """
    Train the autoencoder. The genre_data array used as training data is a 2D array containing vectors of everynoise
    data that was scraped from Spotify.
    """

    start = time.time()
    with open('../data/everynoise_distributions.pickle', 'rb') as f:
        genre_data = pickle.load(f)
    genre_data = np.array(genre_data)
    print(f"Read file in {time.time() - start}s")

    train_data, val_data = train_test_split(genre_data, test_size=0.10)
    print(len(train_data))
    print(len(val_data))
    train_loader = DataLoader(dataset=train_data, batch_size=512, shuffle=True)

    # Uncomment if loading from saved checkpoint
    # filename = 'model_best.pth.tar'
    # checkpoint = load(filename)

    model = Autoencoder(input_size=genre_data.shape[1])

    final_loss = train_epocs(model, train_loader, val_data, epochs=100, lr=0.001, wd=1e-6)
    print(f"LOSS AFTER 100 EPOCHS: {final_loss}")
    final_loss = train_epocs(model, train_loader, val_data, epochs=10, lr=0.0005, wd=0.0)
    print(f"LOSS AT END: {final_loss}")


if __name__ == '__main__':
    main()
