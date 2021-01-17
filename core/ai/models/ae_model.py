import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import shutil
import numpy as np


class Autoencoder(nn.Module):
    """
    This encoder maps everynoise genres down to a smaller encoding vector and then back to the full everynoise genre
    vector. The input vector consists of 3076 values (representing the 3076 everynoise genres that were scraped from
    Spotify). A -1 indicates that the genre does not appear in the distribution. Otherwise the values will range from
    0 to 1 on a logarithmic scale representing how common that genre was in the distribution.
    """
    def __init__(self, input_size, enc_size=256, p=0.1):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 768), nn.LeakyReLU(),
            nn.Linear(768, 512), nn.LeakyReLU(),
            nn.Linear(512, enc_size), nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.Linear(enc_size, 512), nn.LeakyReLU(),
            nn.Linear(512, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 2048), nn.LeakyReLU(),
            nn.Linear(2048, input_size), nn.Tanh())
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        x = 2*self.decoder(x)

        # for negative values, scale them to be close to -1 (but not exactly to prevent vanishing gradient)
        y = x.clone()
        mask = y < 0
        y[mask] = -1.125 + torch.pow(2, (torch.pow(y[mask] + 1, 2) - 3))  # -1.125 + 2^(((y+1)^2)-3)
        return y


def to_numpy(tensor):
    # Used for debugging, takes the tensor off of the GPU and converts it to a numpy ndarray
    return tensor.detach().numpy()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def custom_loss(y_hat, batch):
    """
    Custom loss function designed to speed up training. It computes the MSE loss as normal, but adds to it the MSE loss
    from only looking at the values that are supposed to be positive. This pushes the model to focus less on the
    multitude of -1's and more on the genres that are actually in the distribution.
    """
    coords = np.where(batch != -1)
    y_hat_non_neg = y_hat[coords]
    true_non_neg = batch[coords]
    full_loss = F.mse_loss(y_hat, batch)
    non_neg_loss = F.mse_loss(y_hat_non_neg, true_non_neg)
    return full_loss + non_neg_loss


def train_epocs(model, train_loader, val, epochs=10, lr=0.01, wd=0.0, checkpoint=None):
    """
    Run the training loop for the specified number of epochs. If checkpoint is passed in, it will load the model and
    optimizer weights from the saved values.
    """
    val = torch.Tensor(val)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    model.train()
    best_val_loss = 1000
    for epoch in range(epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, batch in loop:
            batch = batch.float()
            y_hat = model(batch)
            loss = custom_loss(y_hat, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())
        val_loss = test_loss(model, val)
        best_val_loss = min(best_val_loss, val_loss)
        print("test loss %.5f " % val_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=(val_loss == best_val_loss))
    return test_loss(model, val)


def test_loss(model, val):
    model.eval()
    y_hat = model(val)
    loss = custom_loss(y_hat, val)
    model.train()
    return loss.item()
