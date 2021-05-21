import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import shutil
import numpy as np
from scipy import spatial

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


class Genre2Vec(nn.Module):
    def __init__(self, input_size, enc_size=256):
        super(Genre2Vec, self).__init__()
        self.input_size = input_size
        self.embedding = nn.Embedding(input_size, enc_size)
        self.context = nn.Embedding(input_size, enc_size)
        self.to(device)

    def forward(self, input_genres, context_genres):
        a = self.embedding(input_genres)
        b = self.context(context_genres)
        dot = (torch.sum(a*b, dim=1))
        return torch.sigmoid(dot)

    def encode(self, idx):
        x = torch.tensor(idx).to(device)
        return self.embedding(x)

    def encode_to_context(self, idx):
        x = torch.tensor(idx).to(device)
        return self.context(x)

    def find_n_closest_idxs(self, input_embedding, n):
        n_closest = [-1]*n
        closest_sims = [0]*n
        for idx in range(self.input_size):
            sim = cosine_similarity(input_embedding, to_numpy(self.encode(idx)))
            if sim > min(closest_sims):
                inside_idx = closest_sims.index(min(closest_sims))
                closest_sims[inside_idx] = sim
                n_closest[inside_idx] = idx
        n_closest = [x for _, x in sorted(zip(closest_sims, n_closest), reverse=True)]
        closest_sims = sorted(closest_sims, reverse=True)
        return n_closest


def test():
    model = Genre2Vec(3, enc_size=5)

    input_genres = torch.LongTensor([0, 0])
    context_genres = torch.LongTensor([0, 1])
    model(input_genres, context_genres)
    x = 1


def cosine_similarity(vec1, vec2):
    return 1 - spatial.distance.cosine(vec1, vec2)


def to_numpy(tensor):
    # Used for debugging, takes the tensor off of the GPU and converts it to a numpy ndarray
    return tensor.cpu().detach().numpy()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def custom_loss(y_hat, batch_outputs):
    loss = nn.MSELoss()
    return loss(y_hat, batch_outputs)


def train_epocs(model, train_loader, test_loader, epochs=10, lr=0.01, wd=0.0, checkpoint=None):
    """
    Run the training loop for the specified number of epochs. If checkpoint is passed in, it will load the model and
    optimizer weights from the saved values.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    model.train()
    best_val_loss = 1000
    for epoch in range(epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (inputs, y_true) in loop:
            batch_input_genres, batch_context_genres = [x.reshape(x.shape[0]) for x in np.hsplit(inputs.to(device), 2)]
            batch_outputs = y_true.to(device).reshape(y_true.shape[0]).float()
            y_hat = model(batch_input_genres, batch_context_genres)
            loss = custom_loss(y_hat, batch_outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())
        val_loss = test_loss(model, test_loader)
        best_val_loss = min(best_val_loss, val_loss)
        print("test loss %.5f " % val_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=(val_loss == best_val_loss))
    return test_loss(model, test_loader)


def test_loss(model, test_loader):
    model.eval()
    avg_loss = torch.tensor(float(0)).to(device)
    num = 0
    for (inputs, y_true) in test_loader:
        inp, ctx = [x.reshape(x.shape[0]) for x in np.hsplit(inputs.to(device), 2)]
        y_true = y_true.to(device).reshape(y_true.shape[0]).float()
        y_hat = model(inp, ctx)
        loss = custom_loss(y_hat, y_true)
        avg_loss += loss
        num += 1
    avg_loss /= num
    model.train()
    return avg_loss.item()
