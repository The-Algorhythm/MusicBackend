"""
Defines a model object and associated methods for training a Variational Autoencoder to encode users' genre preference
distributions. The code for the VAE is modified from:
https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/

"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import shutil
from ray import tune

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


class GenreVAE(nn.Module):
    def __init__(self, input_size=1554, enc_size=128):
        super(GenreVAE, self).__init__()

        self.input_size = input_size
        self.enc_size = enc_size

        # ENCODER
        self.enc_hidden1 = nn.Linear(self.input_size, 512)
        self.enc_hidden2 = nn.Linear(512, 256)
        # self.enc_relu = nn.ReLU()
        self.enc_tanh = nn.Tanh()
        self.enc_mu = nn.Linear(256, self.enc_size)
        self.enc_logvar = nn.Linear(256, self.enc_size)

        # DECODER
        self.dec_hidden1 = nn.Linear(enc_size, 256)
        self.dec_hidden2 = nn.Linear(256, 512)
        self.dec_hidden3 = nn.Linear(512, self.input_size)
        # self.dec_tanh = nn.Tanh()

        self.training = True
        self.to(device)

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        """
        On the forward pass, we encode, then decode.
        :param x: The input tensor of size input_size
        :return: Tuple of decoded tensor of size input_size, mu of size enc_size, logvar of size enc_size
        """
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, x: Variable) -> (Variable, Variable):
        """
        Encode input vector. (input_size -> 512 -> 256 -> tanh -> (enc_size, enc_size))
        :param x: Input tensor
        :return: a tuple of encoded vectors representing the mu and logvar of the encoding distribution for this input
        """

        x = self.enc_hidden1(x)
        x = self.enc_hidden2(x)
        x = self.enc_tanh(x)
        return self.enc_mu(x), self.enc_logvar(x)

    def decode(self, z: Variable) -> Variable:
        """
        Decodes an encoded vector. (enc_size -> 256 -> 512 -> input_size)
        :param z: Encoded tensor of size enc_size
        :return: Decoded tensor of size input_size
        """
        z = self.dec_hidden1(z)
        z = self.dec_hidden2(z)
        z = self.dec_hidden3(z)
        # z = self.dec_tanh(z) TODO maybe rescale data so we can use tanh?
        return z

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        if self.training:
            std = logvar.mul(0.5).exp_()  # convert log variance to standard deviation
            eps = Variable(std.data.new(std.size()).normal_())  # sample from normal distribution with mu: 0, std: 1
            return eps.mul(std).add_(mu)  # scale newly sampled points to our given mu and std

        else:
            # When evaluating, we don't sample, we just use mu (because it is the most likely value)
            return mu

    def save(self, filename='checkpoint.pth.tar'):
        state = {'state_dict': self.state_dict()}
        torch.save(state, filename)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def custom_loss(recon_x, x, mu, logvar) -> Variable:
    """
    Computes loss as reconstruction loss + KL-Divergence loss. This both tries to make the reconstructions as accurate
    as possible while trying to ensure a nice Gaussian distribution.
    :param recon_x:
    :param x:
    :param mu:
    :param logvar:
    :return:
    """

    input_size = recon_x.shape[1]
    batch_size = recon_x.shape[0]

    # Reconstruction loss (tries to make reconstruction as accurate as possible)
    RECON = F.mse_loss(recon_x, x.view(-1, input_size))

    # KL-Divergence (tries to push distributions towards unit Gaussian)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size * input_size  # Normalise by the same number of elements as in reconstruction

    return RECON + KLD


def stolen_loss(y_hat, inputs, mu, logvar, kld_weight=0.005) -> dict:
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    :param y_hat:
    :param inputs:
    :param mu:
    :param logvar:
    :param kld_weight: Account for the minibatch samples from the dataset
    :return:
    """

    recons_loss = F.mse_loss(y_hat, inputs)

    # kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    kld_loss /= inputs.view(-1, inputs.shape[1]).data.shape[0] * inputs.shape[1]

    loss = recons_loss + kld_weight * kld_loss
    return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}


def train_epocs(model, train_loader, test_loader, epochs=10, lr=0.01, wd=0.0, checkpoint=None, do_tune=False,
                epoch_start=0):
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
        if do_tune:
            loop = tqdm(enumerate(train_loader), total=len(train_loader), disable=True)
        else:
            loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, inputs in loop:
            inputs = inputs.float().to(device)
            y_hat, mu, logvar = model(inputs)
            loss = stolen_loss(y_hat, inputs, mu, logvar)['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update progress bar
            loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            loop.set_postfix(loss=loss.item())
        # val_loss = test_loss(model, test_loader)
        #     best_val_loss = min(best_val_loss, val_loss)
        # print("test loss %.5f " % val_loss)
    #     save_checkpoint({
    #         'epoch': epoch + 1,
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #     }, is_best=(val_loss == best_val_loss))
    val_loss = test_loss(model, test_loader)
    if do_tune:
        tune.report(mean_loss=val_loss)
    return val_loss


def test_loss(model, test_loader):
    model.eval()
    avg_loss = 0
    num = 0
    for inputs in test_loader:
        inputs = inputs.float().to(device)
        y_hat, mu, logvar = model(inputs)
        loss = stolen_loss(y_hat, inputs, mu, logvar)
        print(f"Test loss: {loss}")
        loss = loss['loss']
        avg_loss += loss.item()
        num += 1
    avg_loss /= num
    model.train()
    return avg_loss
