import time
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from ray import tune
import ray
import copy
import os
import numpy as np

from core.ai.models.genre_vae import GenreVAE, train_epocs


class GenreDistributionsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


ray.init(ignore_reinit_error=True, num_cpus=10, object_store_memory=14000*1024*1024)

start = time.time()
# data_path = os.path.join(os.path.dirname(__file__), 'user_distrib_dump_tiny_10.gz')
data_path = os.path.join(os.path.dirname(__file__), 'user_distrib_dump_large_1m.gz')
data = np.loadtxt(data_path)
input_size = data.shape[1]
print(f"Loaded data from files in {time.time() - start}s")
train_data, test_data = train_test_split(data, test_size=0.10)
train_dataset = GenreDistributionsDataset(train_data)
test_dataset = GenreDistributionsDataset(test_data)

train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=256)

# Create model
enc_size = 128
model = GenreVAE(input_size=input_size, enc_size=enc_size)

train_loader_id = ray.put(train_loader)
test_loader_id = ray.put(test_loader)


def train_with_tune(config):
    train_loader = ray.get(train_loader_id)
    test_loader = ray.get(test_loader_id)
    train_epocs(config["model"], train_loader, test_loader, epochs=5, lr=config["lr"], wd=0.0, do_tune=True)


def find_lr(model, prev_lr):
    start = time.time()
    model_cpy = copy.deepcopy(model)
    potential_lrs = [prev_lr * 5, prev_lr * 2, prev_lr, prev_lr / 2, prev_lr / 5, prev_lr * 0.1]
    analysis = tune.run(train_with_tune, config={"model": model_cpy, "lr": tune.grid_search(potential_lrs)},
                        resources_per_trial={'gpu': 1},
                        num_samples=3,
                        max_failures=-1,
                        raise_on_failed_trial=False)
    print(f"Found lr in {time.time() - start}s")
    print("Best config: ", analysis.get_best_config(metric="mean_loss", mode='min'))
    return analysis.get_best_config(metric="mean_loss", mode='min')['lr']


def main():
    global model

    # final_loss = train_epocs(model, train_loader, test_loader, epochs=100, lr=0.0001, wd=0.0)

    num_trials = 25
    num_epochs_per_trial = 15
    epoch_start = 0
    lr = 0.0001
    while True:
        lr = find_lr(model, lr)
        best_model = None
        best_loss = 1000
        for i in range(num_trials):
            print("-------------------------")
            print(
                f"Running trial {i + 1} of {num_trials} (epochs {epoch_start + 1}-{epoch_start + num_epochs_per_trial})...")
            model_cpy = copy.deepcopy(model)
            final_loss = train_epocs(model_cpy, train_loader, test_loader, epochs=num_epochs_per_trial, lr=lr,
                                     wd=0.0, epoch_start=epoch_start)
            print(f"Finished trial {i + 1} with loss {final_loss}")
            if final_loss < best_loss:
                print(f"New best loss found, saving...")
                best_model = model_cpy
                best_model.save(
                    os.path.join(os.path.dirname(__file__),
                                 f"checkpoints/best_model_enc{enc_size}_ep{epoch_start + num_epochs_per_trial}_{round(final_loss, 4)}.pth.tar"))
                best_loss = final_loss
        model = best_model
        epoch_start += num_epochs_per_trial


if __name__ == '__main__':
    main()
