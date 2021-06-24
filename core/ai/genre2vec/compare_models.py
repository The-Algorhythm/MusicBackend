import time
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import load
import os

from core.ai.models.genre2vec_model import Genre2Vec, test_loss
from core.ai.genre2vec.train_genre2vec import Genre2VecDataset


# Load data
start = time.time()
data_path = os.path.join(os.path.dirname(__file__), '../data/genre2vec/genre_training_data_all.csv')
data = pd.read_csv(data_path)

inputs_full = data[['center_genre', 'context_genre']].to_numpy()
outputs_full = data[['score']].to_numpy()

num_genres = len(data.center_genre.unique())

full_dataset = Genre2VecDataset(inputs_full, outputs_full)
full_data_loader = DataLoader(dataset=full_dataset, batch_size=512, shuffle=True)

print(f"Loaded data from files in {time.time() - start}s")


enc_size = 128
model1 = Genre2Vec(input_size=4444, enc_size=enc_size)
model2 = Genre2Vec(input_size=4444, enc_size=enc_size)
model3 = Genre2Vec(input_size=4444, enc_size=enc_size)
model4 = Genre2Vec(input_size=4444, enc_size=enc_size)
model5 = Genre2Vec(input_size=4444, enc_size=enc_size)

model1.load_state_dict(
    load('checkpoints/best_model_enc128_ep60_0.007.pth.tar')['state_dict'])
model2.load_state_dict(
    load('checkpoints/best_model_enc128_ep75_0.007.pth.tar')['state_dict'])
model3.load_state_dict(
    load('checkpoints/best_model_enc128_ep90_0.007.pth.tar')['state_dict'])
model4.load_state_dict(
    load('checkpoints/best_model_enc128_ep105_0.007.pth.tar')['state_dict'])
model5.load_state_dict(
    load('checkpoints/best_model_enc128_ep120_0.007.pth.tar')['state_dict'])

model1_loss = test_loss(model1, full_data_loader)
model2_loss = test_loss(model2, full_data_loader)
model3_loss = test_loss(model3, full_data_loader)
model4_loss = test_loss(model4, full_data_loader)
model5_loss = test_loss(model5, full_data_loader)

print(f"Model 1 loss: {model1_loss}\nModel 2 loss: {model2_loss}\nModel 3 loss: {model3_loss}\nModel 4 loss: {model4_loss}\nModel 5 loss: {model5_loss}")
