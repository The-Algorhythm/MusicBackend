import pandas as pd
import time
import json

start = time.time()
user_data = pd.read_csv('train_triplets.csv')
song_data = pd.read_csv('MSD_Spotify.csv')
unique_users = user_data['user_id'].unique()
num_to_user_id = {idx: id for idx, id in enumerate(unique_users)}
user_id_to_num = {id: idx for idx, id in enumerate(unique_users)}

loaded = time.time()
print(f"Loaded data in {loaded - start}s")

song_data = song_data[song_data.song_id.isin(user_data.song_id.unique())]
song_data = song_data[song_data.spotify_id.notnull()]

print(user_data.shape)
user_data = user_data[user_data.song_id.isin(song_data.song_id)]
print(user_data.shape)
user_data['user_id'] = user_data['user_id'].map(user_id_to_num)

joined = user_data.join(song_data.set_index('song_id'), on='song_id')
head = joined.head()

user_data = joined[['user_id', 'spotify_id', 'listen_count']]

user_data.to_csv('user_dataset_lite.csv', index=False)
with open('user_id_map.json', 'w') as f:
    f.write(json.dumps(num_to_user_id))
