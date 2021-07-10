import pandas as pd
import numpy as np
import time
import threading
import json
from itertools import product
import os

from ast import literal_eval
from collections import Counter

from core.ai.util import rescale_distribution, chunk_it
from core.ai.genre2vec.cluster import get_clusters_from_genre_dict, enc_labels, genre2idx
from core.ai.db.util import exec_sql_file, insert_user, query_user


def load_data():
    user_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/msd/user_dataset_lite.csv'))
    genres = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/msd/spotify_genres.csv'))
    genres['genres'] = [literal_eval(x) for x in genres['genres']]
    genres = genres.drop_duplicates(subset=['spotify_id'])
    user_data = user_data.join(genres.set_index('spotify_id'), on='spotify_id')
    unique_users = user_data.user_id.unique()
    return user_data, unique_users


# Globals
all_data = dict()
global_counter = 0
last_checkpoint = time.time()
update_lock = threading.Lock()

# Load data
load_start = time.time()
user_data, unique_users = load_data()
load_end = time.time()
print(f"Loaded data in {load_end - load_start}s")


def normalize(vals):
    if max(vals) == min(vals):
        return [1 / len(vals) for x in vals]
    return [(0.9 * (x - min(vals)) / (max(vals) - min(vals)))+0.1 for x in vals]


def process_user(user_id):
    user_genres = []
    rows_for_user = user_data.loc[user_data['user_id'] == user_id]

    # For each song the user listened to, add as many copies of the genres as times they listened to the song
    [user_genres.extend(genres * cnt) for genres, cnt in zip(rows_for_user['genres'], rows_for_user['listen_count'])]

    user_genres = [x for x in user_genres if x in genre2idx.keys()]  # only use scraped everynoise genres
    user_genres = dict(Counter(user_genres))  # Convert to map of genre to amount listened
    if len(user_genres) >= 6:
        user_genres = rescale_distribution(user_genres)  # Rescale values to be between 0 and 1
        # Get dataframe of clustered encoded genres
        cluster_df, ranked_clusters, cluster_sums = get_clusters_from_genre_dict(user_genres,
                                                                                 n_clusters=min(8, len(user_genres)))
        ranked_clusters.append(-1)
        cluster_sizes = normalize([len(cluster_df.loc[cluster_df['cluster'] == x]) for x in ranked_clusters])
        cluster_rankings = normalize([sum(cluster_df.loc[cluster_df['cluster'] == x].ranking) for x in ranked_clusters])
        distribution_data = np.array([])
        for idx, cluster_label in enumerate(ranked_clusters):
            enc_data_for_cluster = cluster_df.loc[cluster_df['cluster'] == cluster_label][list(enc_labels)].values
            mean = enc_data_for_cluster.mean(axis=0)
            std = enc_data_for_cluster.std(axis=0)

            # Create cluster vector. The first value is the size, the second is the ranking, the third is based on the
            # scaled sum of all clusters, the next 128 are the mean vector of the cluster, and the final 128 are the
            # std dev vector of the cluster. There are 258 total values in each cluster vector, making the whole
            # distribution vector have 1554 values.
            scalars = np.array([cluster_sizes[idx], cluster_rankings[idx], cluster_sums[idx]])
            cluster_vec = np.append(scalars, np.append(mean, std))
            distribution_data = np.append(distribution_data, cluster_vec)
        update_all_data(user_id, distribution_data)


def process_user_lst(user_lst):
    print("Thread started")
    for user_id in user_lst:
        process_user(user_id)
    print("Thread finished")


def update_all_data(user_id, distribution_data):
    update_lock.acquire()
    try:
        global all_data, global_counter, last_checkpoint
        insert_user(int(user_id), distribution_data)
        global_counter += 1
        if global_counter % 50000 == 0:
            print(f"Processed {global_counter} of {len(unique_users)} total users in "
                  f"{(time.time() - last_checkpoint)/60} min. Saved checkpoint.")
            last_checkpoint = time.time()
    finally:
        update_lock.release()


def main():
    global all_data

    exec_sql_file('../db/init.sql')
    num_threads = 10
    # num_threads = 1
    users_for_threads = chunk_it(unique_users, num_threads)
    threads = []

    for i in range(num_threads):
        user_lst = users_for_threads[i]
        threads.append(threading.Thread(target=process_user_lst, args=(user_lst,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    print("Finished. Saving final csv...")
    with open("user_genre_data.json", 'w') as f:
        f.write(json.dumps(all_data))


if __name__ == '__main__':
    main()
