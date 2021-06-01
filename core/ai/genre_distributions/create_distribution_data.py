from itertools import repeat
import multiprocessing as mp
import os
import pprint
import time
import pandas as pd
import numpy as np
import json

from ast import literal_eval
from collections import Counter

from core.ai.util import rescale_distribution
from core.ai.genre2vec.cluster import get_clusters_from_genre_dict_given_data, enc_labels


def normalize(vals):
    if max(vals) == min(vals):
        return [1 / len(vals) for x in vals]
    return [(x - min(vals)) / (max(vals) - min(vals)) for x in vals]


def process_user(out_dict, ns, user_id):
    genres = ns.genres
    user_data = ns.user_data
    genre2enc = ns.genre2enc
    genre2idx = ns.genre2idx
    user_genres = []
    rows_for_user = user_data.loc[user_data['user_id'] == user_id]

    # For each song the user listened to, add as many copies of the genres as times they listened to the song
    for i in range(len(rows_for_user)):
        row_genres = literal_eval(genres.loc[genres['spotify_id'] == rows_for_user.iloc[i].spotify_id].genres.values[0])
        row_listen_count = int(rows_for_user.iloc[i].listen_count)
        user_genres.extend(row_genres * row_listen_count)

    user_genres = [x for x in user_genres if x in genre2idx.keys()]  # only use scraped everynoise genres
    user_genres = rescale_distribution(dict(Counter(user_genres)))  # Convert to map of genre to amount listened
    if len(user_genres) >= 6:
        # Get dataframe of clustered encoded genres
        cluster_df, ranked_clusters = get_clusters_from_genre_dict_given_data(user_genres, genre2enc, n_clusters=min(8, len(user_genres) - 1))
        ranked_clusters.append(-1)
        cluster_sizes = normalize([len(cluster_df.loc[cluster_df['cluster'] == x]) for x in ranked_clusters])
        cluster_rankings = normalize([sum(cluster_df.loc[cluster_df['cluster'] == x].ranking) for x in ranked_clusters])
        distribution_data = np.array([])
        for idx, cluster_label in enumerate(ranked_clusters):
            rows_for_cluster = cluster_df.loc[cluster_df['cluster'] == cluster_label]
            enc_data_for_cluster = cluster_df[list(enc_labels)].values
            mean = enc_data_for_cluster.mean(axis=0)
            std = enc_data_for_cluster.std(axis=0)

            # Create cluster vector. The first value is the ranking, the second is the size, the next 128 are the
            # mean vector of the cluster, and the final 128 are the std dev vector of the cluster. There are 258
            # total values in each cluster vector, making the whole distribution vector have 1548 values.
            cluster_vec = np.insert(np.insert(np.append(mean, std), 0, cluster_sizes[idx]), 0, cluster_rankings[idx])
            distribution_data = np.append(distribution_data, cluster_vec)

        out_dict[user_id] = distribution_data
        print(f"Processed user {user_id}")
        if len(out_dict) % 500 == 0:
            print(f"Processed {len(out_dict)} users. Saving checkpoint...")
            with open('checkpoint.json', 'w') as f:
                ser = {key: val.tolist() for key, val in dict(out_dict).items()}
                f.write(json.dumps(ser))
            print(f"Checkpoint saved.")


def main():
    # Load data
    load_start = time.time()
    user_data = pd.read_csv('../data/msd/user_dataset_lite.csv')
    genres = pd.read_csv('../data/msd/spotify_genres.csv')
    unique_users = user_data.user_id.unique()

    with open('../data/genre2vec/genre2idx.json', 'r') as f:
        genre2idx = json.loads(f.read())
    with open('../data/genre2vec/idx2genre.json', 'r') as f:
        idx2genre = json.loads(f.read())
        idx2genre = {int(k): idx2genre[k] for k in idx2genre.keys()}
    idx2enc = np.loadtxt('../data/genre2vec/idx2enc.csv', delimiter=',')
    genre2enc = {genre_str: idx2enc[genre2idx[genre_str]] for genre_str in genre2idx.keys()}
    load_end = time.time()
    print(f"Loaded data in {load_end - load_start}s")

    start = time.time()
    with mp.Manager() as manager:
        d = manager.dict()
        ns = manager.Namespace()
        ns.user_data = user_data
        ns.genres = genres
        ns.genre2enc = genre2enc
        ns.genre2idx = genre2idx
        with manager.Pool(processes=2, maxtasksperchild=1000) as pool:
            pool.starmap(process_user, zip(repeat(d, len(unique_users)), repeat(ns, len(unique_users)), unique_users), chunksize=100)
        # `d` is a DictProxy object that can be converted to dict
    with open('checkpoint.json', 'w') as f:
        f.write(json.dumps(dict(d)))
    print(f"Finished in {time.time() - start}s")


if __name__ == '__main__':
    main()

