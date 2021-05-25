from bs4 import BeautifulSoup
import requests
from urllib.parse import quote
import math
import pandas as pd
import threading
import os
import json

final_df = pd.DataFrame(columns=['center_genre', 'context_genre', 'similarity'])
genre_count = 0
BASE_URL = "https://everynoise.com/everynoise1d.cgi?scope=all"

failed_genres = []
genre2idx = dict()
idx2genre = dict()

update_lock = threading.Lock()
failure_lock = threading.Lock()


def get_genres(url, retry_count=0):
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.content, features="html.parser")
        if url == BASE_URL:
            return [x.contents[2].contents[0].contents[0] for x in soup.findAll("tr")]
        else:
            data = [(genre, float(overlap.split(': ')[1]), float(acoustic_distance.split(': ')[1]))
                    for overlap, acoustic_distance, genre in
                    [x.contents[0].attrs['title'].split(',') + [x.contents[2].contents[0].contents[0]]
                     for x in soup.findAll("tr")]]
            max_overlap = -1
            min_overlap = 1000
            max_acoustic_distance = -1
            min_acoustic_distance = 1000
            for point in data:
                if 100 > point[1] > max_overlap:
                    max_overlap = point[1]
                if point[1] < min_overlap:
                    min_overlap = point[1]
                if point[2] > max_acoustic_distance:
                    max_acoustic_distance = point[2]
                if point[2] < min_acoustic_distance:
                    min_acoustic_distance = point[2]
            return (max_overlap, min_overlap, max_acoustic_distance, min_acoustic_distance), data
    except:
        if retry_count > 3:
            print(f"FAILURE FOR URL: {url}")
            return None
        else:
            print(f"Retry #{retry_count+1} for url: {url}")
            return get_genres(url, retry_count=retry_count+1)


def rank2prob(x):
    if x <= 20:
        return 1
    return 1.765*(1.16-0.1*math.log(20.05*x))


def data2prob(rank, overlap, acoustic_distance, word_sim, max_and_mins):
    rank = rank2prob(rank)
    if max_and_mins[0] == max_and_mins[1]:
        scaled_overlap = 0
    else:
        scaled_overlap = (overlap - max_and_mins[1]) / (max_and_mins[0] - max_and_mins[1])
    scaled_acoustic_factor = 1 - ((acoustic_distance - max_and_mins[3]) / (max_and_mins[2] - max_and_mins[3]))
    weighted_avg = ((3 * rank) + scaled_overlap + scaled_acoustic_factor + word_sim) / 6
    return weighted_avg


def word_similarity(w1, w2):
    w1_splt = w1.split(" ")
    w2_splt = w2.split(" ")
    intersect = list(set(w1_splt) & set(w2_splt))
    inter_size = len(intersect)
    if inter_size == 1:
        return 0.5
    if inter_size == 2:
        return 0.75
    if inter_size == 3:
        return 0.9
    if inter_size == 4:
        return 1.0
    return 0


def update_df(new_data, checkpoint_every_n=500):
    update_lock.acquire()
    try:
        global final_df, genre_count, failed_genres
        genre_count += 1
        final_df = pd.concat([final_df, new_data])
        if genre_count % checkpoint_every_n == 0:
            print(f"Saving checkpoint for {genre_count} genres...")

            final_df.to_csv(f"../data/genre2vec/chkpt_genre2vec_training_data_{genre_count}.csv", index=False)

            with open(f"../data/genre2vec/chkpt_failed_genres_{genre_count}.json", 'w') as f:
                f.write(json.dumps(failed_genres))

            if genre_count != checkpoint_every_n:
                print(f"Removing previous checkpoint (chkpt_genre2vec_training_data_{genre_count-checkpoint_every_n}.csv)...")
                prev_n = genre_count - checkpoint_every_n
                os.remove(f"../data/genre2vec/chkpt_genre2vec_training_data_{prev_n}.csv")
                os.remove(f"../data/genre2vec/chkpt_failed_genres_{prev_n}.json")
    finally:
        update_lock.release()


def update_failures(failed_genre):
    failure_lock.acquire()
    try:
        global failed_genres
        failed_genres.append(failed_genre)
    finally:
        failure_lock.release()


def update_for_genre(genre):
    genre_url = BASE_URL + "&root=" + quote(genre)
    max_and_mins, genres_for_genre = get_genres(genre_url)
    if genres_for_genre is not None:
        ranked_genres = [(genre2idx[genre], genre2idx[g],
                          data2prob(idx, overlap, acoustic_distance, word_similarity(genre, g), max_and_mins))
                         for (idx, (g, overlap, acoustic_distance)) in enumerate(genres_for_genre[1:])]
        ranked_genres_df = pd.DataFrame(ranked_genres, columns=['center_genre', 'context_genre', 'similarity'])
        update_df(ranked_genres_df)
        print(f"Finished processing genre: {genre}")
    else:
        update_failures(genre)


def run_genre_list(genre_list, i):
    for genre in genre_list:
        update_for_genre(genre)
    print(f"THREAD #{i} COMPLETED")


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def main():
    global genre2idx, idx2genre, final_df
    num_threads = 10

    genres = get_genres(BASE_URL)
    # with open('../data/genre2vec/genre2idx.json', 'r') as f:
    #     genre2idx = json.loads(f.read())
    # with open('../data/genre2vec/idx2genre.json', 'r') as f:
    #     idx2genre = json.loads(f.read())
    #     idx2genre = {int(k): idx2genre[k] for k in idx2genre.keys()}

    # final_df = pd.read_csv('genre2vec_training_data_3790.csv')

    # unique_values = [int(x) for x in final_df.center_genre.unique()]
    # genres = [idx2genre[x] for x in set(idx2genre.keys()) - set(unique_values)]

    if genres is not None:

        genre2idx = {g: idx for (idx, g) in enumerate(genres)}
        idx2genre = {idx: g for (idx, g) in enumerate(genres)}

        with open('../data/genre2vec/genre2idx.json', 'w') as f:
            f.write(json.dumps(genre2idx))
        with open('../data/genre2vec/idx2genre.json', 'w') as f:
            f.write(json.dumps(idx2genre))

        print("Saved map files as json")

        genres_for_threads = chunk_it(genres, num_threads)
        threads = []

        for genre_list in genres_for_threads:
            threads.append(threading.Thread(target=run_genre_list, args=[genre_list, len(threads)]))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        final_df.to_csv('../data/genre2vec/genre2vec_training_data_weighted.csv', index=False)

        with open(f"../data/genre2vec/failed_genres.json", 'w') as f:
            f.write(json.dumps(failed_genres))

    else:
        raise Exception("There was a problem retrieving the base genres")


if __name__ == '__main__':
    main()
