import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import threading
import json
import ast
from collections import Counter

from core.ai.util import chunk_it

update_lock = threading.Lock()
client_credentials = SpotifyClientCredentials(client_id = 'a20a59fe314d4f23bb7bd658c5a1ca36', client_secret = '868f0f4956e747bdbdd16ae7bc871b4f')
spotify = spotipy.Spotify(client_credentials_manager = client_credentials)
genres = []
num_processed = 0
sleep_time = 5


def process_song_list(song_lst, start_idx, thread_num):
    id_idx_map = {spotify_id: global_idx for global_idx, spotify_id in zip(song_lst.index, song_lst.spotify_id)}
    song_lst = song_lst.spotify_id
    mini_lists = chunk_it_max(song_lst, 50)
    for i, song_ids in enumerate(mini_lists):
        song_ids = list(song_ids)
        results = query_tracks_with_retry(song_ids)
        if results is not None:
            for j, res in enumerate(results):
                mid_idx = sum([len(x) for x in mini_lists[:i]])
                global_idx = id_idx_map[mini_lists[i].iloc[j]]
                update_df(global_idx, res)
    print(f"THREAD #{thread_num} FINISHED")


def query_tracks_with_retry(song_ids, retry=0):
    try:
        res = spotify.tracks(song_ids)
        artists_flat = [y['id'] for x in res['tracks'] for y in x['artists']]
        artists_lists = chunk_it_max(artists_flat, 50)
        artist_genres_dict = dict()
        for artist_lst in artists_lists:
            artist_genres_dict.update(query_artists_with_retry(artist_lst))
        genres_for_each_song = [[item for sublist in x for item in sublist] for x in [[artist_genres_dict[y['id']] for y in x['artists']] for x in res['tracks']]]
        return genres_for_each_song
    except:
        if retry >= 5:
                print(f"Search failed for track_ids: {song_ids}")
                return None
        else:
            time.sleep(sleep_time)
            print(f"RETRY #{retry+1} FOR track_ids: {song_ids}")
            return query_tracks_with_retry(song_ids, retry + 1)


def query_artists_with_retry(artist_ids, retry=0):
    try:
        res = spotify.artists(artist_ids)
        return {x['id']: (x['genres'] if x['genres'] != [] else query_related_artist_genres_with_retry(x['id'])) for x in res['artists']}
    except:
        if retry >= 5:
                print(f"Search failed for artist_ids: {artist_ids}")
                return None
        else:
            time.sleep(sleep_time)
            print(f"RETRY #{retry+1} FOR artist_ids: {artist_ids}")
            return query_artists_with_retry(artist_ids, retry + 1)


def query_related_artist_genres_with_retry(artist_id, retry=0):
    try:
        res = spotify.artist_related_artists(artist_id)
        related_genres_flat = [item for sublist in [x['genres'] for x in res['artists']] for item in sublist]
        found_genres = [genre for genre, count in dict(Counter(related_genres_flat)).items() if count > 1]
        if not found_genres:
            found_genres = related_genres_flat
        return found_genres
    except:
        if retry >= 5:
                print(f"Search failed for artist_id: {artist_id}")
                return None
        else:
            time.sleep(sleep_time)
            print(f"RETRY #{retry+1} FOR artist_id: {artist_id}")
            return query_artists_with_retry(artist_id, retry + 1)


def update_df(idx, new_genres):
    update_lock.acquire()
    try:
        global genres, num_processed
        genres[idx] = new_genres
        num_processed += 1

        if num_processed % 5000 == 0:
            print(f"Saving checkpoint after processing {num_processed} songs")
            with open('checkpoint.json', 'w') as f:
                f.write(json.dumps(genres))
    finally:
        update_lock.release()


def chunk_it_max(seq, max_len):
    out = []
    last = 0.0

    while last < len(seq):
        if 2*max_len > len(seq) - last:
            out.append(seq[int(last):int(last + (len(seq) - last)/2)])
            last += int((len(seq) - last)/2)
            out.append(seq[int(last):])
            break
        else:
            out.append(seq[int(last):int(last + max_len)])
            last += max_len

    return out


def main():
    global genres

    # songs = pd.read_csv('songs_in_user_dataset.csv')
    # songs = songs['spotify_id']
    #
    # genres = [[]] * len(songs)
    songs = pd.read_csv('../data/msd/spotify_genres.csv')

    genres = [ast.literal_eval(x) for x in songs.genres]
    not_found = songs[[x == '[]' for x in songs['genres']]]  # only look at those ids where no genre was found

    num_threads = 10
    songs_for_threads = chunk_it(not_found, num_threads)
    threads = []

    for i in range(num_threads):
        song_list = songs_for_threads[i]
        start_idx = sum([len(x) for x in songs_for_threads[:i]])
        threads.append(threading.Thread(target=process_song_list, args=[song_list, start_idx, i+1]))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    df = pd.DataFrame({'spotify_id': songs.spotify_id, 'genres': genres})
    df.to_csv('spotify_genres_more.csv', index=False)


if __name__ == '__main__':
    main()
