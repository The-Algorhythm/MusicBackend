import pickle
from threading import Thread
import itertools
from collections import Counter
from core.spotify.auth import get_spotify_authenticator
from core.spotify.communicator import SpotifyCommunicator
from core.ai.util import rescale_distribution

spotify_auth = get_spotify_authenticator()  # generic authenticator used by all users
with open('genre_data.pickle', 'rb') as f:
    genre_data = pickle.load(f)
# with open('skipped.pickle', 'rb') as f:
#     skipped = pickle.load(f)


def generate_combinations(user_spotify):
    """
    Used before running the main program to get all possible combinations of lengths 1, 2, and 3 of the Spotify seed
    genres.
    """
    combs = []
    genres = user_spotify.sp.recommendation_genre_seeds()['genres']
    for num_seeds in range(1, 4):
        for comb in itertools.combinations(genres, num_seeds):
            combs.append(comb)

    with open('combinations.pickle', 'wb') as f:
        pickle.dump(combs, f)


def extract_genres(res, user_spotify):
    """
    Given a recommendation response, extract the genres. All the artists are extracted from the song and grouped into
    lists of max length 50 (since this is the max limit for an artists request on Spotify). The list of genres is then
    grouped into counts and rescaled logarithmically to be between 0 and 1.
    """
    genres = []
    artist_ids = []
    artists_data = []
    for track in res:
        artists = track['artists']
        artist_ids.extend([x['id'] for x in artists])

    # split artist list into sublists of max size 50
    artist_ids = [artist_ids[x:x+50] for x in range(0, len(artist_ids), 50)]
    for lst in artist_ids:
        artists_data.extend(user_spotify.get_artists(artists=lst)['artists'])
    for artist in artists_data:
        genres.extend(artist['genres'])
    return rescale_distribution(dict(Counter(genres)))


def loop(start, thread_num, num_threads, combinations, user_spotify, num_samples=1):
    """
    The loop run by each thread. Loops through the combinations, and if this thread is allowed to evaluate the given
    combination, it will query for recommendations and then extract the genres from the given recommendations.
    """
    index = 0
    for comb in combinations:
        index += 1
        if index % num_threads == thread_num:
            data = []
            for _ in range(num_samples):
                res = user_spotify.get_recommendations(seed_genres=list(comb), limit=100)['tracks']
                genres = extract_genres(res, user_spotify)
                data.append(genres)
            genre_data[comb].extend(data)

            if index % 750 == 0:
                print(f"Thread #{thread_num} | Saving checkpoint, idx: {index}, map size: {len(genre_data)}")
                with open('genre_data.pickle', 'wb') as f:
                    pickle.dump(genre_data, f)


def gather_genre_data(user_spotify):
    """
    Method used for scraping Spotify for genre data. This is done by querying Spotify for recommendations based on
    combinations of the possible seed genres. The everynoise genres are extracted from the artists of the resulting
    tracks. A map, genre_data, is created that maps seed genre tuples to everynoise genre lists. This querying takes a
    long time with a large number of combinations, so it is threaded to try to make as many requests as possible.
    """
    with open('combinations.pickle', 'rb') as f:
        # Combinations is a list of tuples of every combination of length 1, 2, and 3 of the 126 Spotify seed genres
        combinations = pickle.load(f)
    threads = []
    num_threads = 10
    start = len(genre_data)
    for i in range(0, num_threads):
        thread = Thread(target=loop, args=(start, i, num_threads, combinations, user_spotify))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    with open('genre_data.pickle', 'wb') as f:
        pickle.dump(genre_data, f)


def main():
    # Need to pass in Spotify token info
    user_spotify = SpotifyCommunicator(token_info)
    gather_genre_data(user_spotify)


if __name__ == '__main__':
    main()
