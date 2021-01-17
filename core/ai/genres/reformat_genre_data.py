import time
import pickle


def get_scraped_genres(genre_data):
    """
    Given the full set of scraped genre_data, creates a key used for encoding genre maps into vectors. The list of
    genres is sorted alphabetically. Returns a tuple of genres, and enc_map, where genres is the list of all genres, and
    enc_map is the mapping of genre strings to their index.
    """
    genres = set()
    enc_map = dict()
    for distrib_lst in genre_data.values():
        for distrib in distrib_lst:
            for genre in distrib.keys():
                    genres.add(genre)
    genres = list(genres)
    for i in range(len(genres)):
        genre = genres[i]
        enc_map[genre] = i
    return genres, enc_map


def get_genre_vec(everynoise_genre_map, enc_map):
    """
    Gets the genre vector for a given everynoise distribution mapping using the given encoding map. The returned vector
    will have one value for each genre in the enc_map. Any genre that appears in the enc_map but not in the given
    distribution is a -1. Any genre that appears in the distribution but not in the enc_map is ignored.
    """
    vec = [-1] * len(enc_map)
    for genre, freq in everynoise_genre_map.items():
        if genre in enc_map:
            vec[enc_map[genre]] = freq
    return vec
