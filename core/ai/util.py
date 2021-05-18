from math import log
import pickle


def rescale_distribution(counts_map):
    """
    Given a mapping where the values are counts, rescales the counts to be between 0 and 1 on a logarithmic scale.
    """
    counts = counts_map.values()
    max_log = log(max(counts))
    min_log = log(min(counts))
    for key, value in counts_map.items():
        scaled_value = (log(value) - min_log) / (max_log - min_log)
        counts_map[key] = scaled_value
    return counts_map


def invert_genre_data():
    with open('../data/genre_data.pickle', 'rb') as f:
        genre_data = pickle.load(f)
    print('finished loading')
    inverted_map = dict()
    for sp_tuple, distrib_lst in genre_data.items():
        for distrib in distrib_lst:
            new_key = tuple(sorted(distrib.items()))
            inverted_map[new_key] = sp_tuple
    with open('data/inverted_genre_data.pickle', 'wb') as f:
        pickle.dump(inverted_map, f)
    print('done')


def get_enc_tuple(vec, enc_map):
    enc_map = list(enc_map.keys())
    unencoded_map = dict()
    for i in range(len(vec)):
        if vec[i] != -1:
            unencoded_map[enc_map[i]] = vec[i]
    return tuple(sorted(unencoded_map.items()))
