from math import log


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
