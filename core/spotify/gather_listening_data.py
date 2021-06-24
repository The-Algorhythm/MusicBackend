import json
from collections import Counter

from core.spotify.util import get_all_results
from core.ai.util import rescale_distribution


def get_listening_data(user_spotify):
    """
    For a given user, get their listening data. This is used to create an initial user profile. The data is drawn from
    the user's liked songs and playlists (both their own and the playlists they follow). Everynoise genres are extracted
    from the artists for each of these tracks, the counts for each song are totaled, and they are rescaled to be between
    0 and 1. The final output is a map of everynoise genres to their frequencies.
    """
    sp = user_spotify.sp
    user_songs = []  # list of all songs gathered from the user's data

    # get saved songs (i.e. liked songs)
    saved_songs = get_all_results(sp.current_user_saved_tracks)
    user_songs.extend([x['track'] for x in saved_songs])

    # get all playlists that the user has created or follows
    playlists = get_all_results(sp.current_user_playlists)
    for playlist in playlists:  # TODO thread this to make it run much faster
        # get all tracks in this playlist
        results = get_all_results(sp.playlist_tracks, playlist_id=playlist['id'])
        user_songs.extend([x['track'] for x in results])

    # get top tracks over all time ranges
    user_songs.extend(get_all_results(sp.current_user_top_tracks, time_range="short_term"))
    user_songs.extend(get_all_results(sp.current_user_top_tracks, time_range="medium_term"))
    user_songs.extend(get_all_results(sp.current_user_top_tracks, time_range="long_term"))

    user_songs = [x for x in user_songs if x is not None and x['id'] is not None]
    counter = Counter([x['id'] for x in user_songs])
    genres = gather_genres(dict(counter), user_songs, user_spotify)

    return genres


def gather_genres(counter, user_songs, user_spotify):
    """
    Get the genres for a set of songs.
    :param counter: Mapping of song ids to their number of occurrences
    :param user_songs: List of Spotify track objects
    :param user_spotify: SpotifyCommunicator object for the user
    :return: A map of everynoise genres to the rescaled frequencies
    """
    genres = []
    artist_ids = []
    artists_data = []
    track_artists_map = dict()
    for track in user_songs:
        artists = track['artists']
        artists = [x['id'] for x in artists if x is not None and x['id'] is not None]
        for artist_id in artists:
            track_artists_map[artist_id] = track['id']
        artist_ids.extend(artists)

    # split artist list into sublists of max size 50
    artist_ids = [artist_ids[x:x + 50] for x in range(0, len(artist_ids), 50)]
    for i in range(len(artist_ids)):   # TODO thread this to make it run much faster
        lst = artist_ids[i]
        artists_data.extend(user_spotify.get_artists(artists=lst)['artists'])
    artists_data = [x for x in artists_data if x is not None]
    for artist in artists_data:
        count = counter[track_artists_map[artist['id']]]
        genres.extend(artist['genres']*count)
    return rescale_distribution(dict(Counter(genres)))

