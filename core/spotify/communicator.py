import time

from spotipy import Spotify
from spotipy import SpotifyException

from core.spotify.auth import get_spotify_authenticator
from core.spotify.util import *

spotify_auth = get_spotify_authenticator()  # generic authenticator used by all users


class SpotifyCommunicator:

    def __init__(self, token_info):
        self.token_info = token_info
        self.sp = Spotify(token_info['access_token'])

    def get_user_info(self):
        return self.sp.current_user()

    def get_top_songs(self, time_range="medium_term", limit=20, offset=0):
        return self.sp.current_user_top_tracks(time_range=time_range, limit=limit, offset=offset)

    def get_recommendations_from_songs(self, songs, limit=20):
        uris = []
        for song in songs:
            uris.append(uri_string(song['uri']))
        return self.sp.recommendations(seed_tracks=uris, limit=limit)

    def get_recommendations(self, limit=50, **kwargs):
        return self.refresh_handler(get_recommendations_given_sp, sp=self.sp, limit=limit, **kwargs)

    def get_artist(self, **kwargs):
        return self.refresh_handler(get_artist_given_sp, sp=self.sp, **kwargs)

    def get_artists(self, **kwargs):
        return self.refresh_handler(get_artists_given_sp, sp=self.sp, **kwargs)

    def refresh_token(self):
        self.token_info = spotify_auth.update_token_info(self.token_info)
        self.sp = Spotify(self.token_info['access_token'])

    def refresh_handler(self, func, recurs=False, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SpotifyException as e:
            if e.args[0] == 429:
                time.sleep(3)
                print('Caught too many requests error, trying again')
                res = self.refresh_handler(func, recurs=True, *args, **kwargs)
                print('Success after waiting')
                return res
            if not recurs:
                if e.args[0] == 401:
                    self.refresh_token()
                    if 'sp' in kwargs:
                        kwargs['sp'] = self.sp
                        res = self.refresh_handler(func, recurs=True, *args, **kwargs)
                        print('Success on retry for token expired')
                        return res
                    else:
                        raise ValueError('sp should be supplied in kwargs')
                else:
                    raise Exception(f"Unknown exception occurred: {e}")
            else:
                print(f"Error on retry: {e}")
                return 'ERROR'


"""
The following methods are wrappers around the Spotify functions to allow passing these functions as parameters
independent of the Spotify class. This is useful when refreshing the auth token since it means creating a new Spotify
instance.
"""

def get_recommendations_given_sp(sp, *args, **kwargs):
    return sp.recommendations(*args, **kwargs)

def get_artist_given_sp(sp, *args, **kwargs):
    return sp.artist(*args, **kwargs)

def get_artists_given_sp(sp, *args, **kwargs):
    return sp.artists(*args, **kwargs)
