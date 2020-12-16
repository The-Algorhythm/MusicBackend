from spotipy import Spotify
from core.spotify.util import *


class SpotifyCommunicator:

    def __init__(self, token_info):
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
