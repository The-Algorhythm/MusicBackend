import spotipy
import os
from spotipy.oauth2 import SpotifyOAuth
from spotipy import Spotify
from django.http import HttpResponseRedirect

class SpotifyCommunicator:

    def __init__(self):
        self.scopes = 'user-read-private user-read-email playlist-read-private playlist-read-collaborative'
        self.sp_oauth = None
        self.sp = None

    def _initOAuth(self, redirect_uri):
        self.sp_oauth = SpotifyOAuth(scope=self.scopes,
                                     client_id=os.getenv("SPOTIFY_CLIENT_ID"),
                                     client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
                                     redirect_uri=redirect_uri)
        return self.sp_oauth.get_authorize_url()

    def initalize_auth(self, request):
        redirect_uri = request.build_absolute_uri('/')+'login'
        return HttpResponseRedirect(self._initOAuth(redirect_uri))

    def test_spotify(self, request, token):
        token_info = self.sp_oauth.get_access_token(token)
        self.sp = Spotify(token_info['access_token'])
        return self.sp.current_user()
