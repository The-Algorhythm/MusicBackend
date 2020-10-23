import spotipy
import os
from spotipy.oauth2 import SpotifyOAuth
from spotipy import Spotify
from django.http import HttpResponseRedirect

redirect_uri = 'http://localhost:8000/login'
scopes = 'user-read-private user-read-email playlist-read-private playlist-read-collaborative'
sp_oauth = SpotifyOAuth(scope=scopes,
                            client_id=os.getenv("SPOTIFY_CLIENT_ID"),
                            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
                            redirect_uri=redirect_uri)


def initalize_auth(request):
    return HttpResponseRedirect(sp_oauth.get_authorize_url())


def test_spotify(request, token):
    token_info = sp_oauth.get_access_token(token)
    sp = Spotify(token_info['access_token'])
    return sp.current_user()
