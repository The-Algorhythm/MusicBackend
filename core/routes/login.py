from django.http import JsonResponse

from core.spotify.auth import get_spotify_authenticator
from core.spotify.communicator import SpotifyCommunicator

spotify_auth = get_spotify_authenticator()  # generic authenticator used by all users


def login(request):
    if request.GET.get('code') is not None:  # triggered on redirect with auth token
        spotify_auth.initialize_auth(request)  # initialize oauth client if not done already
        token_info = spotify_auth.get_initial_token_info(request.GET.get('code'))
        user_spotify = SpotifyCommunicator(token_info)  # get communicator specific to this user
        user = user_spotify.get_user_info()
        return JsonResponse({'current_user': user, 'token_info': token_info})
    else:  # triggered when user who is not logged in requests site
        return spotify_auth.initialize_auth(request)
