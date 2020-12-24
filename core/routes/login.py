from django.http import JsonResponse

from core.spotify.auth import get_spotify_authenticator
from core.spotify.communicator import SpotifyCommunicator

from core.models import User

spotify_auth = get_spotify_authenticator()  # generic authenticator used by all users


def login(request):
    if request.GET.get('code') is not None:  # triggered on redirect with auth token
        spotify_auth.initialize_auth(request)  # initialize oauth client if not done already
        token_info = spotify_auth.get_initial_token_info(request.GET.get('code'))
        user_spotify = SpotifyCommunicator(token_info)  # get communicator specific to this user
        user = user_spotify.get_user_info()

        spotify_id = user_spotify.get_user_info()['id']
        models = User.objects.filter(spotify_id__exact=spotify_id)
        if len(models) == 0:
            # There is not yet a database entry for this user so we need to create one
            User.objects.create(spotify_id=spotify_id)
        return JsonResponse({'current_user': user, 'token_info': token_info})
    else:  # triggered when user who is not logged in requests site
        return spotify_auth.initialize_auth(request)
