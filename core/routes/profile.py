from django.http import JsonResponse

from core.spotify.auth import get_spotify_authenticator
from core.spotify.communicator import SpotifyCommunicator
import json

spotify_auth = get_spotify_authenticator()  # generic authenticator used by all users


def get_profile(request):
    body = json.loads(request.body)
    token_info = spotify_auth.update_token_info(body["token_info"])
    user_spotify = SpotifyCommunicator(token_info)  # get communicator specific to this user
    user = user_spotify.get_user_info()
    return JsonResponse({'current_user': user, 'token_info': token_info})
