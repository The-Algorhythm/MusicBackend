from django.http import JsonResponse
import time
import json

from core.spotify.auth import get_spotify_authenticator
from core.spotify.communicator import SpotifyCommunicator
from core.spotify.gather_listening_data import get_listening_data

spotify_auth = get_spotify_authenticator()  # generic authenticator used by all users


def create_profile(request):
    start = time.time()
    token_info = spotify_auth.update_token_info(json.loads(request.META['HTTP_TOKEN']))
    user_spotify = SpotifyCommunicator(token_info)  # communicator specific to this use

    listening_data = get_listening_data(user_spotify)
    return JsonResponse({"listening_data": listening_data, "time": time.time() - start})
