import os
from django.http import JsonResponse
from django.http import HttpResponseRedirect
import json
import torch
import time

from core.spotify.auth import get_spotify_authenticator
from core.spotify.communicator import SpotifyCommunicator
from core.spotify.util import extract_song_data
from core.spotify.gather_listening_data import get_listening_data

spotify_auth = get_spotify_authenticator()  # generic authenticator used by all users


def test_torch(request):
    token_info = spotify_auth.update_token_info(json.loads(request.META['HTTP_TOKEN']))
    user_spotify = SpotifyCommunicator(token_info)
    listening_data = get_listening_data(user_spotify)
    return JsonResponse({"cuda_available": torch.cuda.is_available(), "rand_tensor": torch.rand(5, 3).numpy().tolist()})


def get_recommendations(request):
    start = time.time()
    token_info = spotify_auth.update_token_info(json.loads(request.META['HTTP_TOKEN']))

    use_canvases = True
    num_songs = 100
    if "use_canvases" in request.GET.keys():
        use_canvases = request.GET["use_canvases"].lower() == 'true'
    if "num_songs" in request.GET.keys():
        num_songs = request.GET["num_songs"]

    user_spotify = SpotifyCommunicator(token_info)  # communicator specific to this user
    top_songs = user_spotify.get_top_songs(limit=5)['items']
    recommendations = user_spotify.get_recommendations_from_songs(top_songs, limit=num_songs)['tracks']
    recommendations = extract_song_data(recommendations, use_canvases=use_canvases)
    return JsonResponse({"recommendations": recommendations, "time": time.time()-start})
