import os
from django.http import JsonResponse
from django.http import HttpResponseRedirect
import json
import torch
import time

from core.spotify.auth import get_spotify_authenticator
from core.spotify.communicator import SpotifyCommunicator
from core.spotify.util import extract_song_data

spotify_auth = get_spotify_authenticator()  # generic authenticator used by all users


def test_torch(request):
    return JsonResponse({"cuda_available": torch.cuda.is_available(), "rand_tensor": torch.rand(5, 3).numpy().tolist()})


def get_recommendations(request):
    start = time.time()
    body = json.loads(request.body)
    token_info = spotify_auth.update_token_info(body["token_info"])
    pref = body["preferences"]

    use_canvases = True
    num_songs = 100
    if "use_canvases" in pref.keys():
        use_canvases = pref["use_canvases"]
    if "num_songs" in pref.keys():
        num_songs = pref["num_songs"]

    user_spotify = SpotifyCommunicator(token_info)  # communicator specific to this user
    top_songs = user_spotify.get_top_songs(limit=5)['items']
    recommendations = user_spotify.get_recommendations_from_songs(top_songs, limit=num_songs)['tracks']
    recommendations = extract_song_data(recommendations, use_canvases=use_canvases)
    return JsonResponse({"recommendations": recommendations, "time": time.time()-start})
