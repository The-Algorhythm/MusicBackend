import os
from django.http import JsonResponse
from django.http import HttpResponseRedirect
import json

from ..routes.auth import *

spotify = SpotifyCommunicator()


def get_index(request):
    if request.GET.get('code') is not None:  # triggered on redirect with auth token
        response = HttpResponseRedirect("/")
        token_info = spotify.get_initial_token_info(request.GET.get('code'))
        response.set_cookie('token_info', json.dumps(token_info))
        return response
    if request.COOKIES.get('token_info') is None:  # triggered when user who is not logged in requests site
        return spotify.initalize_auth(request)

    spotify.initalize_auth(request)  # initalize oauth client if not done already
    token_info = spotify.update_token_info(json.loads(request.COOKIES.get('token_info')))
    user = spotify.test_spotify(token_info)
    return JsonResponse({'current_user': user})


def login(request):
    return HttpResponseRedirect(f"/?code={request.GET.get('code')}")
