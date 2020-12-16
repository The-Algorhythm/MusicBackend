import os
from django.http import JsonResponse
from django.http import HttpResponseRedirect
import json

from ..routes.auth import *

spotify = SpotifyCommunicator()


def login(request):
    if request.GET.get('code') is not None:  # triggered on redirect with auth token
        token_info = spotify.get_initial_token_info(request.GET.get('code'))
        spotify.initalize_auth(request)  # initalize oauth client if not done already
        user = spotify.test_spotify(token_info)
        return JsonResponse({'current_user': user, 'token_info': token_info})
    else:  # triggered when user who is not logged in requests site
        return spotify.initalize_auth(request)
