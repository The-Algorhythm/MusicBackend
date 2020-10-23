import os
from django.http import JsonResponse
from django.http import HttpResponseRedirect
import json

from routes.auth import *


def get_index(request):
    if request.GET.get('code') is not None:  # triggered on redirect with auth token
        request.session['token'] = request.GET.get('code')
        return HttpResponseRedirect("/")
    if request.session.get('token') is None:  # triggered when user who is not logged in requests site
        return initalize_auth(request)

    user = test_spotify(request, request.session.get('token'))
    return JsonResponse({'current_user': user})


def login(request):
    return HttpResponseRedirect(f"/?code={request.GET.get('code')}")
