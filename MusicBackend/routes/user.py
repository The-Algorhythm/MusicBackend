import os
from django.http import JsonResponse


def user(request):
    return JsonResponse({'user_id': "1"})
