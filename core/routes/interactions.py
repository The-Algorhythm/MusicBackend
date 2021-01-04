from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse
import time

from core.routes.util import get_user
from core.spotify.util import extract_song_data

from core.models import UserActivity


def interaction(request):
    if request.method == 'GET':
        return interaction_get(request)
    elif request.method == 'POST':
        return interaction_post(request)


def get_liked_songs(request):
    start = time.time()

    success, user_spotify, result = get_user(request)
    if not success:
        return HttpResponseBadRequest(result)
    user_model = result

    activity_models = UserActivity.objects.filter(user=user_model, activity_type=UserActivity.ActivityType.LIKE)
    song_ids = [x.spotify_id for x in activity_models]
    if len(song_ids) > 0:
        tracks = user_spotify.sp.tracks(song_ids)['tracks']
        song_data = extract_song_data(tracks, use_canvases=False, ensure_preview_url=False)
    else:
        song_data = []
    return JsonResponse({"tracks": song_data, "time": time.time()-start})


def interaction_get(request):
    start = time.time()

    interaction_type = None
    if "type" in request.GET.keys():
        interaction_type = request.GET["type"]

    success, _, result = get_user(request)
    if not success:
        return HttpResponseBadRequest(result)
    user_model = result

    if interaction_type is None:
        activity_models = UserActivity.objects.filter(user=user_model)
    elif interaction_type.upper() == "LIKE":
        activity_models = UserActivity.objects.filter(user=user_model, activity_type=UserActivity.ActivityType.LIKE)
    elif interaction_type.upper() == "SHARE":
        activity_models = UserActivity.objects.filter(user=user_model, activity_type=UserActivity.ActivityType.SHARE)
    elif interaction_type.upper() == "OPEN":
        activity_models = UserActivity.objects.filter(user=user_model, activity_type=UserActivity.ActivityType.OPEN)
    elif interaction_type.upper() == "LISTEN_LENGTH":
        activity_models = UserActivity.objects.filter(user=user_model, activity_type=UserActivity.ActivityType.LISTEN_LENGTH)
    elif interaction_type.upper() == "DISLIKE":
        activity_models = UserActivity.objects.filter(user=user_model, activity_type=UserActivity.ActivityType.DISLIKE)

    res = [{"song": x.spotify_id, "type": x.activity_type} for x in activity_models]
    return JsonResponse({"interactions": res, "time": time.time()-start})


def interaction_post(request):
    start = time.time()

    interaction_type = None
    song_id = None
    artist_ids = None
    listen_length = None
    if "type" in request.GET.keys():
        interaction_type = request.GET["type"]
    if "song_id" in request.GET.keys():
        song_id = request.GET["song_id"]
    if "artist_ids" in request.GET.keys():
        artist_ids = request.GET["artist_ids"]
    if "listen_length" in request.GET.keys():
        listen_length = request.GET["listen_length"]

    success, _, result = get_user(request)
    if not success:
        return HttpResponseBadRequest(result)
    else:
        user_model = result

        if interaction_type is None:
            return HttpResponseBadRequest('Must specify interaction type in query params')
        if song_id is None and artist_ids is None:
            return HttpResponseBadRequest('Must specify id in query params')

        action = None
        data = {}
        if interaction_type.upper() == "LIKE":
            action = UserActivity.ActivityType.LIKE
        elif interaction_type.upper() == "SHARE":
            action = UserActivity.ActivityType.SHARE
        elif interaction_type.upper() == "OPEN":
            action = UserActivity.ActivityType.OPEN
        elif interaction_type.upper() == "LISTEN_LENGTH":
            if listen_length is None:
                return HttpResponseBadRequest('Must give value for listen length in query params')
            action = UserActivity.ActivityType.LISTEN_LENGTH
            data = {"ms": listen_length}
        elif interaction_type.upper() == "UNLIKE":
            liked_post = UserActivity.objects.filter(user=user_model, spotify_id=song_id,
                                                     activity_type=UserActivity.ActivityType.LIKE)
            if len(liked_post) == 0:
                return HttpResponseBadRequest('There is not any record of this user liking that post')
            liked_post = liked_post.get()
            UserActivity.objects.filter(id=liked_post.id).delete()
            return JsonResponse({"time": time.time() - start})
        elif interaction_type.upper() == "DISLIKE":
            action = UserActivity.ActivityType.DISLIKE
            if song_id is not None:
                data = {"type": "song"}
                UserActivity.objects.create(user=user_model, spotify_id=song_id, data=data, activity_type=action)
            else:
                data = {"type": "artist"}
                artist_id_lst = artist_ids.split(',')
                for artist_id in artist_id_lst:
                    UserActivity.objects.create(user=user_model, spotify_id=artist_id, data=data, activity_type=action)
            return JsonResponse({"time": time.time() - start})

        identical_actions = UserActivity.objects.filter(user=user_model, spotify_id=song_id, data=data,
                                                        activity_type=action)
        if len(identical_actions) == 0:
            UserActivity.objects.create(user=user_model, spotify_id=song_id, data=data, activity_type=action)
        else:
            return HttpResponseBadRequest('There is already an existing entry with that same data')

        return JsonResponse({"time": time.time()-start})
