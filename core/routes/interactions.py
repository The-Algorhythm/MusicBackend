import time

from django.http import JsonResponse, HttpResponseBadRequest

from core.models import UserActivity
from core.routes.util import get_user
from core.spotify.util import extract_song_data


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
    return JsonResponse({"tracks": song_data, "time": time.time() - start})


def interaction_get(request):
    start = time.time()

    interaction_type = None
    if "interaction_type" in request.GET.keys():
        interaction_type = request.GET["interaction_type"]

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
        activity_models = UserActivity.objects.filter(user=user_model,
                                                      activity_type=UserActivity.ActivityType.LISTEN_LENGTH)
    elif interaction_type.upper() == "DISLIKE":
        activity_models = UserActivity.objects.filter(user=user_model, activity_type=UserActivity.ActivityType.DISLIKE)

    res = [{"song": x.spotify_id, "type": x.activity_type} for x in activity_models]
    return JsonResponse({"interactions": res, "time": time.time() - start})


def interaction_post(request):
    start = time.time()

    interaction_type = None
    spotify_id = None
    listen_length = None
    object_type = "TRACK"  # track by default
    if "interaction_type" in request.GET.keys():
        interaction_type = request.GET["interaction_type"]
    if "spotify_id" in request.GET.keys():
        spotify_id = request.GET["spotify_id"]
    if "listen_length" in request.GET.keys():
        listen_length = request.GET["listen_length"]
    if "object_type" in request.GET.keys():
        object_type = request.GET["object_type"]

    if object_type.upper() == "ARTIST":
        obj_type_enum = UserActivity.ObjectType.ARTIST
    elif object_type.upper() == "ALBUM":
        obj_type_enum = UserActivity.ObjectType.ALBUM
    elif object_type.upper() == "PLAYLIST":
        obj_type_enum = UserActivity.ObjectType.PLAYLIST
    else:
        obj_type_enum = UserActivity.ObjectType.TRACK

    success, _, result = get_user(request)
    if not success:
        return HttpResponseBadRequest(result)
    else:
        user_model = result

        if interaction_type is None:
            return HttpResponseBadRequest('Must specify interaction type in query params')
        if spotify_id is None:
            return HttpResponseBadRequest('Must specify spotify id in query params')

        action = None
        data = {}
        response = HttpResponseBadRequest('There is already an existing entry with that same data')

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
            liked_post = UserActivity.objects.filter(user=user_model, spotify_id=spotify_id, object_type=obj_type_enum,
                                                     activity_type=UserActivity.ActivityType.LIKE)
            if len(liked_post) == 0:
                return HttpResponseBadRequest('There is not any record of this user liking that post')
            liked_post = liked_post.get()
            UserActivity.objects.filter(id=liked_post.id).delete()
            return JsonResponse({"time": time.time() - start})

        elif interaction_type.upper() == "DISLIKE":
            action = UserActivity.ActivityType.DISLIKE
            if obj_type_enum == UserActivity.ObjectType.ARTIST:
                artist_id_lst = spotify_id.split(',')

                for artist_id in artist_id_lst:
                    response_code = check_duplicate(start, user_model, artist_id, data, obj_type_enum, action)
                    if response_code == 200:
                        # Only give 400 error if all the artists in the request are duplicates, i.e. return 200 for any
                        # non-duplicate.
                        response = JsonResponse({"time": time.time() - start})
                return response

        response_code = check_duplicate(start, user_model, spotify_id, data, obj_type_enum, action)
        if response_code == 200:
            response = JsonResponse({"time": time.time() - start})
        return response


def check_duplicate(start_time, user_model, spotify_id, data, object_type, action):
    identical_actions = UserActivity.objects.filter(user=user_model, spotify_id=spotify_id, data=data,
                                                    object_type=object_type, activity_type=action)
    if len(identical_actions) == 0:
        UserActivity.objects.create(user=user_model, spotify_id=spotify_id, object_type=object_type,
                                    data=data, activity_type=action)
        return 200
    else:
        return 400
