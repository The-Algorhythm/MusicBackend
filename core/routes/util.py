import json
from core.spotify.auth import get_spotify_authenticator
from core.spotify.communicator import SpotifyCommunicator
from core.models import User

spotify_auth = get_spotify_authenticator()  # generic authenticator used by all users


def get_user_model(request):
    token_info = spotify_auth.update_token_info(json.loads(request.META['HTTP_TOKEN']))
    user_spotify = SpotifyCommunicator(token_info)  # communicator specific to this user

    spotify_id = user_spotify.get_user_info()['id']
    models = User.objects.filter(spotify_id__exact=spotify_id)
    if len(models) > 0:
        return True, models.get()
    else:
        return False, 'No database entry found for this user'
