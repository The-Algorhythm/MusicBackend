import json
from core.spotify.auth import get_spotify_authenticator
from core.spotify.communicator import SpotifyCommunicator
from core.models import User

spotify_auth = get_spotify_authenticator()  # generic authenticator used by all users


def get_user(request):
    """
    Pulls the token information from the request and gets both the SpotifyCommunicator object for the user and the
    database model for the user.
    :param request: the HTTP request object
    :return: a tuple of (bool success, SpotifyCommunicator user_spotify, User user_model (or String error message))
    """
    token_info = spotify_auth.update_token_info(json.loads(request.META['HTTP_TOKEN']))
    user_spotify = SpotifyCommunicator(token_info)  # communicator specific to this user

    spotify_id = user_spotify.get_user_info()['id']
    models = User.objects.filter(spotify_id__exact=spotify_id)
    if len(models) > 0:
        return True, user_spotify, models.get()
    else:
        return False, None, 'No database entry found for this user'
