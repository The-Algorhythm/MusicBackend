import os
import json
import subprocess
import re


def init_credentials():
    """
    Creates the credentials.json file used by the canvas jar program if none exists.
    :return:
    """
    global_path = os.path.dirname(os.path.realpath(__file__))
    global_path = global_path[:-len('core/spotify/canvas/')]
    filename = f"{global_path}/cred/credentials.json"
    if not os.path.isfile(filename):
        with open(filename, 'w+') as f:
            credentials = {"username": os.getenv("SPOTIFY_USERNAME"),
                           "credentials": os.getenv("SPOTIFY_BLOB_CREDS"),
                           "type": "AUTHENTICATION_STORED_SPOTIFY_CREDENTIALS"}
            f.write(json.dumps(credentials))


def get_canvases(uris):
    """
    Given a list of Spotify song URIs, finds the URLs of the canvas videos for those songs from Spotify. Runs a Java
    program located at core/spotify/canvas/java/canvas.jar to do this. This jar file is a modified version of the
    librespot library found here: https://github.com/librespot-org/librespot. The jar file requires config.toml and
    cred/credentials.json to exist. It also required Java 8. The output of the program is analyzed and the URLs are
    extracted. Since not all songs have canvas videos, there may be less songs in the response than in the request.
    :param uris: A list of Spotify song URIs. It is expected that each URI begins with "spotify:track:".
    :return: A map of song URIs to canvas URLs
    """
    init_credentials()
    curr_path = os.path.dirname(os.path.realpath(__file__))
    uri_str = ",".join(uris)
    print(uri_str)
    cmd = subprocess.Popen(['java', '-jar', f"{curr_path}/java/canvas.jar", f"--uris={uri_str}"], stdout=subprocess.PIPE)
    cmd_out, cmd_err = cmd.communicate()
    res = cmd_out.decode("utf-8")
    relevant_indices = [m.start() for m in re.finditer('spotify:track:', res)]
    results = {}
    for idx in relevant_indices:
        end = res.find("\n", idx)
        line_splt = res[idx:end].split(" | ")
        results[line_splt[0].strip()] = line_splt[1].strip()
    return results
