import os
import json
import subprocess
import re


def init_credentials():
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
    init_credentials()
    curr_path = os.path.dirname(os.path.realpath(__file__))
    uri_str = ",".join(uris)
    print(uri_str)
    cmd = subprocess.Popen(['java', '-jar', f"{curr_path}/java/canvas.jar", f"--uris={uri_str}"], stdout=subprocess.PIPE)
    cmd_out, cmd_err = cmd.communicate()
    res = cmd_out.decode("utf-8")
    relevant_indices = [m.start() for m in re.finditer('spotify:track:', res)]
    results = []
    for idx in relevant_indices:
        end = res.find("\n", idx)
        line_splt = res[idx:end].split(" | ")
        results.append({line_splt[0].strip(): line_splt[1].strip()})
    return results
