import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import unquote
from core.spotify.canvas.canvas import get_canvases


def get_preview_url(uri):
    """
    Finds the preview url for a given URI. The API can be inconsistent for returning this URL, so this method is a
    workaround that queries the Spotify embed url for the track to extract the preview URL that way.
    :param uri: the URI string for a track
    :return: the URL of the 30 second preview for the song if it exists
    """
    url = 'https://open.spotify.com/embed/track/' + uri
    r = requests.get(url)
    soup = BeautifulSoup(r.content, features="html.parser")
    song_json = json.loads(unquote(soup.find("script", {"id": "resource"}).contents[0]))
    return song_json['preview_url']


def uri_string(uri):
    """
    Extracts the URI string from a uri in the form of spotify:track:URI
    :param uri:
    :return:
    """
    return uri[len('spotify:track:'):]


def extract_song_data(songs, use_canvases=True, ensure_preview_url=True):
    """
    Given a list of Spotify song dictionaries, extracts the relevant info needed and populates any missing info
    :param songs: list of Spotify song dictionaries
    :param use_canvases: Bool for whether or not to retrieve canvas urls
    :param ensure_preview_url: Bool for whether or not to query the embed url for the preview if it does not exist
    :return: list of dicts containing relevant song data
    """
    song_data = []
    uris = [x['uri'] for x in songs]
    if use_canvases:
        canvases = get_canvases(uris)
    for song in songs:
        artist_str = ""
        for artist in song['artists']:
            artist_str += artist['name'] + ", "
        artist_str = artist_str[:-2]

        preview_url = song['preview_url']
        if ensure_preview_url and preview_url is None:
            preview_url = get_preview_url(song['id'])
        data = {
            "song_title": song['name'],
            "album_title": song['album']['name'],
            "album_art_url": song['album']['images'][0]['url'],
            "artist": artist_str,
            "preview_url": preview_url,
            "uri": song['uri']
        }
        if use_canvases:
            data["canvas_url"] = canvases[song['uri']] if song['uri'] in canvases.keys() else None
        else:
            data["canvas_url"] = None
        song_data.append(data)
    return song_data
