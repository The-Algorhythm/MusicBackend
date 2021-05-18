from bs4 import BeautifulSoup, NavigableString
import requests
import json
from urllib.parse import quote, unquote, urlparse, parse_qs
import pickle
import time


with open('../data/enc_map.pickle', 'rb') as f:
    enc_map = pickle.load(f)

data = []
count = 0
for genre in enc_map.keys():
    start = time.time()
    url = 'https://everynoise.com/everynoise1d.cgi?scope=all&root=' + quote(genre)
    r = requests.get(url)
    soup = BeautifulSoup(r.content, features="html.parser")
    for row in soup.find_all("tr"):
        overlap = -1
        acoustic_distance = -1
        name = 'Error'
        rank = -1
        for tag in row.descendants:
            try:
                if tag.attrs['align'] == 'right' and tag.attrs['class'] == ['note']:
                    similarity_attrs = tag.attrs['title'].split(',')
                    overlap = float(similarity_attrs[0].split(':')[1])
                    acoustic_distance = float(similarity_attrs[1].split(':')[1])
            except (KeyError, AttributeError):
                pass
            if type(tag) == NavigableString:
                if tag.isnumeric():
                    rank = int(tag)
                elif len(tag) > 1:
                    name = str(tag)
        if name in enc_map.keys():
            data.append((enc_map[genre], enc_map[name], rank, overlap, acoustic_distance))
    print(f"Finished processing \"{genre}\" after {time.time() - start}s")

    if count % 15 == 0:
        print(f"Saving checkpoint after processing {count + 1} genres")
        with open('../data/genre_similarity.pickle', 'wb') as data_pickle:
            pickle.dump(data, data_pickle)
    count += 1
