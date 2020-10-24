from google_sheets import get_creds, SheetsApiException, create_creds
from googleapiclient.discovery import build

from django.http import JsonResponse
from django.http import Http404, HttpResponse

import re
import os

TOKEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../token.pickle')
creds = None
service = None
try:
    creds = get_creds(TOKEN_DIR)
    service = build('sheets', 'v4', credentials=creds)
except SheetsApiException:
    pass


def get_sheet_data(spreadsheet_id, sheet_name, artist_range, album_range):
    if service is None:
        raise SheetsApiException

    sheet = service.spreadsheets()

    album_data_range = f"{sheet_name}!{album_range}"
    artist_data_range = f"{sheet_name}!{artist_range}"
    album_values = sheet.values().get(spreadsheetId=spreadsheet_id, range=album_data_range).execute().get('values', [])
    artist_values = sheet.values().get(spreadsheetId=spreadsheet_id, range=artist_data_range).execute().get('values', [])

    # Validate data
    if not album_values:
        raise SheetsApiException('No data found in albums range')
    if not artist_values:
        raise SheetsApiException('No data found in artists range')
    if len(album_values) != len(artist_values):
        raise SheetsApiException('Mismatch between number of albums and number of artists')

    results = []
    for i in range(len(album_values)):
        album = album_values[i][0]
        artist = artist_values[i][0]
        results.append([artist, album])
    return results


def extract_spreadsheet_id(url):
    begin, end = re.search('/d/.*/', url).regs[0]
    return url[begin + len('/d/'): end-1]


def check_sheets_setup(request):
    response = get_sheet_data(extract_spreadsheet_id('https://docs.google.com/spreadsheets/d/12UjuH1FGgulN1GTiegGzmxtEn6Zao4QHNz4CBBRVGYM/edit#gid=0'), 'Sheet1', 'A2:A', 'B2:B')
    return JsonResponse(response, safe=False)


def create_token(request):
    global creds
    global service

    if not request.user.is_superuser:
        raise Http404
    else:
        if creds is None:
            create_creds()
            creds = get_creds(TOKEN_DIR)
            service = build('sheets', 'v4', credentials=creds)
            return HttpResponse()
