from __future__ import division

import datetime
import json
import os
import urllib

import omdb
from requests import HTTPError

import api_keys
import config

from langdetect import detect

omdb.set_default('apikey', api_keys.OMDB_API_KEY)

series = []
pageNb = 1
pageCount = 2

if not os.path.exists('out/'):
    os.makedirs('out/')

while True:
    with urllib.request.urlopen(
            config.SERIES_URL + config.paramPage + str(pageNb) + config.paramAPI + api_keys.TMDB_API_KEY) as url:
        data = json.loads(url.read().decode())
        print('Collecting series: ' + str(data['page']) + '/' + str(data['total_pages']))
        series.extend(data['results'])

        pageNb += 1
        if pageNb > data['total_pages']:
            break

with open('out/series_raw.json', 'w') as outfile:
    json.dump(series, outfile)

# with open('out/series_raw.json') as infile:
#     series = json.load(infile)

genre = {
    10759: ['Action', 'Adventure'],
    35: ['Comedy'],
    80: ['Crime'],
    99: ['Documentary'],
    18: ['Drama'],
    10751: ['Family'],
    10762: ['Kids'],
    9648: ['Mystery'],
    10763: ['News'],
    10764: ['Reality'],
    10765: ['Science Fiction', 'Fantasy'],
    10766: ['Soap'],
    10767: ['Talk'],
    10768: ['War & Politics'],
    37: ['Western'],
    28: ['Action'],
    12: ['Adventure'],
    14: ['Fantasy'],
    36: ['History'],
    27: ['Horror'],
    10402: ['Music'],
    10749: ['Romance'],
    878: ['Science Fiction'],
    10770: ['TV Movie'],
    53: ['Thriller'],
    10752: ['War & Politics'],
}

i = 1
iM = len(series)

for serie in series:
    serie['genres'] = []
    for genreId in serie['genre_ids']:
        if genreId in genre:
            serie['genres'].extend(genre[genreId])

    serie['genres'] = list(dict.fromkeys(serie['genres']))

    del serie['genre_ids']
    del serie['original_name']
    del serie['popularity']
    del serie['origin_country']
    del serie['vote_count']
    del serie['backdrop_path']
    del serie['original_language']
    del serie['vote_average']
    del serie['poster_path']
    del serie['id']
    del serie['overview']

    omdbDetails = {}
    try:
        if ('first_air_date' in serie) and (serie['first_air_date'] != ''):
            try:
                omdbDetails = omdb.get(title=serie['name'],
                                       year=datetime.datetime.strptime(serie['first_air_date'], '%Y-%m-%d').strftime(
                                           '%Y'),
                                       fullplot=True)
            except ValueError:
                pass
        else:
            omdbDetails = omdb.get(title=serie['name'],
                                   fullplot=True)
    except HTTPError as e:
        print(
            'Error while collecting the plot of ' + serie['name'] + '(' + serie['first_air_date'] + ')' + ': ' + str(e))

    if bool(omdbDetails):
        serie['overview'] = omdbDetails['plot']
    else:
        serie['overview'] = ''

    print('Collecting plots: ' + str(i) + '/' + str(iM))
    i += 1

print('Cleaning data...')

for serie in list(series):
    if serie['overview'] == '' or serie['overview'] == 'N/A' or serie['genres'] == [] or detect(
            serie['overview']) != 'en':
        series.remove(serie)

with open('out/series_OMDB_plot.json', 'w') as outfile:
    json.dump(series, outfile)
