"""
    A map-reduce that extracts data on tracks from the Million Song Database with one track per line, where each line is represented by 54 EchoNest fields as described here:     
    http://labrosa.ee.columbia.edu/millionsong/pages/field-list

    A track is represented as a dictionary of the following:
    [track id] - string
    [title] - string
    [artist terms] - string array
    [artist name] - string
    [danceability] - float
    [energy] - float
    [song hotttnesss] - float
    [tempo] - float
    [year] - int
"""

from mrjob.job import MRJob
import sys
import pprint
import math

def load_track(line):
    """ Loads a track from a single line """
    t = {}

    f = line.split('\t')
    if len(f) == 54:
        t['track_id'] = f[0]
        t['title'] = f[51]
        t['artist_terms'] = f[14]
        t['artist_name'] = f[12]
        t['danceability'] = float(f[22])
        t['energy'] = float(f[25])
        hotness = float(f[43])
        if hotness == float('nan'):
            hotness = float(0)
        t['song_hotttnesss'] = hotness
        t['tempo'] = float(f[48])
        t['year'] = int(f[53])
        return t
    else:
        # ensures the correct number of fields loaded
        print 'mismatched fields, found', len(f), 'should have 54'
        return None

class MRSongs(MRJob):

    # maps outputting relevant fields for each track
    def mapper(self, _, line):
        t = load_track(line)
        if t:
            print t['track_id'],'\t', t['title'],'\t',t['artist_terms'],'\t',t['artist_name'],'\t',t['danceability'],'\t',t['energy'],'\t',t['song_hotttnesss'],'\t',t['tempo'],'\t',t['year']

    # no need for a reducer

if __name__ == '__main__':
    MRSongs.run()