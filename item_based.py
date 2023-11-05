from tqdm import tqdm
import numpy as np
from dataset import get_mappings
from joblib import dump
from functools import lru_cache


def compute_overlap_matrix(user_to_song, song_list):
    num_songs = len(song_list)
    overlap_matrix = np.zeros((num_songs, num_songs))

    for i, song1 in enumerate(song_list):
        for j, song2 in enumerate(song_list):
            if i <= j:  # Only compute for one half since the matrix is symmetric
                users_song1 = user_to_song.get(song1, set())
                users_song2 = user_to_song.get(song2, set())
                overlap = len(users_song1.intersection(users_song2))
                overlap_matrix[i, j] = overlap
                overlap_matrix[j, i] = overlap

    return overlap_matrix


# x and y are songs to be compared.
# user_to_song is a mapping of the set of users to a set of songs they listen to.

def calc_weight(x, y, overlap_matrix, song_list):
    index_x = song_list.index(x)
    index_y = song_list.index(y)

    listened_x = np.sum(overlap_matrix[index_x])
    listened_y = np.sum(overlap_matrix[index_y])

    overlap = overlap_matrix[index_x, index_y]

    denom = np.sqrt(listened_x) * np.sqrt(listened_y)

    if denom == 0:
        return 0

    return overlap / denom


def score_song(user, song, user_to_song, song_list, overlap_matrix):
    print(f"scoring song: {song}")
    indicator = np.array([1 if i in user_to_song[user]
                         and not song == i else 0 for i in song_list])
    weights = np.array([calc_weight(i, song, overlap_matrix,
                       song_list) if i != song else 0 for i in song_list])
    return np.sum(np.dot(indicator, weights))


def build_user_to_rec():
    user_to_song, song_meta, song_list = get_mappings()
    user_to_rec = {}
    overlap_matrix = compute_overlap_matrix(user_to_song, song_list)

    for user in tqdm(user_to_song, desc="Processing users"):
        scores = [(song, score_song(user, song, user_to_song, song_list, overlap_matrix))
                  for song in song_list]
        scores.sort(key=lambda x: x[1])
        user_to_rec[user] = np.array(scores)

    dump(user_to_rec, 'user_to_rec.joblib')


build_user_to_rec()
