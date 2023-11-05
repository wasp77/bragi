import pandas as pd
from joblib import dump, load
import os


def build_mapping(df_train, df_metadata):
    df_metadata.drop_duplicates(subset='song_id', keep='first', inplace=True)

    df_metadata.set_index('song_id', inplace=True)

    user_to_song = df_train.groupby('user_id')['song_id'].apply(set).to_dict()
    song_to_meta = df_metadata[['artist', 'song']].to_dict(orient='index')
    song_list = df_metadata.index.tolist()

    dump(song_list, 'song_list.joblib')
    dump(user_to_song, 'user_to_song.joblib')
    dump(song_to_meta, 'song_to_metadata.joblib')

    return user_to_song, song_to_meta


def load_dataset():
    print('loading dataset...')
    df_train = pd.read_csv('./train.csv')
    df_metadata = pd.read_csv('./metadata.csv')
    print('dataset loaded!')
    return df_train, df_metadata


def get_mappings(small=True):
    if not os.path.isfile('./user_to_song.joblib'):
        train, metadata = load_dataset()
        return build_mapping(train, metadata)

    if small:
        user_to_song = load('user_to_song_sm.joblib')
        song_list = load('song_list_sm.joblib')
    else:
        user_to_song = load('user_to_song.joblib')
        song_list = load('song_list.joblib')
    song_meta = load('song_to_metadata.joblib')
    return user_to_song, song_meta, song_list

# user_to_song, song_meta, song_list = get_mappings()

# print(f"number of users: {len(user_to_song.keys())}")
# print(f"number of songs: {len(song_list)}")

# count = 0
# for user in user_to_song:
#     for song in user_to_song[user]:
#         count += 1

# print(f"Total songs for users: {count}")

# songs = load('song_list.joblib')
# new_len = int(0.01 * len(songs))
# dump(songs[:new_len], 'song_list_sm.joblib')


# from time import time

# start = time()
# user_to_song = load('user_to_song.joblib')
# end = time()
# print(end - start)
# print(len(user_to_song.keys()))

# from itertools import islice

# def extract_portion(d, portion):
#     n_items = round(len(d) * portion)
#     return dict(islice(d.items(), n_items))

# result = extract_portion(user_to_song, 0.001)
# dump(result, 'user_to_song_sm.joblib')


# Initialize an empty dictionary for user_to_song

# df_user_song = pd.read_csv('train_triplets.txt', sep='\t', names=['user_id', 'song_id', 'play_count'])
# df_song_meta = pd.read_csv('unique_tracks.txt', sep='<SEP>', names=['track_id', 'song_id', 'artist', 'song'], engine='python')
# df_user_song.to_csv('train.csv', index=False)
# df_song_meta.to_csv('metadata.csv', index=False)
