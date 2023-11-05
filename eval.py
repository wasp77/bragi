import numpy as np


def precision_at_k(recs, songs, k):
    listened = np.array([1 if rec in songs else 0 for rec in recs[:k]])
    return np.sum(listened) / k


def avg_precision(recs, songs, t):
    recs = recs[:t]
    precisions = np.array([precision_at_k(recs, songs, k+1) for k, rec in enumerate(recs) if rec in songs])
    n_relevant = len(set(recs) & set(songs))
    if n_relevant == 0:
        return 0.0
    return np.sum(precisions) / n_relevant


def mean_avg_precision(user_to_recs, user_to_song, t):
    aps = np.array([avg_precision(user_to_recs[user], user_to_song[user], t)
                   for user in user_to_song])
    return np.mean(aps)
