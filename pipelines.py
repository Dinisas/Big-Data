"""
pipelines.py — shared data-loading and labeling functions.

Import in any notebook:
    %load_ext autoreload
    %autoreload 2
    from pipelines import build_labeled_audio, CANDIDATES, KEY_FEATURES, NON_REDUNDANT
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

FEATURES_DIR = 'Project_Features'

CANDIDATES = [
    'Cotrim_Figueiredo', 'Filipe', 'Gouveia_Melo',
    'Marques_Mendes', 'Martins', 'Pinto', 'Seguro', 'Ventura'
]

KEY_FEATURES = [
    'meanF0Hz',         # pitch
    'stdevF0Hz',        # pitch variability
    'HNR',              # voice clarity
    'localJitter',      # pitch irregularity
    'localShimmer',     # amplitude irregularity
    'speechrate',       # syllables/sec including pauses
    'articulationrate', # syllables/sec excluding pauses
    'npause',           # number of pauses
    'asd',              # average syllable duration
    'f1_mean',          # first formant
    'f2_mean',          # second formant
    'fdisp',            # formant dispersion
]

NON_REDUNDANT = [
    'meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter', 'localShimmer',
    'speechrate', 'npause', 'f1_mean', 'f2_mean', 'fdisp',
]

# Consistent candidate colours (left=progressive → right=nationalist)
CAND_COLOR = {
    'Martins':           '#e41a1c',
    'Filipe':            '#984ea3',
    'Pinto':             '#ff7f00',
    'Seguro':            '#377eb8',
    'Gouveia_Melo':      '#4daf4a',
    'Cotrim_Figueiredo': '#a65628',
    'Marques_Mendes':    '#f781bf',
    'Ventura':           '#222222',
}

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_embedding(s):
    return s if isinstance(s, np.ndarray) else np.fromstring(s.strip('[]'), sep=' ')


def extract_candidates(video_name):
    """Return (candidateA, candidateB) parsed from a debate video filename."""
    for c in CANDIDATES:
        if video_name.startswith(c):
            rest = video_name[len(c) + len('_vs_'):]
            for c2 in CANDIDATES:
                if rest.startswith(c2):
                    return c, c2
    return None, None


def _cluster_and_label(data, debate_videos):
    """k=3 per debate + cross-debate centroid anchoring → speaker column."""
    debate_centroids = {}
    cluster_col = np.full(len(data), -1, dtype=int)

    for v in debate_videos:
        idx = data.index[data['video'] == v]
        E = np.stack(data.loc[idx, 'speak_embeddings'].values)
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_col[idx] = km.fit_predict(E)
        debate_centroids[v] = km.cluster_centers_
    data['cluster_k3'] = cluster_col

    # Pass 1: best-matching centroid per candidate per debate
    def _best_centroids(candidate, dc):
        their = [v for v in dc if candidate in v]
        if len(their) < 2:
            return {}
        best = {}
        for video in their:
            others = [v for v in their if v != video]
            cen = dc[video]
            scores = [
                sum(cdist(cen[ci:ci+1], dc[ov], metric='cosine')[0].min()
                    for ov in others)
                for ci in range(3)
            ]
            best[video] = cen[int(np.argmin(scores))]
        return best

    global_centroids = {}
    for cand in CANDIDATES:
        per_debate = _best_centroids(cand, debate_centroids)
        if per_debate:
            global_centroids[cand] = np.mean(np.stack(list(per_debate.values())), axis=0)

    # Pass 2: assign speaker label per segment
    speaker_col = np.full(len(data), 'unknown', dtype=object)
    for v in debate_videos:
        cA, cB = extract_candidates(v)
        if cA is None:
            continue
        idx = data.index[data['video'] == v]
        cen = debate_centroids[v]
        dA = cdist(cen, global_centroids[cA].reshape(1, -1), metric='cosine').flatten()
        dB = cdist(cen, global_centroids[cB].reshape(1, -1), metric='cosine').flatten()
        avail = {0, 1, 2}
        c2s = {}
        bA = int(np.argmin([dA[i] if i in avail else np.inf for i in range(3)]))
        c2s[bA] = cA; avail.remove(bA)
        bB = int(np.argmin([dB[i] if i in avail else np.inf for i in range(3)]))
        c2s[bB] = cB; avail.remove(bB)
        c2s[avail.pop()] = 'host'
        speaker_col[idx] = [c2s[l] for l in data.loc[idx, 'cluster_k3'].values]

    data['speaker'] = speaker_col
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_labeled_audio(features_dir=None):
    """
    Load all debate audio pkls, run k=3 speaker clustering per debate,
    apply cross-debate centroid anchoring to assign candidate names, and
    return a DataFrame with 'speaker' and 'party' columns ready for analysis.

    Returns
    -------
    data : pd.DataFrame
        One row per audio segment. Extra columns vs raw pkl:
        'video', 'cluster_k3', 'speaker', 'party'.
    """
    fd = features_dir or FEATURES_DIR

    audio_files = sorted(f for f in os.listdir(fd) if f.endswith('_audio.pkl'))
    dfs = []
    for f in audio_files:
        df = pd.read_pickle(os.path.join(fd, f))
        df['video'] = f.replace('_audio.pkl', '')
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data['speak_embeddings'] = data['speak_embeddings'].apply(parse_embedding)

    parties = pd.read_pickle(os.path.join(fd, 'candidate_party.pkl'))
    candidate_to_party = dict(zip(parties['Candidate'], parties['Party']))

    debate_videos = sorted(v for v in data['video'].unique() if 'vs' in v)
    data = _cluster_and_label(data, debate_videos)
    data['party'] = data['speaker'].map(candidate_to_party).fillna('host')

    print(f'build_labeled_audio: {len(data)} segments | '
          f'{len(debate_videos)} debates | '
          f'{data["speaker"].nunique()} unique speakers')
    return data
