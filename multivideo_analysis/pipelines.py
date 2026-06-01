"""
pipelines.py — shared data-loading and labeling functions.

Import in any notebook:
    %load_ext autoreload
    %autoreload 2
    from pipelines import build_labeled_audio, CANDIDATES, KEY_FEATURES, NON_REDUNDANT
"""

import os
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
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


# ─────────────────────────────────────────────────────────────────────────────
# Visual pipeline — extracted from visual_label.ipynb and visual_and_audio.ipynb
# ─────────────────────────────────────────────────────────────────────────────

def build_visual_features(video_name, features_dir=None):
    """
    Extract per-face features from a debate's visual pkl.

    Combines the feature extraction loops from:
    - visual_label.ipynb / visual_and_audio.ipynb cell 5
      (geometric ratios from facial landmarks + body matching)
    - corelations_updated.ipynb cell 10
      (all 8 emotion probabilities + raw pose keypoints)

    Returns
    -------
    df_people : pd.DataFrame
        One row per detected face per frame.
        Columns include Face_BBox, Face_X_Center, People_Count, Person_Index,
        all Ratio_* columns, all prob_* emotion columns, Pose_Keypoints, etc.
    """
    fd = features_dir or FEATURES_DIR

    def _dist(a, b):
        # visual_label.ipynb / visual_and_audio.ipynb cell 5: dist()
        return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    df = pd.read_pickle(os.path.join(fd, f'{video_name}_visual.pkl'))
    df['Video_Name']   = df['Frame'].str.extract(r'Frames/([^/]+)/')
    df['Frame_Number'] = df['Frame'].str.extract(r'frame_(\d+)\.jpg').astype(int)
    df = df.sort_values(['Video_Name', 'Frame_Number']).reset_index(drop=True)

    EMOTIONS = ['Anger', 'Contempt', 'Disgust', 'Fear',
                'Happiness', 'Neutral', 'Sadness', 'Surprise']

    master_data = []

    # visual_and_audio.ipynb cell 5: main per-frame loop
    for _, row in df.iterrows():
        frame_id     = row['Frame']
        frame_number = row['Frame_Number']
        faces        = row['Fer']
        poses        = row['Poses']

        if isinstance(faces, list) and 0 < len(faces) <= 3:
            valid_faces  = [f for f in faces if isinstance(f, dict) and 'bbox' in f]
            people_count = len(valid_faces)

            for i, face in enumerate(valid_faces):
                f_box    = face['bbox']
                f_width  = f_box[2] - f_box[0]
                f_height = f_box[3] - f_box[1]
                f_x_center = f_box[0] + f_width / 2
                f_y_center = f_box[1] + f_height / 2
                face_area  = f_width * f_height

                top_emotion = face.get('top_emotion')
                probs       = face.get('probabilities', {})
                probability = probs.get(top_emotion)
                landmarks   = face.get('landmarks')

                # ── ratio 1: face width / face height ────────────────────────
                ratio_face = f_width / f_height if f_height > 0 else None

                # ── landmark-based ratios (visual_label.ipynb cell 5) ─────────
                ratio_up_low_face = ratio_eyes_face = ratio_nose_face = None
                nose_tip = None
                if landmarks is not None:
                    try:
                        upper_face       = _dist(landmarks[10], landmarks[26])
                        lower_face       = _dist(landmarks[15], landmarks[31])
                        ratio_up_low_face = upper_face / lower_face if lower_face > 0 else None

                        dist_eyes       = _dist(landmarks[89], landmarks[39])
                        face_w          = _dist(landmarks[10], landmarks[26])
                        ratio_eyes_face = dist_eyes / face_w if face_w > 0 else None

                        dist_nose       = _dist(landmarks[72], landmarks[80])
                        face_l          = _dist(landmarks[72], landmarks[0])
                        ratio_nose_face = dist_nose / face_l if face_l > 0 else None

                        nose_tip = landmarks[86]
                    except (IndexError, TypeError):
                        pass

                # ── body matching + shoulder ratios ───────────────────────────
                matched_body_bbox      = None
                matched_pose_keypoints = None
                ratio_shoulder_face    = None
                ratio_nose_shoulder    = None

                if isinstance(poses, list):
                    for person in poses:
                        if isinstance(person, dict) and len(person) > 0:
                            b_box = person['bbox']
                            if (b_box[0] <= f_x_center <= b_box[2] and
                                    b_box[1] <= f_y_center <= b_box[3]):
                                matched_body_bbox      = b_box
                                matched_pose_keypoints = person.get('pose')
                                if matched_pose_keypoints is not None:
                                    l_sho = matched_pose_keypoints[5][0:2]
                                    r_sho = matched_pose_keypoints[6][0:2]
                                    shoulder_width = abs(r_sho[0] - l_sho[0])
                                    ratio_shoulder_face = shoulder_width / f_width if f_width > 0 else None
                                    if nose_tip is not None:
                                        d_l = _dist(nose_tip, l_sho)
                                        d_r = _dist(nose_tip, r_sho)
                                        ratio_nose_shoulder = d_r / d_l if d_l > 0.001 else 0
                                break

                master_data.append({
                    'Frame':        frame_id,
                    'Video_Name':   row['Video_Name'],
                    'Frame_Number': frame_number,
                    'People_Count': people_count,
                    'Person_Index': i,
                    'Face_X_Center': f_x_center,
                    'Face_Y_Center': f_y_center,
                    'Face_Area':    face_area,
                    'Face_BBox':    f_box,
                    'Body_BBox':    matched_body_bbox,
                    'Top_Emotion':  top_emotion,
                    'Emotion_Prob': probability,
                    'Landmarks':    landmarks,
                    'Pose_Keypoints': matched_pose_keypoints,
                    # geometric ratios (visual_label.ipynb cell 5)
                    'Ratio_Face':           ratio_face,
                    'Ratio_Up_Low_Face':    ratio_up_low_face,
                    'Ratio_Eyes_Face':      ratio_eyes_face,
                    'Ratio_Nose_Face':      ratio_nose_face,
                    'Ratio_Shoulder_Face':  ratio_shoulder_face,
                    'Ratio_Nose_Shoulder':  ratio_nose_shoulder,
                    # all 8 emotion probabilities (corelations_updated.ipynb cell 10)
                    **{f'prob_{e}': probs.get(e) for e in EMOTIONS},
                })
        else:
            master_data.append({
                'Frame': frame_id, 'Video_Name': row['Video_Name'],
                'Frame_Number': frame_number, 'People_Count': 0, 'Person_Index': None,
                'Face_X_Center': None, 'Face_Y_Center': None, 'Face_Area': None,
                'Face_BBox': None, 'Body_BBox': None, 'Top_Emotion': None,
                'Emotion_Prob': None, 'Landmarks': None, 'Pose_Keypoints': None,
                'Ratio_Face': None, 'Ratio_Up_Low_Face': None,
                'Ratio_Eyes_Face': None, 'Ratio_Nose_Face': None,
                'Ratio_Shoulder_Face': None, 'Ratio_Nose_Shoulder': None,
                **{f'prob_{e}': None for e in EMOTIONS},
            })

    return pd.DataFrame(master_data)


def label_debate_visual(video_name, data_audio, features_dir=None):
    """
    Classify every face in every frame of a debate as candidate_left,
    candidate_right, or moderator, then resolve to real candidate names
    using audio speaker labels.

    Exact logic from:
    - visual_label.ipynb / visual_and_audio.ipynb cells 9-17  (GMM clustering)
    - visual_and_audio.ipynb cells 29-33                      (name resolution)

    Returns
    -------
    DataFrame with columns matching labels.pkl:
        Video_Name, Frame_Number, Person_Index, final_label, final_name, speaker
    """
    RATIO_FEATURES = ['Ratio_Face', 'Ratio_Up_Low_Face', 'Ratio_Eyes_Face',
                      'Ratio_Nose_Face', 'Ratio_Shoulder_Face', 'Ratio_Nose_Shoulder']
    EMPTY = pd.DataFrame(columns=['Video_Name', 'Frame_Number', 'Person_Index',
                                   'final_label', 'final_name', 'speaker'])

    df_people = build_visual_features(video_name, features_dir)

    # ── visual_and_audio.ipynb cell 9: 2-person frames, label left/right by bbox ─
    df_two_people = df_people[df_people['People_Count'] == 2].copy()
    index_to_side = {}
    for frame_num, people in df_two_people.groupby('Frame_Number'):
        sorted_index = people.sort_values('Face_X_Center').index
        index_to_side[sorted_index[0]] = 'left'
        index_to_side[sorted_index[1]] = 'right'
    df_two_people['side'] = df_two_people.index.map(index_to_side)

    # ── visual_and_audio.ipynb cell 11: separation_accuracy + select_ratios ──────
    def separation_accuracy(col_feature):
        d = df_two_people[['side', col_feature]].dropna()
        if len(d) == 0:
            return 0.0
        mean_left  = d[d['side'] == 'left' ][col_feature].mean()
        mean_right = d[d['side'] == 'right'][col_feature].mean()
        if pd.isna(mean_left) or pd.isna(mean_right):
            return 0.0
        thr  = (mean_left + mean_right) / 2
        pred = np.where(d[col_feature] >= thr,
                        'left' if mean_left > mean_right else 'right',
                        'right' if mean_left > mean_right else 'left')
        return (pred == d['side'].values).mean()

    acc_dict = {r: separation_accuracy(r) for r in RATIO_FEATURES}

    def select_ratios(acc_dict, identity_ratios, min_acc=0.65):
        # exact copy of visual_and_audio.ipynb cell 11: select_ratios()
        ratios   = {}
        high_acc = 0
        key_acc  = None
        for rac in acc_dict.keys():
            if acc_dict[rac] >= min_acc:
                ratios[rac] = acc_dict[rac]
            if acc_dict[rac] > high_acc:
                high_acc = acc_dict[rac]
                key_acc  = rac
        if len(ratios) == 0:
            ratios[key_acc] = high_acc
        for rac in identity_ratios:
            if rac in ratios.keys():
                return ratios           # at least one identity ratio present
        best_id  = None
        best_acc = 0
        for rac in identity_ratios:
            if acc_dict[rac] > best_acc:
                best_acc = acc_dict[rac]
                best_id  = rac
        ratios[best_id] = best_acc
        return ratios

    ratios_to_use = select_ratios(acc_dict, RATIO_FEATURES[:-1], min_acc=0.7)
    chosen_ratios = list(ratios_to_use.keys())

    # ── visual_and_audio.ipynb cell 13: keep faces with all chosen ratios ────────
    chosen_mask = df_people[chosen_ratios].notna().all(axis=1)
    data = df_people[chosen_mask].copy()
    if len(data) == 0:
        return EMPTY

    # z-score (cell 15 uses its own scaler)
    scaler = StandardScaler().fit(data[chosen_ratios])
    Xz     = scaler.transform(data[chosen_ratios])

    # known labels from bbox position (cells 13)
    data['known'] = None
    for frame, group in data[data['People_Count'] == 2].groupby('Frame_Number'):
        order = group.sort_values('Face_X_Center').index
        if len(order) == 2:
            data.loc[order[0], 'known'] = 'left'
            data.loc[order[1], 'known'] = 'right'
    for frame, group in data[data['People_Count'] == 3].groupby('Frame_Number'):
        order = group.sort_values('Face_X_Center').index
        if len(order) == 3:
            data.loc[order[0], 'known'] = 'left'
            data.loc[order[1], 'known'] = 'moderator'
            data.loc[order[2], 'known'] = 'right'

    # ── visual_and_audio.ipynb cell 15: GMM clustering ───────────────────────────
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
    gmm.fit(Xz)
    data['cluster']      = gmm.predict(Xz)
    data['cluster_conf'] = gmm.predict_proba(Xz).max(axis=1)

    two_rows = data[data['People_Count'] == 2]
    if ('known' in two_rows.columns and
            'left' in two_rows['known'].values and
            'right' in two_rows['known'].values):
        tab          = pd.crosstab(two_rows['cluster'], two_rows['known'])
        left_cluster = tab['left'].idxmax() if 'left' in tab.columns else 0
        remaining    = [c for c in tab.index if c != left_cluster]
        right_cluster = (tab.loc[remaining, 'right'].idxmax()
                         if remaining and 'right' in tab.columns
                         else (remaining[0] if remaining else 1))
    else:
        left_cluster, right_cluster = 0, 1

    name_map = {left_cluster: 'candidate_left', right_cluster: 'candidate_right'}
    for c in data['cluster'].dropna().unique():
        name_map.setdefault(int(c), 'moderator')

    thrs_conf = 0.75
    data['person'] = data['cluster'].map(name_map)
    data.loc[data['cluster_conf'] < thrs_conf, 'person'] = 'uncertain'

    # ── visual_and_audio.ipynb cell 17: final_label assignment ───────────────────
    known_to_label = {'left': 'candidate_left', 'right': 'candidate_right',
                      'moderator': 'moderator'}
    data['final_label'] = data['known'].map(known_to_label)
    gap = data['final_label'].isna()
    data.loc[gap, 'final_label'] = data.loc[gap, 'person']

    MOD_OVERRIDE_CONF = 0.8
    two  = data['People_Count'] == 2
    flip = two & (data['person'] == 'moderator') & (data['cluster_conf'] >= MOD_OVERRIDE_CONF)
    data.loc[flip, 'final_label'] = 'moderator'

    for frame_n, group in data.groupby('Frame_Number'):
        moderators = group[group['final_label'] == 'moderator'].index
        if len(moderators) >= 2:
            data.loc[moderators, 'final_label'] = 'uncertain'

    # ── visual_and_audio.ipynb cell 29: build frame→speaker map from audio ───────
    debate_audio = (data_audio[data_audio['video'] == video_name]
                    .copy()
                    .sort_values('time stamp')
                    .reset_index(drop=True))

    debate_audio['frame_start'] = (debate_audio['time stamp']
                                   .apply(math.floor) + 1).astype(int)
    debate_audio['frame_end']   = (debate_audio['time stamp'] +
                                   debate_audio['duration']).apply(math.ceil).astype(int)
    debate_audio['frames']      = debate_audio.apply(
        lambda r: list(range(r['frame_start'], r['frame_end'] + 1)), axis=1)

    # ── visual_and_audio.ipynb cell 31: merge audio speaker into visual data ─────
    audio_df = debate_audio[['video', 'frames', 'speaker']].explode('frames')
    audio_df = audio_df.rename(columns={'video': 'Video_Name', 'frames': 'Frame_Number'})
    audio_df['Frame_Number'] = audio_df['Frame_Number'].astype(int)
    audio_df = audio_df.drop_duplicates(subset=['Video_Name', 'Frame_Number'])

    data = data.drop(columns=['speaker', 'speaker_x', 'speaker_y'], errors='ignore')
    data['Video_Name'] = data['Frame'].str.extract(r'Frames/([^/]+)/')
    data = data.merge(audio_df, on=['Video_Name', 'Frame_Number'], how='left')

    # ── visual_and_audio.ipynb cell 33: resolve left/right → real candidate names ─
    one_person = data[(data['People_Count'] == 1) & data['speaker'].notna()]
    left_name = right_name = 'unknown'

    if len(one_person) > 0:
        tab_names  = pd.crosstab(one_person['final_label'], one_person['speaker'])
        candidates_audio = [c for c in tab_names.columns if c != 'host']
        if 'candidate_left' in tab_names.index and candidates_audio:
            left_name = tab_names.loc['candidate_left', candidates_audio].idxmax()
            remaining_names = [c for c in candidates_audio if c != left_name]
            if remaining_names and 'candidate_right' in tab_names.index:
                right_name = tab_names.loc['candidate_right', remaining_names].idxmax()
            elif remaining_names:
                right_name = remaining_names[0]

    real = {'candidate_left': left_name, 'candidate_right': right_name,
            'moderator': 'moderator', 'uncertain': 'uncertain'}
    data['final_name'] = data['final_label'].map(real)

    return data[['Video_Name', 'Frame_Number', 'Person_Index',
                 'final_label', 'final_name', 'speaker']].copy()


def build_labels_all(debate_videos, data_audio, features_dir=None):
    """
    Build visual identity labels for every debate in debate_videos.
    Calls label_debate_visual() per debate and concatenates the results.

    Returns
    -------
    DataFrame in the same schema as labels.pkl but covering all debates.
    """
    dfs = []
    for v in debate_videos:
        print(f'  labeling {v} ...')
        try:
            df = label_debate_visual(v, data_audio, features_dir)
            dfs.append(df)
        except Exception as e:
            print(f'  WARNING: {v} skipped — {e}')
    if not dfs:
        return pd.DataFrame(columns=['Video_Name', 'Frame_Number', 'Person_Index',
                                     'final_label', 'final_name', 'speaker'])
    result = pd.concat(dfs, ignore_index=True)
    print(f'build_labels_all: {len(result)} face-frame rows | '
          f'{result["Video_Name"].nunique()} debates')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Joint audio+visual dataset — extracted from corelations_updated.ipynb
# ─────────────────────────────────────────────────────────────────────────────

def build_seg_all(debate_videos, data_audio, labels_all, features_dir=None):
    """
    Build the segment-level joint audio+visual+emotion+pose dataset
    across all debates.

    Exact logic from corelations_updated.ipynb:
    - cell 8  : frame → audio segment join
    - cell 10 : per-face feature extraction  (via build_visual_features)
    - cell 11 : pose features from keypoints (extract_pose_features)
    - cell 13 : identity join + speaking_face flag
    - cell 16 : segment-level aggregation

    Parameters
    ----------
    debate_videos : list of str
    data_audio    : DataFrame returned by build_labeled_audio()
    labels_all    : DataFrame returned by build_labels_all()

    Returns
    -------
    seg_all : pd.DataFrame
        One row per audio segment that had a visible, identified speaking face.
        Columns: video, segment_id, final_name, n_frames,
                 NON_REDUNDANT acoustic features,
                 prob_* emotion columns, pose feature columns.
    """
    POSE_COLS    = ['shoulder_slope', 'body_openness', 'head_tilt',
                    'wrist_height', 'torso_height']
    EMOTION_COLS = [f'prob_{e}' for e in
                    ['Anger', 'Contempt', 'Disgust', 'Fear',
                     'Happiness', 'Neutral', 'Sadness', 'Surprise']]

    # corelations_updated.ipynb cell 11: extract_pose_features()
    def _extract_pose_features(keypoints):
        if keypoints is None:
            return pd.Series({c: None for c in POSE_COLS})
        kp = np.array(keypoints)          # 17×3: [x, y, visibility]
        l_shoulder = kp[5];  r_shoulder = kp[6]
        l_wrist    = kp[9];  r_wrist    = kp[10]
        l_hip      = kp[11]; r_hip      = kp[12]
        nose       = kp[0]
        shoulder_slope = r_shoulder[1] - l_shoulder[1]
        body_openness  = abs(r_shoulder[0] - l_shoulder[0])
        shoulder_mid_x = (l_shoulder[0] + r_shoulder[0]) / 2
        head_tilt      = nose[0] - shoulder_mid_x
        shoulder_mid_y = (l_shoulder[1] + r_shoulder[1]) / 2
        wrist_height   = shoulder_mid_y - ((l_wrist[1] + r_wrist[1]) / 2)
        hip_mid_y      = (l_hip[1] + r_hip[1]) / 2
        torso_height   = abs(hip_mid_y - shoulder_mid_y)
        return pd.Series({'shoulder_slope': shoulder_slope,
                          'body_openness':  body_openness,
                          'head_tilt':      head_tilt,
                          'wrist_height':   wrist_height,
                          'torso_height':   torso_height})

    # corelations_updated.ipynb cell 16: mode_or_nan()
    def _mode_or_nan(s):
        m = s.mode()
        return m.iloc[0] if len(m) else np.nan

    all_segs = []

    for v in debate_videos:
        print(f'  seg {v} ...')

        # ── build_visual_features → df_master_valid (cells 10-11) ────────────
        try:
            df_people = build_visual_features(v, features_dir)
        except Exception as e:
            print(f'  WARNING: {v} visual failed — {e}'); continue

        has_pose    = df_people['Pose_Keypoints'].notna()
        df_valid    = df_people[has_pose].copy()
        if len(df_valid) == 0:
            continue

        pose_feats  = df_valid['Pose_Keypoints'].apply(_extract_pose_features)
        df_valid    = pd.concat([df_valid, pose_feats], axis=1)

        # rename to lowercase convention used in corelations_updated.ipynb
        df_valid = df_valid.rename(columns={
            'Frame_Number': 'frame_number',
            'Person_Index': 'person_index',
        })

        # ── frame → audio join (corelations_updated cell 8) ──────────────────
        df_audio_v           = data_audio[data_audio['video'] == v].copy()
        df_audio_v['time_end'] = df_audio_v['time stamp'] + df_audio_v['duration']

        frame_rows = []
        for t in sorted(df_valid['frame_number'].unique()):
            active = df_audio_v[
                (df_audio_v['time stamp'] <= t) & (df_audio_v['time_end'] > t)]
            if len(active) == 1:
                seg_row = active.iloc[0]
                r = {'frame_number': int(t), 'has_audio': True,
                     'segment_id': int(active.index[0])}
                r.update({c: seg_row[c] for c in NON_REDUNDANT})
            else:
                r = {'frame_number': int(t), 'has_audio': False, 'segment_id': -1}
                r.update({c: np.nan for c in NON_REDUNDANT})
            frame_rows.append(r)

        audio_by_frame = pd.DataFrame(frame_rows)

        # ── identity join + speaking_face flag (corelations_updated cell 13) ──
        labels_v = (labels_all[labels_all['Video_Name'] == v]
                    .rename(columns={'Frame_Number': 'frame_number',
                                     'Person_Index': 'person_index'})
                    .dropna(subset=['person_index'])
                    .copy())
        if len(labels_v) == 0:
            continue

        labels_v['frame_number'] = labels_v['frame_number'].astype(int)
        labels_v['person_index'] = labels_v['person_index'].astype(int)
        df_valid['frame_number'] = df_valid['frame_number'].astype(int)
        df_valid['person_index'] = df_valid['person_index'].astype(int)

        df_all = df_valid.merge(
            labels_v[['frame_number', 'person_index', 'final_name', 'speaker']],
            on=['frame_number', 'person_index'], how='left')
        df_all['speaker'] = df_all['speaker'].replace({'host': 'moderator'})

        df_all = df_all.merge(audio_by_frame, on='frame_number', how='left')
        df_all['has_audio']     = df_all['has_audio'].fillna(False)
        df_all['segment_id']    = df_all['segment_id'].fillna(-1).astype(int)
        df_all['speaking_face'] = (df_all['has_audio'] &
                                   (df_all['final_name'] == df_all['speaker']))

        # ── segment aggregation (corelations_updated cell 16) ─────────────────
        spk = df_all[df_all['speaking_face']].copy()
        if len(spk) == 0:
            continue

        existing_emotion = [c for c in EMOTION_COLS if c in spk.columns]
        existing_pose    = [c for c in POSE_COLS    if c in spk.columns]
        existing_audio   = [c for c in NON_REDUNDANT if c in spk.columns]

        agg = {c: 'mean' for c in existing_emotion + existing_pose}
        agg.update({c: 'first' for c in existing_audio})
        agg['final_name']   = 'first'
        agg['top_emotion']  = _mode_or_nan
        agg['frame_number'] = 'count'

        seg = (spk.groupby('segment_id').agg(agg)
                  .rename(columns={'frame_number': 'n_frames'})
                  .reset_index())
        seg['video'] = v
        all_segs.append(seg)

    if not all_segs:
        return pd.DataFrame()

    seg_all = pd.concat(all_segs, ignore_index=True)
    print(f'build_seg_all: {len(seg_all)} segments | '
          f'{seg_all["video"].nunique()} debates | '
          f'{seg_all["final_name"].nunique()} speakers')
    return seg_all
