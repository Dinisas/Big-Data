from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_WEIGHTS = {
    "semantic_shift": 0.35,
    "ocr_headline_shift": 0.25,
    "ocr_layout_shift": 0.15,
}

HEADLINE_COLUMNS = [
    "video", "headline", "start_s", "end_s", "duration_s", "observations",
    "mean_conf", "mean_x", "mean_y",
]

PT_STOPWORDS = {
    "a", "ao", "aos", "as", "ate", "com", "como", "da", "das", "de", "do",
    "dos", "e", "em", "entre", "era", "esta", "estao", "foi", "ha", "isso",
    "isto", "mais", "mas", "mesmo", "na", "nas", "no", "nos", "o", "os",
    "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "que", "se",
    "sem", "ser", "sua", "sao", "tambem", "tem", "uma", "um", "vai",
    "ter", "ja", "nao", "hoje", "agora", "sobre", "porque", "quando",
    "onde", "foram", "esta", "este", "esta", "todos", "todo", "toda",
    "num", "numa", "lhe", "lhes", "ele", "ela", "eles", "elas", "nosso",
    "nossa", "seu", "sua", "suas", "seus", "pode", "podem", "tambem",
    "anos", "parte", "caso", "momento", "pessoas", "pessoa",
}

BOILERPLATE_RE = re.compile(
    r"(?i)\b("
    r"rtp|tvi|telejornal|jornal|direto|directo|noticias|noticias|"
    r"seg|ter|qua|qui|sex|sab|sabado|dom|domingo"
    r")\b"
)
TIME_RE = re.compile(r"\b\d{1,2}[:h]\d{2}\b")
DATE_RE = re.compile(r"(?i)\b(seg|ter|qua|qui|sex|sab|s[aá]b|dom)\.?\s*\d{1,2}\b")
TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"[a-zA-ZÀ-ÿ]+(?:-[a-zA-ZÀ-ÿ]+)?")


@dataclass(frozen=True)
class SegmentationConfig:
    features_dir: str | Path = "Project_Features"
    ocr_conf_threshold: float = 0.60
    headline_similarity_threshold: float = 0.78
    headline_max_gap_s: float = 8.0
    headline_min_observations: int = 2
    headline_min_duration_s: float = 3.0
    semantic_window_segments: int = 3
    boundary_min_gap_s: float = 35.0
    min_item_duration_s: float = 20.0
    snap_window_s: float = 10.0
    max_topic_clusters: int = 8


def list_newscast_videos(features_dir: str | Path = "Project_Features") -> list[str]:
    features_path = Path(features_dir)
    speech = {p.name.replace("_speech.pkl", "") for p in features_path.glob("Telejornal_*_speech.pkl")}
    ocr = {p.name.replace("_ocr.pkl", "") for p in features_path.glob("Telejornal_*_ocr.pkl")}
    missing_ocr = sorted(speech - ocr)
    missing_speech = sorted(ocr - speech)
    if missing_ocr or missing_speech:
        raise AssertionError(
            "Mismatched Telejornal speech/OCR files. "
            f"Missing OCR: {missing_ocr}; missing speech: {missing_speech}"
        )
    return sorted(speech)


def parse_embedding(value) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(float, copy=False)
    if isinstance(value, (list, tuple)):
        return np.nan_to_num(np.asarray(value, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if "..." in text:
        return None
    text = text.strip("[]")
    arr = np.fromstring(text.replace(",", " "), sep=" ")
    if arr.size == 0:
        return None
    return np.nan_to_num(arr.astype(float, copy=False), nan=0.0, posinf=0.0, neginf=0.0)


def _timestamp_col(df: pd.DataFrame) -> str:
    for col in ("timestamp", "time stamp", "time_stamp", "start", "start_s"):
        if col in df.columns:
            return col
    raise KeyError(f"No timestamp-like column found. Columns: {list(df.columns)}")


def load_speech(features_dir: str | Path, videos: Iterable[str]) -> pd.DataFrame:
    rows = []
    features_path = Path(features_dir)
    for video in videos:
        df = pd.read_pickle(features_path / f"{video}_speech.pkl").copy()
        df["video"] = video
        ts_col = _timestamp_col(df)
        df["start_s"] = pd.to_numeric(df[ts_col], errors="coerce")
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
        df["end_s"] = df["start_s"] + df["duration"]
        df["transcript"] = df.get("transcript", "").fillna("").astype(str)
        embed_cols = [c for c in df.columns if "embed" in c.lower()]
        if embed_cols:
            emb_col = "text_embedding" if "text_embedding" in embed_cols else embed_cols[0]
            df["parsed_text_embedding"] = df[emb_col].apply(parse_embedding)
        else:
            df["parsed_text_embedding"] = None
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    speech = pd.concat(rows, ignore_index=True)
    speech = speech.dropna(subset=["start_s"]).sort_values(["video", "start_s"]).reset_index(drop=True)
    return speech


def _extract_video_from_frame(frame_value: str) -> str | None:
    match = re.search(r"Frames/([^/]+)/", str(frame_value))
    return match.group(1) if match else None


def _extract_frame_number(frame_value: str) -> int | None:
    match = re.search(r"frame_(\d+)\.", str(frame_value))
    return int(match.group(1)) if match else None


def load_ocr(features_dir: str | Path, videos: Iterable[str]) -> pd.DataFrame:
    frames = []
    features_path = Path(features_dir)
    for video in videos:
        df = pd.read_pickle(features_path / f"{video}_ocr.pkl").copy()
        df["video"] = video
        if "Frame" not in df.columns:
            raise KeyError(f"{video}_ocr.pkl has no Frame column")
        df["frame_video"] = df["Frame"].apply(_extract_video_from_frame)
        df["frame_number"] = df["Frame"].apply(_extract_frame_number)
        df["frame_number"] = pd.to_numeric(df["frame_number"], errors="coerce")
        frames.append(df)
    ocr_frames = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if ocr_frames.empty:
        return pd.DataFrame()

    rows = []
    for row in ocr_frames.itertuples(index=False):
        items = getattr(row, "OCR", None)
        if not isinstance(items, list):
            continue
        video = getattr(row, "video")
        frame_number = getattr(row, "frame_number")
        frame_path = getattr(row, "Frame")
        for item in items:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox", [np.nan, np.nan, np.nan, np.nan])
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                bbox = [np.nan, np.nan, np.nan, np.nan]
            rows.append(
                {
                    "video": video,
                    "frame_path": frame_path,
                    "frame_number": frame_number,
                    "time_s": float(frame_number) if pd.notna(frame_number) else np.nan,
                    "bbox": list(bbox),
                    "x1": float(bbox[0]) if pd.notna(bbox[0]) else np.nan,
                    "y1": float(bbox[1]) if pd.notna(bbox[1]) else np.nan,
                    "x2": float(bbox[2]) if pd.notna(bbox[2]) else np.nan,
                    "y2": float(bbox[3]) if pd.notna(bbox[3]) else np.nan,
                    "text": str(item.get("text", "")),
                    "conf": float(item.get("conf", np.nan)),
                }
            )
    ocr = pd.DataFrame(rows)
    if ocr.empty:
        return ocr
    ocr["clean_text"] = ocr["text"].apply(clean_ocr_text)
    ocr["text_len"] = ocr["clean_text"].str.len()
    ocr["x_center"] = (ocr["x1"] + ocr["x2"]) / 2
    ocr["y_center"] = (ocr["y1"] + ocr["y2"]) / 2
    ocr["box_area"] = (ocr["x2"] - ocr["x1"]).clip(lower=0) * (ocr["y2"] - ocr["y1"]).clip(lower=0)
    return ocr.dropna(subset=["time_s"]).sort_values(["video", "time_s"]).reset_index(drop=True)


def clean_ocr_text(text: str) -> str:
    text = TAG_RE.sub(" ", str(text))
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.upper()
    text = text.replace("Ç", "C")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -:;,.")


def is_boilerplate_ocr(text: str) -> bool:
    clean = clean_ocr_text(text)
    if len(clean) <= 8:
        return True
    if TIME_RE.search(clean) or DATE_RE.search(clean):
        return True
    if BOILERPLATE_RE.fullmatch(clean):
        return True
    if BOILERPLATE_RE.search(clean) and len(clean.split()) <= 3:
        return True
    digit_share = sum(ch.isdigit() for ch in clean) / max(len(clean), 1)
    return digit_share > 0.45


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _infer_position_columns(ocr: pd.DataFrame) -> pd.DataFrame:
    ocr = ocr.copy()
    geom = (
        ocr.groupby("video")
        .agg(max_x=("x2", "max"), max_y=("y2", "max"))
        .replace(0, np.nan)
    )
    ocr = ocr.join(geom, on="video")
    ocr["x_norm"] = ocr["x_center"] / ocr["max_x"]
    ocr["y_norm"] = ocr["y_center"] / ocr["max_y"]
    return ocr


def detect_headline_intervals(
    ocr: pd.DataFrame,
    conf_threshold: float = 0.60,
    similarity_threshold: float = 0.78,
    max_gap_s: float = 8.0,
    min_observations: int = 2,
    min_duration_s: float = 3.0,
) -> pd.DataFrame:
    if ocr.empty:
        return pd.DataFrame(columns=HEADLINE_COLUMNS)
    ocr_pos = _infer_position_columns(ocr)
    candidates = ocr_pos[
        (ocr_pos["conf"] >= conf_threshold)
        & (ocr_pos["text_len"] > 8)
        & (~ocr_pos["clean_text"].apply(is_boilerplate_ocr))
    ].copy()
    if candidates.empty:
        return pd.DataFrame(columns=HEADLINE_COLUMNS)

    # Favor lower-third text, but keep strong long headline text even if geometry is noisy.
    candidates["headline_region"] = candidates["y_norm"].between(0.38, 0.96) | candidates["y_norm"].isna()
    candidates = candidates[candidates["headline_region"] | (candidates["text_len"] >= 18)]
    candidates = candidates.sort_values(["video", "time_s", "conf", "text_len"], ascending=[True, True, False, False])

    # One headline candidate per second keeps repeated OCR boxes from fragmenting intervals.
    per_second = (
        candidates.assign(time_bin=candidates["time_s"].round().astype(int))
        .sort_values(["video", "time_bin", "headline_region", "conf", "text_len"], ascending=[True, True, False, False, False])
        .drop_duplicates(["video", "time_bin"], keep="first")
        .sort_values(["video", "time_s"])
    )

    intervals = []
    for video, group in per_second.groupby("video", sort=True):
        active = None
        for row in group.itertuples(index=False):
            text = row.clean_text
            time_s = float(row.time_s)
            if active is None:
                active = _new_headline_interval(video, row)
                continue
            gap = time_s - active["last_s"]
            sim = _text_similarity(active["headline"], text)
            if gap <= max_gap_s and sim >= similarity_threshold:
                active["last_s"] = time_s
                active["texts"].append(text)
                active["confs"].append(float(row.conf))
                active["xs"].append(float(row.x_center) if pd.notna(row.x_center) else np.nan)
                active["ys"].append(float(row.y_center) if pd.notna(row.y_center) else np.nan)
                active["headline"] = _representative_text(active["texts"])
            else:
                _append_interval(intervals, active, min_observations, min_duration_s)
                active = _new_headline_interval(video, row)
        if active is not None:
            _append_interval(intervals, active, min_observations, min_duration_s)

    result = pd.DataFrame(intervals)
    if result.empty:
        return pd.DataFrame(columns=HEADLINE_COLUMNS)
    return result.sort_values(["video", "start_s"]).reset_index(drop=True)


def _new_headline_interval(video: str, row) -> dict:
    x = float(row.x_center) if pd.notna(row.x_center) else np.nan
    y = float(row.y_center) if pd.notna(row.y_center) else np.nan
    return {
        "video": video,
        "headline": row.clean_text,
        "start_s": float(row.time_s),
        "last_s": float(row.time_s),
        "texts": [row.clean_text],
        "confs": [float(row.conf)],
        "xs": [x],
        "ys": [y],
    }


def _representative_text(texts: list[str]) -> str:
    counts = pd.Series(texts).value_counts()
    return str(counts.index[0])


def _append_interval(rows: list[dict], active: dict, min_observations: int, min_duration_s: float) -> None:
    duration = max(1.0, active["last_s"] - active["start_s"] + 1.0)
    if len(active["texts"]) < min_observations and duration < min_duration_s:
        return
    rows.append(
        {
            "video": active["video"],
            "headline": _representative_text(active["texts"]),
            "start_s": active["start_s"],
            "end_s": active["last_s"] + 1.0,
            "duration_s": duration,
            "observations": len(active["texts"]),
            "mean_conf": float(np.nanmean(active["confs"])),
            "mean_x": float(np.nanmean(active["xs"])),
            "mean_y": float(np.nanmean(active["ys"])),
        }
    )


def run_newscast_engine(
    config: SegmentationConfig | None = None,
    videos: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = config or SegmentationConfig()
    all_videos = list_newscast_videos(cfg.features_dir)
    if videos is None:
        videos = all_videos
    else:
        missing = [v for v in videos if v not in all_videos]
        if missing:
            raise ValueError(f"Unknown newscast videos: {missing}")
    speech = load_speech(cfg.features_dir, videos)
    ocr = load_ocr(cfg.features_dir, videos)
    headline_intervals = detect_headline_intervals(
        ocr,
        conf_threshold=cfg.ocr_conf_threshold,
        similarity_threshold=cfg.headline_similarity_threshold,
        max_gap_s=cfg.headline_max_gap_s,
        min_observations=cfg.headline_min_observations,
        min_duration_s=cfg.headline_min_duration_s,
    )

    boundary_frames = []
    item_frames = []
    item_embeddings = {}
    for video in videos:
        speech_v = speech[speech["video"] == video].copy()
        ocr_v = ocr[ocr["video"] == video].copy()
        headlines_v = headline_intervals[headline_intervals["video"] == video].copy()
        boundaries, component_series = score_video_boundaries(
            video,
            speech_v,
            ocr_v,
            headlines_v,
            cfg,
        )
        items, embeddings = build_news_items(
            video,
            speech_v,
            headlines_v,
            boundaries,
            component_series["video_end_s"],
            cfg,
        )
        boundary_frames.append(boundaries)
        item_frames.append(items)
        item_embeddings.update(embeddings)

    newscast_boundaries = pd.concat(boundary_frames, ignore_index=True) if boundary_frames else pd.DataFrame()
    newscast_items = pd.concat(item_frames, ignore_index=True) if item_frames else pd.DataFrame()
    newscast_items = merge_adjacent_similar_items(newscast_items)
    newscast_items = assign_topic_clusters(newscast_items, item_embeddings, cfg.max_topic_clusters)
    validate_outputs(newscast_items, newscast_boundaries, speech, ocr)
    return {
        "videos": pd.DataFrame({"video": videos}),
        "speech": speech,
        "ocr": ocr,
        "headline_intervals": headline_intervals,
        "newscast_boundaries": newscast_boundaries,
        "newscast_items": newscast_items,
    }


def score_video_boundaries(
    video: str,
    speech: pd.DataFrame,
    ocr: pd.DataFrame,
    headlines: pd.DataFrame,
    cfg: SegmentationConfig,
) -> tuple[pd.DataFrame, dict]:
    video_end = infer_video_end_s(speech, ocr, headlines)
    if video_end <= 1:
        return pd.DataFrame(), {"video_end_s": video_end}
    n = int(math.ceil(video_end)) + 1
    series = {
        "semantic_shift": np.zeros(n, dtype=float),
        "ocr_headline_shift": np.zeros(n, dtype=float),
        "ocr_layout_shift": np.zeros(n, dtype=float),
    }
    availability = {
        "semantic_shift": False,
        "ocr_headline_shift": False,
        "ocr_layout_shift": False,
    }

    semantic_points = compute_semantic_shift(speech, cfg.semantic_window_segments)
    _place_points(series["semantic_shift"], semantic_points)
    availability["semantic_shift"] = bool(len(semantic_points))

    headline_points = [
        (float(row.start_s), 1.0)
        for row in headlines.itertuples(index=False)
        if float(row.start_s) > cfg.min_item_duration_s
    ]
    _place_points(series["ocr_headline_shift"], headline_points)
    availability["ocr_headline_shift"] = bool(len(headline_points))

    layout_points = compute_ocr_layout_shift(ocr)
    _place_points(series["ocr_layout_shift"], layout_points)
    availability["ocr_layout_shift"] = bool(len(layout_points))

    for key in series:
        normalized = normalize_signal(series[key])
        series[key] = np.maximum(normalized, smooth_signal(normalized, radius=2))

    active_weights = {
        key: DEFAULT_WEIGHTS[key]
        for key, is_available in availability.items()
        if is_available and np.nanmax(series[key]) > 0
    }
    if not active_weights:
        active_weights = {"semantic_shift": 1.0}
    weight_total = sum(active_weights.values())
    total = np.zeros(n, dtype=float)
    for key, weight in active_weights.items():
        total += (weight / weight_total) * series[key]

    boundaries = select_boundaries(total, series, headlines, speech, video_end, cfg)
    rows = []
    for boundary_s in boundaries:
        idx = int(round(boundary_s))
        evidence = evidence_flags(series, idx)
        rows.append(
            {
                "video": video,
                "boundary_s": float(boundary_s),
                "score": float(window_max(total, idx)),
                "semantic_shift": float(window_max(series["semantic_shift"], idx)),
                "ocr_headline_shift": float(window_max(series["ocr_headline_shift"], idx)),
                "ocr_layout_shift": float(window_max(series["ocr_layout_shift"], idx)),
                "nearest_ocr_headline": nearest_headline(headlines, boundary_s),
                "boundary_confidence": confidence_tier(float(window_max(total, idx)), evidence),
                "evidence_flags": ",".join(evidence),
                "before_transcript": transcript_snippet(speech, boundary_s - 20, boundary_s),
                "after_transcript": transcript_snippet(speech, boundary_s, boundary_s + 20),
            }
        )
    boundary_df = pd.DataFrame(rows)
    if not boundary_df.empty:
        boundary_df = boundary_df.sort_values("boundary_s").reset_index(drop=True)
    return boundary_df, {
        "video_end_s": video_end,
        "total": total,
        **series,
    }


def infer_video_end_s(*frames: pd.DataFrame) -> float:
    candidates = []
    for df in frames:
        if df is None or df.empty:
            continue
        if {"end_s"}.issubset(df.columns):
            candidates.append(float(df["end_s"].max()))
        if {"start_s", "duration"}.issubset(df.columns):
            candidates.append(float((df["start_s"] + df["duration"]).max()))
        if "time_s" in df.columns:
            candidates.append(float(df["time_s"].max()))
        if "end_s" in df.columns:
            candidates.append(float(df["end_s"].max()))
    candidates = [c for c in candidates if pd.notna(c) and c > 0]
    return max(candidates) if candidates else 0.0


def compute_semantic_shift(speech: pd.DataFrame, window: int = 3) -> list[tuple[float, float]]:
    if speech.empty:
        return []
    embeddings = speech.get("parsed_text_embedding")
    if embeddings is not None and embeddings.apply(lambda x: isinstance(x, np.ndarray) and x.size > 0).any():
        return compute_embedding_shift(speech, "parsed_text_embedding", "start_s", window)
    return compute_tfidf_shift(speech, window)


def compute_embedding_shift(
    df: pd.DataFrame,
    embedding_col: str,
    time_col: str = "start_s",
    window: int = 3,
) -> list[tuple[float, float]]:
    if df.empty or embedding_col not in df.columns:
        return []
    valid = df[df[embedding_col].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)].copy()
    if len(valid) < window * 2 + 1:
        return []
    valid = valid.sort_values(time_col).reset_index(drop=True)
    dims = valid[embedding_col].apply(lambda x: x.shape[0])
    target_dim = int(dims.mode().iloc[0])
    valid = valid[valid[embedding_col].apply(lambda x: x.shape[0] == target_dim)].reset_index(drop=True)
    if len(valid) < window * 2 + 1:
        return []
    matrix = np.stack(valid[embedding_col].values)
    points = []
    for i in range(window, len(valid) - window):
        before = matrix[i - window:i].mean(axis=0)
        after = matrix[i:i + window].mean(axis=0)
        score = cosine_distance(before, after)
        points.append((float(valid.loc[i, time_col]), float(score)))
    return points


def compute_tfidf_shift(speech: pd.DataFrame, window: int = 3) -> list[tuple[float, float]]:
    if len(speech) < window * 2 + 1:
        return []
    texts = speech.sort_values("start_s")["transcript"].fillna("").astype(str).tolist()
    if not any(t.strip() for t in texts):
        return []
    vectorizer = TfidfVectorizer(min_df=1, max_features=4000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts).toarray()
    points = []
    ordered = speech.sort_values("start_s").reset_index(drop=True)
    for i in range(window, len(ordered) - window):
        before = matrix[i - window:i].mean(axis=0)
        after = matrix[i:i + window].mean(axis=0)
        points.append((float(ordered.loc[i, "start_s"]), cosine_distance(before, after)))
    return points


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


def compute_ocr_layout_shift(ocr: pd.DataFrame) -> list[tuple[float, float]]:
    if ocr.empty:
        return []
    ocr_pos = _infer_position_columns(ocr)
    per_second = (
        ocr_pos.assign(time_bin=ocr_pos["time_s"].round().astype(int))
        .groupby("time_bin")
        .agg(
            count=("clean_text", "size"),
            mean_x=("x_norm", "mean"),
            mean_y=("y_norm", "mean"),
            total_chars=("text_len", "sum"),
        )
        .sort_index()
        .fillna(0.0)
    )
    if len(per_second) < 2:
        return []
    values = per_second[["count", "mean_x", "mean_y", "total_chars"]].to_numpy(dtype=float)
    scale = np.nanstd(values, axis=0)
    scale[scale <= 1e-9] = 1.0
    diffs = np.linalg.norm(np.diff(values, axis=0) / scale, axis=1)
    times = per_second.index.to_numpy()[1:]
    return list(zip(times.astype(float), diffs.astype(float)))


def _place_points(signal: np.ndarray, points: list[tuple[float, float]]) -> None:
    if len(signal) == 0:
        return
    for time_s, value in points:
        if pd.isna(time_s) or pd.isna(value):
            continue
        idx = int(round(float(time_s)))
        if 0 <= idx < len(signal):
            signal[idx] = max(signal[idx], float(value))


def normalize_signal(values: np.ndarray) -> np.ndarray:
    out = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    positive = out[out > 0]
    if positive.size == 0:
        return out
    hi = np.quantile(positive, 0.95)
    if hi <= 1e-12:
        return out
    return np.clip(out / hi, 0.0, 1.0)


def smooth_signal(values: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0 or len(values) == 0:
        return values
    kernel = np.ones(radius * 2 + 1, dtype=float)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def select_boundaries(
    total: np.ndarray,
    series: dict[str, np.ndarray],
    headlines: pd.DataFrame,
    speech: pd.DataFrame,
    video_end_s: float,
    cfg: SegmentationConfig,
) -> list[float]:
    if len(total) == 0:
        return []

    margin = int(cfg.min_item_duration_s)
    all_peaks, all_props = find_peaks(
        total,
        distance=int(cfg.boundary_min_gap_s),
        prominence=0.10,
        wlen=int(min(120, len(total))),
    )

    keep = (all_peaks >= margin) & (all_peaks <= int(video_end_s) - margin)
    peak_indices = all_peaks[keep]
    prominences = all_props["prominences"][keep]
    if len(peak_indices) == 0:
        return []

    headline_boost = np.array([
        1.5 if series["ocr_headline_shift"][idx] >= 0.4 else 1.0
        for idx in peak_indices
    ])
    scores = prominences * headline_boost

    ranked = sorted(zip(peak_indices, scores), key=lambda x: x[1], reverse=True)
    selected: list[float] = []
    for idx, _ in ranked:
        snapped = snap_boundary(float(idx), speech, headlines, cfg.snap_window_s)
        if snapped < cfg.min_item_duration_s or snapped > video_end_s - cfg.min_item_duration_s:
            continue
        if all(abs(snapped - existing) >= cfg.boundary_min_gap_s for existing in selected):
            selected.append(snapped)
    return sorted(selected)


def snap_boundary(time_s: float, speech: pd.DataFrame, headlines: pd.DataFrame, window_s: float) -> float:
    choices = []
    if not speech.empty:
        choices.extend(speech["start_s"].dropna().astype(float).tolist())
    if not headlines.empty:
        choices.extend(headlines["start_s"].dropna().astype(float).tolist())
    nearby = [c for c in choices if abs(c - time_s) <= window_s]
    if not nearby:
        return float(time_s)
    return float(min(nearby, key=lambda c: abs(c - time_s)))


def evidence_flags(series: dict[str, np.ndarray], idx: int, threshold: float = 0.20) -> list[str]:
    flags = []
    for key in ("semantic_shift", "ocr_headline_shift", "ocr_layout_shift"):
        values = series.get(key)
        if values is None or len(values) == 0:
            continue
        lo = max(0, idx - 2)
        hi = min(len(values), idx + 3)
        if np.nanmax(values[lo:hi]) >= threshold:
            flags.append(key.replace("_shift", ""))
    return flags


def window_max(values: np.ndarray, idx: int, radius: int = 2) -> float:
    if values is None or len(values) == 0:
        return 0.0
    lo = max(0, idx - radius)
    hi = min(len(values), idx + radius + 1)
    return float(np.nanmax(values[lo:hi]))


def confidence_tier(score: float, evidence: list[str]) -> str:
    if score >= 0.55 and len(evidence) >= 3:
        return "high"
    if score >= 0.35 and len(evidence) >= 2:
        return "medium"
    if "ocr_headline" in evidence and len(evidence) >= 2:
        return "medium"
    return "low"


def nearest_headline(headlines: pd.DataFrame, time_s: float) -> str:
    if headlines.empty:
        return ""
    df = headlines.copy()
    df["distance"] = np.minimum((df["start_s"] - time_s).abs(), (df["end_s"] - time_s).abs())
    row = df.sort_values("distance").iloc[0]
    return str(row["headline"])


def transcript_snippet(speech: pd.DataFrame, start_s: float, end_s: float, max_chars: int = 260) -> str:
    if speech.empty:
        return ""
    mask = (speech["end_s"] >= start_s) & (speech["start_s"] <= end_s)
    text = " ".join(speech.loc[mask, "transcript"].fillna("").astype(str).tolist())
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def build_news_items(
    video: str,
    speech: pd.DataFrame,
    headlines: pd.DataFrame,
    boundaries: pd.DataFrame,
    video_end_s: float,
    cfg: SegmentationConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    boundary_times = boundaries["boundary_s"].tolist() if not boundaries.empty else []
    points = [0.0] + [float(t) for t in boundary_times] + [float(video_end_s)]
    points = sorted(set(round(p, 3) for p in points))
    points = merge_short_points(points, headlines, cfg.min_item_duration_s)

    rows = []
    embeddings = {}
    boundary_lookup = (
        boundaries.set_index(boundaries["boundary_s"].round(3)).to_dict("index")
        if not boundaries.empty
        else {}
    )
    for i in range(len(points) - 1):
        start_s = float(points[i])
        end_s = float(points[i + 1])
        if end_s <= start_s:
            continue
        item_id = i + 1
        item_key = f"{video}::{item_id}"
        text = transcript_snippet(speech, start_s, end_s, max_chars=1200)
        headline = headline_for_interval(headlines, start_s, end_s) or fallback_headline(text)
        start_boundary = boundary_lookup.get(round(start_s, 3), {})
        evidence = start_boundary.get("evidence_flags", "program_start" if i == 0 else "")
        confidence = start_boundary.get("boundary_confidence", "program_start" if i == 0 else "low")
        rows.append(
            {
                "video": video,
                "item_id": item_id,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                "headline": headline,
                "topic_cluster": np.nan,
                "boundary_confidence": confidence,
                "evidence_flags": evidence,
                "transcript_excerpt": text[:500],
                "item_text": text,
                "source_item_keys": item_key,
            }
        )
        emb = item_embedding_for_interval(speech, start_s, end_s)
        if emb is not None:
            embeddings[item_key] = emb
    return pd.DataFrame(rows), embeddings


def merge_short_points(points: list[float], headlines: pd.DataFrame, min_duration_s: float) -> list[float]:
    if len(points) <= 2:
        return points
    result = [points[0]]
    for i in range(1, len(points) - 1):
        prev = result[-1]
        cur = points[i]
        nxt = points[i + 1]
        has_headline = not headlines[
            (headlines["start_s"] >= cur - 2) & (headlines["start_s"] <= cur + 2)
        ].empty if not headlines.empty else False
        if cur - prev < min_duration_s and not has_headline:
            continue
        if nxt - cur < min_duration_s and not has_headline:
            continue
        result.append(cur)
    result.append(points[-1])
    return result


def headline_for_interval(headlines: pd.DataFrame, start_s: float, end_s: float) -> str:
    if headlines.empty:
        return ""
    df = headlines.copy()
    df["overlap_s"] = np.maximum(0.0, np.minimum(df["end_s"], end_s) - np.maximum(df["start_s"], start_s))
    df = df[df["overlap_s"] > 0]
    if df.empty:
        df = headlines.copy()
        df["distance"] = np.minimum((df["start_s"] - start_s).abs(), (df["end_s"] - end_s).abs())
        row = df.sort_values("distance").iloc[0]
    else:
        row = df.sort_values(["overlap_s", "duration_s", "mean_conf"], ascending=False).iloc[0]
    return str(row["headline"])


def fallback_headline(text: str, max_terms: int = 5) -> str:
    tokens = []
    for match in WORD_RE.finditer(str(text).lower()):
        token = normalize_token(match.group(0))
        if len(token) >= 4 and token not in PT_STOPWORDS:
            tokens.append(token.upper())
    if not tokens:
        return "NO OCR HEADLINE"
    counts = pd.Series(tokens).value_counts().head(max_terms)
    return " / ".join(counts.index.tolist())


def normalize_token(token: str) -> str:
    replacements = str.maketrans("ÁÀÂÃÉÊÍÓÔÕÚÇáàâãéêíóôõúç", "AAAAEEIOOOUCaaaaeeiooouc")
    return token.translate(replacements).lower()


def item_embedding_for_interval(speech: pd.DataFrame, start_s: float, end_s: float) -> np.ndarray | None:
    if speech.empty or "parsed_text_embedding" not in speech.columns:
        return None
    mask = (
        (speech["end_s"] >= start_s)
        & (speech["start_s"] <= end_s)
        & speech["parsed_text_embedding"].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)
    )
    values = speech.loc[mask, "parsed_text_embedding"].tolist()
    if not values:
        return None
    dims = pd.Series([v.shape[0] for v in values])
    target_dim = int(dims.mode().iloc[0])
    values = [v for v in values if v.shape[0] == target_dim]
    if not values:
        return None
    return np.nan_to_num(np.stack(values).mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)


def merge_adjacent_similar_items(
    items: pd.DataFrame,
    similarity_threshold: float = 0.88,
) -> pd.DataFrame:
    if items.empty:
        return items
    merged_rows = []
    for _, group in items.sort_values(["video", "start_s"]).groupby("video", sort=False):
        active = None
        for row in group.to_dict("records"):
            if active is None:
                active = row.copy()
                continue
            sim = _text_similarity(str(active.get("headline", "")), str(row.get("headline", "")))
            same_headline = (
                active.get("headline")
                and row.get("headline")
                and active.get("headline") == row.get("headline")
                and active.get("headline") != "NO OCR HEADLINE"
            )
            if same_headline or sim >= similarity_threshold:
                active["end_s"] = row["end_s"]
                active["duration_s"] = active["end_s"] - active["start_s"]
                active["transcript_excerpt"] = (
                    str(active.get("transcript_excerpt", "")) + " " + str(row.get("transcript_excerpt", ""))
                ).strip()[:500]
                active["item_text"] = (
                    str(active.get("item_text", "")) + " " + str(row.get("item_text", ""))
                ).strip()
                active["source_item_keys"] = "|".join(
                    key
                    for key in [
                        str(active.get("source_item_keys", "")).strip(),
                        str(row.get("source_item_keys", "")).strip(),
                    ]
                    if key
                )
                active["evidence_flags"] = merge_flag_strings(active.get("evidence_flags", ""), row.get("evidence_flags", ""))
                active["boundary_confidence"] = stronger_confidence(
                    active.get("boundary_confidence", "low"),
                    row.get("boundary_confidence", "low"),
                )
            else:
                merged_rows.append(active)
                active = row.copy()
        if active is not None:
            merged_rows.append(active)
    merged = pd.DataFrame(merged_rows)
    if merged.empty:
        return merged
    merged["item_id"] = merged.groupby("video").cumcount() + 1
    return merged.reset_index(drop=True)


def merge_flag_strings(a: str, b: str) -> str:
    flags = []
    for part in f"{a},{b}".split(","):
        part = part.strip()
        if part and part not in flags:
            flags.append(part)
    return ",".join(flags)


def stronger_confidence(a: str, b: str) -> str:
    order = {"program_start": 3, "high": 2, "medium": 1, "low": 0, "": 0}
    return a if order.get(str(a), 0) >= order.get(str(b), 0) else b


def assign_topic_clusters(
    items: pd.DataFrame,
    item_embeddings: dict[str, np.ndarray],
    max_clusters: int = 8,
) -> pd.DataFrame:
    if items.empty:
        return items
    items = items.copy()
    grouped_keys = []
    if "source_item_keys" in items.columns:
        grouped_keys = [
            [key for key in str(keys).split("|") if key]
            for keys in items["source_item_keys"].fillna("")
        ]
    else:
        grouped_keys = [[f"{row.video}::{row.item_id}"] for row in items.itertuples(index=False)]
    if item_embeddings and all(keys and all(key in item_embeddings for key in keys) for keys in grouped_keys):
        matrix = np.stack([
            np.stack([item_embeddings[key] for key in keys]).mean(axis=0)
            for keys in grouped_keys
        ])
        matrix = safe_feature_matrix(matrix)
    else:
        texts = items["item_text"].fillna("").astype(str).tolist()
        if len(items) < 2 or not any(t.strip() for t in texts):
            items["topic_cluster"] = 0
            return items.drop(columns=["item_text", "source_item_keys"], errors="ignore")
        matrix = safe_feature_matrix(
            TfidfVectorizer(max_features=5000, min_df=1, ngram_range=(1, 2)).fit_transform(texts).toarray()
        )

    row_norms = np.linalg.norm(matrix, axis=1)
    usable = row_norms > 1e-9
    labels = np.full(len(items), -1, dtype=int)
    if usable.sum() == 0:
        labels[:] = 0
    elif usable.sum() == 1:
        labels[usable] = 0
    else:
        k = min(max_clusters, max(2, int(round(math.sqrt(int(usable.sum()))))), int(usable.sum()))
        labels[usable] = KMeans(n_clusters=k, random_state=42, n_init=10, init="random").fit_predict(matrix[usable])
    items["topic_cluster"] = labels.astype(int)
    return items.drop(columns=["item_text", "source_item_keys"], errors="ignore")


def safe_feature_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.clip(matrix, -1e6, 1e6)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    return matrix / norms


def validate_outputs(
    items: pd.DataFrame,
    boundaries: pd.DataFrame,
    speech: pd.DataFrame,
    ocr: pd.DataFrame,
) -> None:
    if not boundaries.empty:
        for video, group in boundaries.groupby("video"):
            times = group["boundary_s"].to_numpy(dtype=float)
            if not np.all(np.diff(times) >= 0):
                raise AssertionError(f"Boundaries are not sorted for {video}")
            video_end = infer_video_end_s(speech[speech["video"] == video], ocr[ocr["video"] == video])
            if (times < 0).any() or (times > video_end + 1).any():
                raise AssertionError(f"Boundary outside video duration for {video}")
    if not items.empty:
        bad = items[items["duration_s"] <= 0]
        if not bad.empty:
            raise AssertionError(f"Found non-positive item durations: {bad[['video', 'item_id']].to_dict('records')}")


def summarize_results(items: pd.DataFrame, headline_intervals: pd.DataFrame) -> pd.DataFrame:
    if items.empty:
        return pd.DataFrame()
    summary = (
        items.groupby("video")
        .agg(
            news_items=("item_id", "count"),
            segmented_minutes=("duration_s", lambda s: s.sum() / 60.0),
            median_item_s=("duration_s", "median"),
            high_confidence_items=("boundary_confidence", lambda s: (s == "high").sum()),
            medium_confidence_items=("boundary_confidence", lambda s: (s == "medium").sum()),
            low_confidence_items=("boundary_confidence", lambda s: (s == "low").sum()),
            items_with_ocr_headline=("headline", lambda s: (s != "NO OCR HEADLINE").sum()),
        )
        .reset_index()
    )
    if not headline_intervals.empty:
        ocr_counts = headline_intervals.groupby("video").size().rename("headline_intervals").reset_index()
        summary = summary.merge(ocr_counts, on="video", how="left")
    else:
        summary["headline_intervals"] = 0
    summary["headline_intervals"] = summary["headline_intervals"].fillna(0).astype(int)
    summary["weak_ocr_agreement"] = summary["items_with_ocr_headline"] < (summary["news_items"] * 0.5)
    return summary.sort_values("video").reset_index(drop=True)
