"""
Melodix - Music Recommender Backend
Flask REST API serving SVD, KNN, Content-Based and Hybrid recommendations
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
import warnings
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from auth import auth_bp, init_db, get_user_from_request, get_db

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API from any origin
app.register_blueprint(auth_bp)

# ─────────────────────────────────────────────
# LOAD ALL ARTEFACTS ON STARTUP
# ─────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))
ARTEFACTS = os.path.join(BASE, "artefacts")

print("Loading model artefacts...")

svd_model     = joblib.load(os.path.join(ARTEFACTS, "svd_model.pkl"))
scaler        = joblib.load(os.path.join(ARTEFACTS, "scaler.pkl"))
user_factors  = np.load(os.path.join(ARTEFACTS, "user_factors.npy"))
item_factors  = np.load(os.path.join(ARTEFACTS, "item_factors.npy"))
feature_array = np.load(os.path.join(ARTEFACTS, "feature_array.npy"))
# SVD scores are computed on-the-fly per request (item_factors @ user_vec) —
# a full precomputed (users x tracks) matrix isn't worth ~90MB in git for a
# dot product that takes <1ms.
train_matrix  = sp.load_npz(os.path.join(ARTEFACTS, "train_matrix.npz"))
tracks_df     = pd.read_csv(os.path.join(ARTEFACTS, "tracks.csv"))

# Clean genre column
tracks_df["genre_clean"] = (
    tracks_df["genre"]
    .str.replace(r"['\[\]]", "", regex=True)
    .str.strip()
    .apply(lambda x: x.split(",")[0].strip())
)

# Pre-fit KNN model on user_factors for user-based CF
print("Fitting KNN model...")
knn_model = NearestNeighbors(n_neighbors=20, metric="cosine", algorithm="brute")
knn_model.fit(user_factors)

# Normalise feature_array once for content-based
print("Normalising feature array for content similarity...")
feature_norm = feature_array / (
    np.linalg.norm(feature_array, axis=1, keepdims=True) + 1e-9
)

init_db(n_train_users=user_factors.shape[0])
track_id_to_idx = {tid: i for i, tid in enumerate(tracks_df["track_id"])}


def get_real_likes_boost(user_row):
    """
    For a logged-in account: build a content-based score vector from their
    actual liked tracks (server-stored), so recs improve as they use the
    app for real — independent of the frozen synthetic SVD profile.
    Returns None if they haven't liked anything yet (falls back to pure
    cold-start svd_profile_id behaviour).
    """
    conn = get_db()
    rows = conn.execute(
        "SELECT track_id FROM liked_tracks WHERE user_id = ?", (user_row["id"],)
    ).fetchall()
    conn.close()
    liked_idx = [track_id_to_idx[r["track_id"]] for r in rows if r["track_id"] in track_id_to_idx]
    if not liked_idx:
        return None
    return cosine_similarity(feature_norm[liked_idx], feature_norm).mean(axis=0)

NUM_USERS  = user_factors.shape[0]   # 800
NUM_TRACKS = item_factors.shape[0]   # 222

print(f"Ready — {NUM_USERS} users, {NUM_TRACKS} tracks, {feature_array.shape[1]} features")


# ─────────────────────────────────────────────
# HELPER: build a track response dict
# ─────────────────────────────────────────────

def track_to_dict(row, score=None, rank=None):
    artist = row["artist"]
    track_id = row["track_id"]
    # Real Spotify track IDs now (song-level data) -> real, playable links
    sp_url = f"https://open.spotify.com/track/{track_id}"
    sp_embed_url = f"https://open.spotify.com/embed/track/{track_id}"
    yt_url = f"https://www.youtube.com/results?search_query={artist.replace(' ', '+').replace('&', '%26')}+{row['track_name'].replace(' ', '+')}"
    d = {
        "track_id":    track_id,
        "track_name":  row["track_name"],
        "artist":      artist,
        "genre":       row["genre_clean"],
        "danceability":round(float(row["danceability"]), 4),
        "energy":      round(float(row["energy"]),       4),
        "valence":     round(float(row["valence"]),      4),
        "acousticness":round(float(row["acousticness"]), 4),
        "instrumentalness": round(float(row["instrumentalness"]), 4),
        "tempo":       round(float(row["tempo"]),        2),
        "popularity":  round(float(row["popularity"]),   1),
        "spotify_url": sp_url,
        "spotify_embed_url": sp_embed_url,
        "youtube_url": yt_url,
    }
    if score is not None:
        d["score"] = round(float(score), 6)
    if rank is not None:
        d["rank"] = rank
    return d


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Melodix Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "GET /recommend": "Get top-K recommendations for a user",
            "GET /similar":   "Get content-similar tracks to a given track",
            "GET /tracks":    "List all tracks",
            "GET /track/<id>":"Get a single track by ID",
            "GET /mood":      "Get tracks matching a mood",
            "GET /search":    "Search tracks by artist or genre",
            "GET /trending":  "Get top tracks by popularity",
            "GET /stats":     "Dataset and model statistics",
        }
    })


# ── 1. RECOMMENDATIONS (SVD / KNN / Content / Hybrid) ──────────────────────

@app.route("/recommend", methods=["GET"])
def recommend():
    """
    GET /recommend?user_id=42&model=svd&k=10
    Authorization: Bearer <token>   (optional — overrides user_id with the
                                      logged-in account's cold-start profile,
                                      and blends in their real liked-track
                                      signal once they have any)

    user_id : int  0–1199  (required unless Authorization header is sent)
    model   : str  svd | knn | content | hybrid  (default: svd)
    k       : int  1–50   (default: 10)
    """
    account = get_user_from_request()
    real_boost = None

    if account is not None:
        user_id = account["svd_profile_id"]
        real_boost = get_real_likes_boost(account)
    else:
        try:
            user_id = int(request.args.get("user_id", -1))
        except ValueError:
            return jsonify({"error": "user_id must be an integer"}), 400
        if user_id < 0 or user_id >= NUM_USERS:
            return jsonify({"error": f"user_id must be between 0 and {NUM_USERS - 1}"}), 400

    model = request.args.get("model", "svd").lower()
    if model not in ("svd", "knn", "content", "hybrid"):
        return jsonify({"error": "model must be svd | knn | content | hybrid"}), 400

    try:
        k = int(request.args.get("k", 10))
        k = max(1, min(k, 50))
    except ValueError:
        return jsonify({"error": "k must be an integer"}), 400

    # ── already-seen tracks (exclude from recommendations) ──
    seen = set(train_matrix[user_id].nonzero()[1])

    # ── score computation ──
    if model == "svd":
        scores = _svd_scores(user_id)

    elif model == "knn":
        scores = _knn_scores(user_id)

    elif model == "content":
        scores = _content_scores(user_id)

    else:  # hybrid
        s_svd     = _svd_scores(user_id)
        s_content = _content_scores(user_id)
        alpha = 0.3   # 30% collaborative, 70% content — tuned via P@10/NDCG@10 sweep
        # normalise each to [0,1] before blending
        s_svd     = _minmax(s_svd)
        s_content = _minmax(s_content)
        scores    = alpha * s_svd + (1 - alpha) * s_content

    # ── blend in real liked-track signal for logged-in users ──
    personalized = False
    if real_boost is not None:
        scores = 0.65 * _minmax(scores) + 0.35 * _minmax(real_boost)
        personalized = True

    # ── filter seen, pick top-k ──
    unseen_idx = [i for i in range(NUM_TRACKS) if i not in seen]
    unseen_scores = [(i, scores[i]) for i in unseen_idx]
    unseen_scores.sort(key=lambda x: x[1], reverse=True)
    top_k = unseen_scores[:k]

    results = []
    for rank, (idx, score) in enumerate(top_k, 1):
        row = tracks_df.iloc[idx]
        results.append(track_to_dict(row, score=score, rank=rank))

    return jsonify({
        "user_id": user_id,
        "model":   model,
        "k":       k,
        "personalized": personalized,
        "recommendations": results
    })


# ── 2. CONTENT SIMILARITY ───────────────────────────────────────────────────

@app.route("/similar", methods=["GET"])
def similar():
    """
    GET /similar?track_id=T00154&k=10

    Returns k tracks most similar to the given track by audio features.
    """
    track_id = request.args.get("track_id", "")
    if not track_id:
        return jsonify({"error": "track_id is required"}), 400

    matches = tracks_df[tracks_df["track_id"] == track_id]
    if matches.empty:
        return jsonify({"error": f"track_id '{track_id}' not found"}), 404

    try:
        k = int(request.args.get("k", 10))
        k = max(1, min(k, 50))
    except ValueError:
        return jsonify({"error": "k must be an integer"}), 400

    idx = matches.index[0]
    query_vec = feature_norm[idx].reshape(1, -1)
    sims = cosine_similarity(query_vec, feature_norm)[0]
    sims[idx] = -1  # exclude the query track itself

    top_k_idx = np.argsort(sims)[::-1][:k]

    results = []
    for rank, i in enumerate(top_k_idx, 1):
        row = tracks_df.iloc[i]
        results.append(track_to_dict(row, score=sims[i], rank=rank))

    return jsonify({
        "query_track": track_to_dict(tracks_df.iloc[idx]),
        "model":       "content",
        "k":           k,
        "similar_tracks": results
    })


# ── 3. ALL TRACKS ───────────────────────────────────────────────────────────

@app.route("/tracks", methods=["GET"])
def all_tracks():
    """
    GET /tracks?sort=popularity&order=desc&genre=pop&limit=50

    sort  : popularity | danceability | energy | valence | acousticness | tempo
    order : asc | desc
    genre : filter by genre (partial match)
    limit : 1–222
    """
    sort_by = request.args.get("sort", "popularity")
    order   = request.args.get("order", "desc")
    genre   = request.args.get("genre", "")
    try:
        limit = int(request.args.get("limit", 222))
        limit = max(1, min(limit, 222))
    except ValueError:
        limit = 222

    valid_sorts = {"popularity", "danceability", "energy", "valence",
                   "acousticness", "instrumentalness", "tempo", "liveness"}
    if sort_by not in valid_sorts:
        sort_by = "popularity"

    df = tracks_df.copy()
    if genre:
        df = df[df["genre_clean"].str.lower().str.contains(genre.lower(), na=False)]

    df = df.sort_values(sort_by, ascending=(order == "asc")).head(limit)

    return jsonify({
        "total":  len(df),
        "sort":   sort_by,
        "order":  order,
        "tracks": [track_to_dict(row) for _, row in df.iterrows()]
    })


# ── 4. SINGLE TRACK ─────────────────────────────────────────────────────────

@app.route("/track/<track_id>", methods=["GET"])
def get_track(track_id):
    matches = tracks_df[tracks_df["track_id"] == track_id]
    if matches.empty:
        return jsonify({"error": f"track_id '{track_id}' not found"}), 404
    row = matches.iloc[0]
    return jsonify(track_to_dict(row))


# ── 5. MOOD RADIO ───────────────────────────────────────────────────────────

MOOD_CONFIG = {
    "happy":     {"valence":     (0.55, 1.0), "energy":      (0.45, 1.0)},
    "chill":     {"energy":      (0.0,  0.55), "acousticness":(0.3,  1.0)},
    "energetic": {"energy":      (0.65, 1.0), "tempo":       (110,  220)},
    "sad":       {"valence":     (0.0,  0.40), "energy":      (0.0,  0.55)},
    "focus":     {"instrumentalness": (0.2, 1.0)},
    "romantic":  {"valence":     (0.35, 0.80), "acousticness":(0.2, 1.0),
                  "energy":      (0.0,  0.65)},
    "hype":      {"danceability":(0.65, 1.0), "energy":      (0.62, 1.0)},
    "sleep":     {"energy":      (0.0,  0.35), "acousticness":(0.5, 1.0)},
}

@app.route("/mood", methods=["GET"])
def mood():
    """
    GET /mood?mood=happy&k=15

    mood : happy | chill | energetic | sad | focus | romantic | hype | sleep
    k    : number of results (default 15)
    """
    mood_name = request.args.get("mood", "").lower()
    if mood_name not in MOOD_CONFIG:
        return jsonify({
            "error": f"mood must be one of: {', '.join(MOOD_CONFIG.keys())}"
        }), 400

    try:
        k = int(request.args.get("k", 15))
        k = max(1, min(k, 50))
    except ValueError:
        k = 15

    cfg = MOOD_CONFIG[mood_name]
    df  = tracks_df.copy()

    for feature, (lo, hi) in cfg.items():
        if feature in df.columns:
            df = df[(df[feature] >= lo) & (df[feature] <= hi)]

    # fallback: if fewer than 5 matches, relax and return by popularity
    if len(df) < 5:
        df = tracks_df.sort_values("popularity", ascending=False).head(k)

    df = df.sort_values("popularity", ascending=False).head(k)

    return jsonify({
        "mood":   mood_name,
        "k":      len(df),
        "tracks": [track_to_dict(row) for _, row in df.iterrows()]
    })


# ── 6. SEARCH ────────────────────────────────────────────────────────────────

@app.route("/search", methods=["GET"])
def search():
    """
    GET /search?q=Taylor&limit=10

    q     : search query (artist name or genre)
    limit : max results (default 10)
    """
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "q parameter is required"}), 400

    try:
        limit = int(request.args.get("limit", 10))
        limit = max(1, min(limit, 50))
    except ValueError:
        limit = 10

    q_lower = q.lower()
    df = tracks_df[
        tracks_df["artist"].str.lower().str.contains(q_lower, na=False) |
        tracks_df["genre_clean"].str.lower().str.contains(q_lower, na=False) |
        tracks_df["track_name"].str.lower().str.contains(q_lower, na=False)
    ].head(limit)

    return jsonify({
        "query":   q,
        "total":   len(df),
        "results": [track_to_dict(row) for _, row in df.iterrows()]
    })


# ── 7. TRENDING ──────────────────────────────────────────────────────────────

@app.route("/trending", methods=["GET"])
def trending():
    """
    GET /trending?k=20

    Returns top-k tracks by popularity score.
    """
    try:
        k = int(request.args.get("k", 20))
        k = max(1, min(k, len(tracks_df)))
    except ValueError:
        k = 20

    df = tracks_df.sort_values("popularity", ascending=False).head(k)
    return jsonify({
        "k":      k,
        "tracks": [track_to_dict(row) for _, row in df.iterrows()]
    })


# ── 8. STATS ─────────────────────────────────────────────────────────────────

@app.route("/stats", methods=["GET"])
def stats():
    genres    = tracks_df["genre_clean"].value_counts().head(10).to_dict()
    avg_feats = {
        "danceability":     round(float(tracks_df["danceability"].mean()),     3),
        "energy":           round(float(tracks_df["energy"].mean()),           3),
        "valence":          round(float(tracks_df["valence"].mean()),          3),
        "acousticness":     round(float(tracks_df["acousticness"].mean()),     3),
        "instrumentalness": round(float(tracks_df["instrumentalness"].mean()), 3),
        "tempo":            round(float(tracks_df["tempo"].mean()),            2),
        "popularity":       round(float(tracks_df["popularity"].mean()),       2),
    }
    density = float(train_matrix.nnz) / (train_matrix.shape[0] * train_matrix.shape[1])
    return jsonify({
        "dataset": {
            "num_tracks":  int(NUM_TRACKS),
            "num_users":   int(NUM_USERS),
            "num_genres":  int(tracks_df["genre_clean"].nunique()),
            "matrix_density_pct": round(density * 100, 4),
            "top_genres":  genres,
        },
        "models": {
            "svd_factors":       int(user_factors.shape[1]),
            "feature_dimensions":int(feature_array.shape[1]),
            "available_models":  ["svd", "knn", "content", "hybrid"],
        },
        "average_audio_features": avg_feats,
    })


# ─────────────────────────────────────────────
# PRIVATE SCORING HELPERS
# ─────────────────────────────────────────────

def _svd_scores(user_id: int) -> np.ndarray:
    """Return SVD-predicted scores for all tracks for a given user (on-the-fly dot product)."""
    return item_factors @ user_factors[user_id]


def _knn_scores(user_id: int) -> np.ndarray:
    """User-based KNN collaborative filtering."""
    user_vec = user_factors[user_id].reshape(1, -1)
    distances, indices = knn_model.kneighbors(user_vec, n_neighbors=21)
    # exclude the user themselves (distance ~ 0)
    neighbor_ids = [i for i in indices[0] if i != user_id][:20]
    neighbor_rows = train_matrix[neighbor_ids].toarray()  # (20, 222)
    # weighted average by (1 - distance) similarity weight
    sims = 1 - distances[0][1:21]
    scores = sims @ neighbor_rows / (sims.sum() + 1e-9)
    return scores


def _content_scores(user_id: int) -> np.ndarray:
    """Content-based: average similarity to user's interacted tracks."""
    interacted = train_matrix[user_id].nonzero()[1]
    if len(interacted) == 0:
        # cold-start: return popularity as proxy
        return tracks_df["popularity"].values / 100.0

    interacted_feats = feature_norm[interacted]           # (n_seen, 64)
    mean_profile     = interacted_feats.mean(axis=0)      # (64,)
    scores           = feature_norm @ mean_profile         # (222,)
    return scores


def _minmax(arr: np.ndarray) -> np.ndarray:
    """Normalise array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    print(f"\nMelodix API running on http://localhost:{port}")
    print("Endpoints: /recommend  /similar  /tracks  /mood  /search  /trending  /stats\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
