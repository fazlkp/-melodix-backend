"""
Melodix — Accounts & Auth
SQLite (stored on the same persistent disk as the model artefacts) +
JWT sessions + bcrypt password hashing. Kept as a separate module so
app.py's recommendation logic stays untouched.

Cold-start strategy: every account is assigned a stable svd_profile_id
(0 - N_TRAIN_USERS-1) at signup, borrowed from the synthetic listener
profiles the SVD model was trained on. This is what lets a brand-new
account get non-random recommendations from second one, before they've
liked anything. As they like tracks, /recommend can optionally blend in
a content-based boost from their real likes (see get_boosted_scores in
app.py) — full personalized retraining is a natural v4 step.
"""
import os
import sqlite3
import time
import bcrypt
import jwt
from functools import wraps
from flask import Blueprint, request, jsonify, g

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "artefacts", "melodix.db")
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
TOKEN_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days

auth_bp = Blueprint("auth", __name__)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(n_train_users: int):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            svd_profile_id INTEGER NOT NULL,
            created_at INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS liked_tracks (
            user_id INTEGER NOT NULL,
            track_id TEXT NOT NULL,
            liked_at INTEGER NOT NULL,
            PRIMARY KEY (user_id, track_id)
        )
    """)
    conn.commit()
    conn.close()
    init_db.n_train_users = n_train_users


def make_token(user_id: int) -> str:
    payload = {"user_id": user_id, "exp": int(time.time()) + TOKEN_TTL_SECONDS}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["user_id"]
    except Exception:
        return None


def get_user_from_request():
    """Returns the user row for a valid Authorization: Bearer <token> header, else None."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ", 1)[1]
    user_id = decode_token(token)
    if user_id is None:
        return None
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return row


def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = get_user_from_request()
        if user is None:
            return jsonify({"error": "Authentication required"}), 401
        g.user = user
        return f(*args, **kwargs)
    return wrapper


def user_public(user_row):
    return {
        "id": user_row["id"],
        "username": user_row["username"],
        "email": user_row["email"],
        "svd_profile_id": user_row["svd_profile_id"],
    }


# ── ROUTES ──────────────────────────────────────────────────────────────

@auth_bp.route("/auth/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters"}), 400
    if "@" not in email:
        return jsonify({"error": "Enter a valid email"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    svd_profile_id = int.from_bytes(os.urandom(2), "big") % init_db.n_train_users

    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash, svd_profile_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (username, email, pw_hash, svd_profile_id, int(time.time())),
        )
        conn.commit()
        user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "Username or email already taken"}), 409
    conn.close()

    token = make_token(user_id)
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return jsonify({"token": token, "user": user_public(row)}), 201


@auth_bp.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()

    if row is None or not bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
        return jsonify({"error": "Invalid email or password"}), 401

    token = make_token(row["id"])
    return jsonify({"token": token, "user": user_public(row)})


@auth_bp.route("/auth/me", methods=["GET"])
@require_auth
def me():
    return jsonify({"user": user_public(g.user)})


@auth_bp.route("/likes", methods=["GET"])
@require_auth
def get_likes():
    conn = get_db()
    rows = conn.execute(
        "SELECT track_id FROM liked_tracks WHERE user_id = ? ORDER BY liked_at DESC",
        (g.user["id"],),
    ).fetchall()
    conn.close()
    return jsonify({"track_ids": [r["track_id"] for r in rows]})


@auth_bp.route("/likes/<track_id>", methods=["POST"])
@require_auth
def add_like(track_id):
    conn = get_db()
    conn.execute(
        "INSERT OR IGNORE INTO liked_tracks (user_id, track_id, liked_at) VALUES (?, ?, ?)",
        (g.user["id"], track_id, int(time.time())),
    )
    conn.commit()
    conn.close()
    return jsonify({"liked": True})


@auth_bp.route("/likes/<track_id>", methods=["DELETE"])
@require_auth
def remove_like(track_id):
    conn = get_db()
    conn.execute(
        "DELETE FROM liked_tracks WHERE user_id = ? AND track_id = ?",
        (g.user["id"], track_id),
    )
    conn.commit()
    conn.close()
    return jsonify({"liked": False})
