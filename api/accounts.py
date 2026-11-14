"""SQLite-backed API keys, monthly quotas and usage accounting.

Only a SHA-256 hash of each key is stored.  Plaintext keys are shown once by the
admin CLI.  This is intentionally dependency-free and works on a laptop or a
single GPU worker; for multi-region billing, replace this store with your
database of record.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY,
    key_hash TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    scopes TEXT NOT NULL,
    monthly_char_quota INTEGER NOT NULL,
    rate_limit_per_min INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    revoked_at TEXT
);
CREATE TABLE IF NOT EXISTS usage (
    key_id TEXT NOT NULL,
    period TEXT NOT NULL,
    chars INTEGER NOT NULL DEFAULT 0,
    audio_ms INTEGER NOT NULL DEFAULT 0,
    requests INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (key_id, period)
);
CREATE INDEX IF NOT EXISTS idx_usage_period ON usage(period);
"""

DEFAULT_SCOPES = ("tts", "voices")
ADMIN_SCOPES = ("tts", "voices", "admin")


class AccountError(ValueError):
    pass


@dataclass(frozen=True)
class Account:
    key_id: str
    name: str
    scopes: tuple[str, ...]
    monthly_char_quota: int
    rate_limit_per_min: int
    created_at: str

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes or "admin" in self.scopes


@dataclass(frozen=True)
class Usage:
    period: str
    chars: int
    audio_ms: int
    requests: int
    monthly_char_quota: int

    @property
    def chars_remaining(self) -> int | None:
        if self.monthly_char_quota < 0:
            return None
        return max(0, self.monthly_char_quota - self.chars)

    def as_dict(self) -> dict:
        return {
            "period": self.period,
            "chars": self.chars,
            "audio_ms": self.audio_ms,
            "requests": self.requests,
            "monthly_char_quota": self.monthly_char_quota,
            "chars_remaining": self.chars_remaining,
        }


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def current_period(now: float | None = None) -> str:
    stamp = time.gmtime(now)
    return time.strftime("%Y-%m", stamp)


class AccountStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=15.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=15000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(SCHEMA)

    def create_key(
        self,
        name: str,
        *,
        scopes: tuple[str, ...] = DEFAULT_SCOPES,
        monthly_char_quota: int = 100_000,
        rate_limit_per_min: int = 60,
    ) -> tuple[str, Account]:
        name = (name or "").strip()
        if not name:
            raise AccountError("A key name is required.")
        if not scopes or any(scope not in ADMIN_SCOPES for scope in scopes):
            raise AccountError(f"Scopes must be a subset of {ADMIN_SCOPES}.")
        raw_key = "vsk_live_" + secrets.token_urlsafe(32)
        key_id = "key_" + secrets.token_hex(8)
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO api_keys(key_id, key_hash, name, scopes, monthly_char_quota, "
                "rate_limit_per_min, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    key_id,
                    hash_key(raw_key),
                    name,
                    " ".join(scopes),
                    int(monthly_char_quota),
                    int(rate_limit_per_min),
                    created_at,
                ),
            )
        return raw_key, Account(
            key_id=key_id,
            name=name,
            scopes=tuple(scopes),
            monthly_char_quota=int(monthly_char_quota),
            rate_limit_per_min=int(rate_limit_per_min),
            created_at=created_at,
        )

    def authenticate(self, raw_key: str) -> Account | None:
        if not raw_key:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT key_id, name, scopes, monthly_char_quota, rate_limit_per_min, created_at "
                "FROM api_keys WHERE key_hash = ? AND revoked_at IS NULL",
                (hash_key(raw_key),),
            ).fetchone()
        if row is None:
            return None
        return Account(
            key_id=row["key_id"],
            name=row["name"],
            scopes=tuple(row["scopes"].split()),
            monthly_char_quota=int(row["monthly_char_quota"]),
            rate_limit_per_min=int(row["rate_limit_per_min"]),
            created_at=row["created_at"],
        )

    def reserve_chars(self, account: Account, chars: int) -> bool:
        """Atomically reserve quota. Returns False when the monthly limit is exhausted."""
        chars = max(0, int(chars))
        period = current_period()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT chars FROM usage WHERE key_id = ? AND period = ?",
                    (account.key_id, period),
                ).fetchone()
                used = int(row["chars"]) if row else 0
                quota = account.monthly_char_quota
                if quota >= 0 and used + chars > quota:
                    connection.rollback()
                    return False
                connection.execute(
                    "INSERT INTO usage(key_id, period, chars, requests) VALUES (?, ?, ?, 1) "
                    "ON CONFLICT(key_id, period) DO UPDATE SET "
                    "chars = chars + excluded.chars, requests = requests + 1",
                    (account.key_id, period, chars),
                )
                connection.commit()
                return True
            except Exception:
                connection.rollback()
                raise

    def refund_chars(self, account: Account, chars: int) -> None:
        chars = max(0, int(chars))
        if chars == 0:
            return
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE usage SET chars = MAX(0, chars - ?) WHERE key_id = ? AND period = ?",
                (chars, account.key_id, current_period()),
            )

    def record_audio_ms(self, account: Account, audio_ms: int) -> None:
        audio_ms = max(0, int(audio_ms))
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO usage(key_id, period, audio_ms) VALUES (?, ?, ?) "
                "ON CONFLICT(key_id, period) DO UPDATE SET audio_ms = audio_ms + excluded.audio_ms",
                (account.key_id, current_period(), audio_ms),
            )

    def usage(self, account: Account) -> Usage:
        period = current_period()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT chars, audio_ms, requests FROM usage WHERE key_id = ? AND period = ?",
                (account.key_id, period),
            ).fetchone()
        return Usage(
            period=period,
            chars=int(row["chars"]) if row else 0,
            audio_ms=int(row["audio_ms"]) if row else 0,
            requests=int(row["requests"]) if row else 0,
            monthly_char_quota=account.monthly_char_quota,
        )

    def list_keys(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT key_id, name, scopes, monthly_char_quota, rate_limit_per_min, "
                "created_at, revoked_at FROM api_keys ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def revoke(self, key_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE key_id = ? AND revoked_at IS NULL",
                (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), key_id),
            )
            return cursor.rowcount > 0
