"""API-key authentication and in-process rate limiting.

For a single worker this needs no Redis.  If you run multiple API processes,
replace the token bucket with a shared limiter (Redis or your gateway) so the
per-key rate is global rather than per process.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from api.accounts import ADMIN_SCOPES, Account, AccountStore
from tts.config import Settings


@dataclass(frozen=True)
class AuthContext:
    account: Account
    raw_key: str | None
    static: bool = False

    @property
    def key_id(self) -> str:
        return self.account.key_id


class RateLimiter:
    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key_id: str, limit_per_minute: int) -> tuple[bool, int]:
        if limit_per_minute <= 0:
            return True, 0
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            bucket = self._events.setdefault(key_id, deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= limit_per_minute:
                retry_after = max(1, int(60.0 - (now - bucket[0])) + 1)
                return False, retry_after
            bucket.append(now)
            return True, 0


class AuthManager:
    def __init__(self, settings: Settings, accounts: AccountStore) -> None:
        self.settings = settings
        self.accounts = accounts
        self.rate_limiter = RateLimiter()

    def _static_account(self) -> Account:
        return Account(
            key_id="static-local",
            name="local-static-key",
            scopes=ADMIN_SCOPES,
            monthly_char_quota=-1,
            rate_limit_per_min=max(0, self.settings.api_default_rate_limit),
            created_at="local",
        )

    @staticmethod
    def _extract_key(request: Request) -> str | None:
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
            if token:
                return token
        key = request.headers.get("x-api-key", "").strip()
        return key or None

    def authenticate(self, raw_key: str | None) -> AuthContext | None:
        if not self.settings.api_require_auth:
            return AuthContext(self._static_account(), None, static=True)
        if not raw_key:
            return None
        if self.settings.api_static_key and secrets.compare_digest(
            raw_key, self.settings.api_static_key
        ):
            return AuthContext(self._static_account(), raw_key, static=True)
        account = self.accounts.authenticate(raw_key)
        if account is None:
            return None
        return AuthContext(account, raw_key, static=False)

    async def require(self, request: Request, scope: str = "tts") -> AuthContext:
        context = self.authenticate(self._extract_key(request))
        if context is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not context.account.has_scope(scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key does not have the {scope!r} scope.",
            )
        allowed, retry_after = self.rate_limiter.allow(
            context.key_id, context.account.rate_limit_per_min
        )
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded.",
                headers={"Retry-After": str(retry_after)},
            )
        return context


def require_tts(request: Request) -> AuthContext:
    manager: AuthManager = request.app.state.auth
    # FastAPI dependencies can be sync; run the coroutine safely via the app
    # state's event loop is unnecessary because this function is only called
    # from sync dependencies.  The async wrapper below is used by routes.
    raise RuntimeError("Use auth.require_scope(...) instead of require_tts directly.")


def require_scope(scope: str):
    async def dependency(request: Request) -> AuthContext:
        manager: AuthManager = request.app.state.auth
        return await manager.require(request, scope)

    return dependency
