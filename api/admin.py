"""Admin CLI for API keys and monthly quotas.

Examples:
    python -m api.admin create-key --name "Acme production" --quota 500000 --rate 120
    python -m api.admin list-keys
    python -m api.admin revoke --key-id key_ab12cd34
"""

from __future__ import annotations

import argparse
import json
import sys

from api.accounts import AccountStore
from tts.config import Settings


def _store() -> AccountStore:
    return AccountStore(Settings.from_env().api_db)


def _parse_scopes(value: str) -> tuple[str, ...]:
    scopes = tuple(part.strip() for part in value.split(",") if part.strip())
    return scopes or ("tts", "voices")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create-key", help="Create a new API key (shown once).")
    create.add_argument("--name", required=True)
    create.add_argument("--quota", type=int, default=100_000, help="Monthly character quota; -1 = unlimited.")
    create.add_argument("--rate", type=int, default=60, help="Requests per minute; 0 = unlimited.")
    create.add_argument(
        "--scopes", default="tts,voices",
        help="Comma-separated scopes: tts,voices,admin.",
    )
    create.add_argument("--json", action="store_true")

    list_cmd = sub.add_parser("list-keys", help="List key metadata (never the secret).")
    list_cmd.add_argument("--json", action="store_true")

    revoke = sub.add_parser("revoke", help="Revoke a key by id.")
    revoke.add_argument("--key-id", required=True)

    args = parser.parse_args(argv)
    store = _store()

    if args.command == "create-key":
        raw, account = store.create_key(
            args.name,
            scopes=_parse_scopes(args.scopes),
            monthly_char_quota=args.quota,
            rate_limit_per_min=args.rate,
        )
        payload = {
            "api_key": raw,
            "key_id": account.key_id,
            "name": account.name,
            "scopes": list(account.scopes),
            "monthly_char_quota": account.monthly_char_quota,
            "rate_limit_per_min": account.rate_limit_per_min,
            "warning": "This is the only time the secret will be shown.",
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"API key (store it now): {raw}")
            print(f"key_id: {account.key_id}")
            print(f"scopes: {', '.join(account.scopes)}")
            print(f"monthly character quota: {account.monthly_char_quota}")
            print(f"rate limit: {account.rate_limit_per_min}/min")
        return 0

    if args.command == "list-keys":
        rows = store.list_keys()
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                status = "revoked" if row["revoked_at"] else "active"
                print(
                    f"{row['key_id']}  {status:7}  {row['name']}  "
                    f"quota={row['monthly_char_quota']}  rate={row['rate_limit_per_min']}/min  "
                    f"scopes={row['scopes']}"
                )
        return 0

    if args.command == "revoke":
        if store.revoke(args.key_id):
            print(f"Revoked {args.key_id}")
            return 0
        print(f"No active key with id {args.key_id}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
