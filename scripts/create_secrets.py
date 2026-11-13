"""Create credentials for a new deployment without overwriting existing secrets."""

import argparse
import getpass
import json
import re
import secrets
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--directory", type=Path, default=Path("secrets"))
    parser.add_argument("--gateway-key-file", type=Path)
    args = parser.parse_args(argv)
    key = (
        args.gateway_key_file.read_text().strip()
        if args.gateway_key_file
        else getpass.getpass("Gateway API key: ").strip()
    )
    if (
        not key
        or not args.subject.strip()
        or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.tenant)
    ):
        parser.error(
            "A gateway API key, named subject and valid tenant identifier are required"
        )
    args.directory.mkdir(mode=0o700, parents=True)
    token = secrets.token_urlsafe(32)
    contents = {
        "auth_tokens.json": json.dumps(
            {
                token: {
                    "subject": args.subject,
                    "tenant_id": args.tenant,
                    "roles": ["admin"],
                    "allowed_domains": ["*"],
                }
            },
            indent=2,
        ),
        "artifact_hmac": secrets.token_urlsafe(48),
        "gateway_key": key,
    }
    for name, value in contents.items():
        path = args.directory / name
        path.write_text(value + "\n", encoding="utf-8")
        path.chmod(0o444)
    print(
        f"Created credential files in {args.directory}. Store the admin token securely and back up the artifact signing key."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
