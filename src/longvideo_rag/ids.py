from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=10).hexdigest()
    return f"{prefix}_{digest}"
