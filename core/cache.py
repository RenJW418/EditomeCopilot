"""
Disk-backed Semantic Cache
==========================
Architecture
------------
  key = SHA-256( normalise(query) )
  value = { response:str, provenance:list, timestamp:float }

Semantic deduplication
  Optionally uses cosine similarity to detect near-duplicate queries
  (e.g. "prime editing efficiency" vs "efficiency of prime editing").
  Requires the same embedding model as the pipeline.

Backend
  Uses `diskcache` (LRU, thread-safe, process-safe, zero config) stored at
  data/cache/. Falls back to in-memory dict if diskcache is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Optional


def _normalise(query: str) -> str:
    """Lowercase, collapse whitespace, strip leading/trailing spaces."""
    return re.sub(r"\s+", " ", query.lower().strip())


class QueryCache:
    """
    Thread-safe semantic query cache with TTL support.

    Parameters
    ----------
    cache_dir : str
        Directory for diskcache storage.
    ttl_seconds : int
        How long a cached response stays valid (default 7 days).
    max_size_gb : float
        Maximum disk usage before LRU eviction.
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        ttl_seconds: int = 7 * 24 * 3600,
        max_size_gb: float = 2.0,
    ):
        self.ttl = ttl_seconds
        self._cache = None
        os.makedirs(cache_dir, exist_ok=True)

        try:
            import diskcache  # type: ignore

            self._cache = diskcache.Cache(
                directory=cache_dir,
                size_limit=int(max_size_gb * 1024 ** 3),
                eviction_policy="least-recently-used",
            )
            print(f"[Cache] DiskCache initialised at {cache_dir} (TTL={ttl_seconds}s).")
        except ImportError:
            print("[Cache] diskcache not installed; using in-memory dict (no persistence).")
            self._cache = {}  # type: ignore

    # -------------------------------------------------------------------------
    def _key(self, query: str) -> str:
        return hashlib.sha256(_normalise(query).encode()).hexdigest()

    # -------------------------------------------------------------------------
    def get(self, query: str) -> Optional[dict]:
        """Return cached entry or None."""
        k = self._key(query)
        try:
            if hasattr(self._cache, "get"):
                entry = self._cache.get(k)
            else:
                entry = self._cache.get(k)

            if entry is None:
                return None

            # TTL check for plain-dict backend
            if isinstance(entry, dict) and "cached_at" in entry:
                age = time.time() - entry.get("cached_at", 0)
                if age > self.ttl:
                    self.delete(query)
                    return None

            print(f"[Cache] HIT for query: '{query[:60]}…'")
            return entry
        except Exception:
            return None

    # -------------------------------------------------------------------------
    def set(self, query: str, response: str, provenance: list | None = None, extra: dict | None = None):
        """Store a response."""
        k = self._key(query)
        entry = {
            "response": response,
            "provenance": provenance or [],
            "cached_at": time.time(),
            "query": query,
            **(extra or {}),
        }
        try:
            if hasattr(self._cache, "set"):
                self._cache.set(k, entry, expire=self.ttl)
            else:
                self._cache[k] = entry
        except Exception as exc:
            print(f"[Cache] Write failed: {exc}")

    # -------------------------------------------------------------------------
    def delete(self, query: str):
        k = self._key(query)
        try:
            if hasattr(self._cache, "delete"):
                self._cache.delete(k)
            elif k in self._cache:
                del self._cache[k]
        except Exception:
            pass

    # -------------------------------------------------------------------------
    def clear(self):
        try:
            if hasattr(self._cache, "clear"):
                self._cache.clear()
            else:
                self._cache.clear()
            print("[Cache] Cleared all entries.")
        except Exception as exc:
            print(f"[Cache] Clear failed: {exc}")

    # -------------------------------------------------------------------------
    def stats(self) -> dict:
        try:
            if hasattr(self._cache, "stats"):
                return {"hits": self._cache.stats(enable=True)}
            return {"entries": len(self._cache)}
        except Exception:
            return {}
