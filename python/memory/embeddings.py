"""Embedding service for the memory system.

Uses sentence-transformers with CPU-only inference to avoid conflicting
with GPU rentals. Tries the embedding daemon first to skip the ~25-second
cold load; falls back to in-process load if the daemon is unreachable.

Set HERMES_EMBED_DAEMON=0 to disable daemon entirely (always in-process).
"""

from __future__ import annotations

import os
import threading

from memory.config import EMBEDDING_MODEL, EMBEDDING_DEVICE


# Pinned commit on Alibaba-NLP/gte-modernbert-base to prevent silent
# model swap if the upstream HF repo changes (or is compromised).
# Update this SHA when intentionally upgrading; re-embedding pipeline
# in migrate_embeddings.py handles the schema migration.
EMBEDDING_REVISION = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"


# Module-level singleton — share one warm model across all in-process
# EmbeddingService instances. Eliminates double-load cost when multiple
# subsystems within the same Python process construct EmbeddingService
# (e.g. MemoryStore + write_gate + cleanup all in one CLI invocation).
# Cross-process sharing requires the daemon (see docs/memory/DAEMON_DESIGN.md).
_SHARED_MODEL = None
_SHARED_MODEL_KEY: tuple[str, str, str] | None = None
_SHARED_LOCK = threading.Lock()


class EmbeddingService:
    """Lazy-loaded sentence transformer for CPU-only embedding."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = EMBEDDING_DEVICE,
        revision: str = EMBEDDING_REVISION,
    ):
        self._model = None
        self._model_name = model_name
        self._device = device
        # Resilient device: if cuda is requested but unavailable (a cpu-only torch
        # build, or a stale HERMES_EMBED_DEVICE=cuda env), fall back to cpu instead
        # of crashing the embed path. A crash here yields 0 candidates -> an empty
        # memory block / lost L1->L2 capture, worse than a slower cpu embed. (2026-06-21)
        if "cuda" in str(self._device).lower():
            try:
                import torch
                if not torch.cuda.is_available():
                    self._device = "cpu"
            except Exception:
                self._device = "cpu"
        self._revision = revision
        self._lock = threading.Lock()
        # Daemon client — lazy-initialized
        self._daemon_client = None
        self._daemon_disabled = os.environ.get("HERMES_EMBED_DAEMON", "1") == "0"

    def _load(self):
        global _SHARED_MODEL, _SHARED_MODEL_KEY
        if self._model is not None:
            return
        # Try to reuse a process-wide warm model first
        key = (self._model_name, self._device, self._revision)
        with _SHARED_LOCK:
            if _SHARED_MODEL is not None and _SHARED_MODEL_KEY == key:
                self._model = _SHARED_MODEL
                return
            with self._lock:
                if self._model is not None:
                    return
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                    revision=self._revision,
                )
                _SHARED_MODEL = self._model
                _SHARED_MODEL_KEY = key

    def _try_daemon(self, texts: list[str]) -> list[list[float]] | None:
        """Try the embedding daemon. Returns None on any failure (caller
        should fall back to in-process load)."""
        if self._daemon_disabled:
            return None
        try:
            if self._daemon_client is None:
                from memory.embed_client import DaemonClient
                self._daemon_client = DaemonClient()
            return self._daemon_client.embed(texts)
        except Exception:
            # On any failure, disable daemon for the rest of THIS process
            # to avoid thrashing retries. A fresh process will try again.
            self._daemon_disabled = True
            return None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns normalized vectors.

        Tries the embedding daemon first (skip cold load); falls back to
        in-process model load if the daemon is unreachable.
        """
        if not texts:
            return []
        result = self._try_daemon(texts)
        if result is not None:
            return result
        # Fallback: in-process load (the old path)
        self._load()
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        self._load()
        return self._model.get_embedding_dimension()
