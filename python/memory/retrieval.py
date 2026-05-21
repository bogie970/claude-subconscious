"""Intelligent retrieval with triple scoring and hybrid search.

Implements the retrieval strategy from Generative Agents (Park et al., 2023):
  score = alpha * relevance + beta * recency + gamma * importance

Combined with hybrid search (vector + BM25) merged via reciprocal rank fusion.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

log = logging.getLogger("memory.retrieval")

from memory.config import (
    ALPHA_RELEVANCE,
    BETA_RECENCY,
    GAMMA_IMPORTANCE,
    RECENCY_DECAY_RATE,
    DEFAULT_TOP_K,
    OVERFETCH_MULTIPLIER,
)
from memory.schema import MemoryRecord, ScoredMemory
from memory.store import MemoryStore


# A2 (Plan A Day 1): drop low-similarity candidates BEFORE min-max normalization.
# Without this floor, min-max inflates noise (~0.2 raw similarity) to relevance=1.0
# whenever ALL candidates are noise — TRACE-3 reproduced this live for the
# "session compaction protocol" query (4/5 results were tier=candidate scratch).
# Keep in sync with aisys/memory/pretooluse_memcheck.py SIM_FLOOR = 0.45.
SIM_FLOOR = 0.45


class TripleScoredRetriever:
    """Retrieves memories scored by relevance + recency + importance (Park et al. 2023)."""

    def __init__(
        self,
        store: MemoryStore,
        alpha: float = ALPHA_RELEVANCE,
        beta: float = BETA_RECENCY,
        gamma: float = GAMMA_IMPORTANCE,
        decay_rate: float = RECENCY_DECAY_RATE,
    ):
        self._store = store
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._decay_rate = decay_rate

    def _compute_recency(self, last_accessed: datetime) -> float:
        now = datetime.now(timezone.utc)
        if last_accessed.tzinfo is None:
            last_accessed = last_accessed.replace(tzinfo=timezone.utc)
        hours = max(0, (now - last_accessed).total_seconds() / 3600)
        return math.exp(-self._decay_rate * hours)

    def _compute_raw_scores(self, candidates: list[ScoredMemory]) -> list[tuple[ScoredMemory, float, float, float]]:
        """Compute raw relevance, recency, importance for each candidate."""
        raw = []
        for sm in candidates:
            recency = self._compute_recency(sm.record.last_accessed)
            raw.append((sm, sm.relevance, recency, sm.record.importance))
        return raw

    def _normalize_and_score(self, raw: list[tuple[ScoredMemory, float, float, float]]) -> list[ScoredMemory]:
        """Min-max normalize each dimension across the candidate set (Park et al. 2023)."""
        if not raw:
            return []

        relevances = [r[1] for r in raw]
        recencies = [r[2] for r in raw]
        importances = [r[3] for r in raw]

        def _minmax(val: float, vals: list[float]) -> float:
            lo, hi = min(vals), max(vals)
            if hi == lo:
                return 1.0
            return (val - lo) / (hi - lo)

        scored = []
        for sm, rel, rec, imp in raw:
            norm_rel = _minmax(rel, relevances)
            norm_rec = _minmax(rec, recencies)
            norm_imp = _minmax(imp, importances)
            combined = (
                self._alpha * norm_rel
                + self._beta * norm_rec
                + self._gamma * norm_imp
            )
            scored.append(ScoredMemory(
                record=sm.record,
                relevance=norm_rel,
                recency=norm_rec,
                importance_score=norm_imp,
                combined_score=combined,
                vector_distance=sm.vector_distance,
            ))
        return scored

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        namespaces: list[str] | None = None,
        category_filter: str | None = None,
        memory_type_filter: str | None = None,
        include_archived: bool = False,
        expand_links: bool = True,
        track_access: bool = True,
    ) -> list[ScoredMemory]:
        """Retrieve top-k memories by triple score with min-max normalization.

        If expand_links=True, linked memories from top results are pulled in
        and scored alongside vector results (graph expansion).
        """
        candidates = self._store.search(
            query,
            k=k * OVERFETCH_MULTIPLIER,
            namespaces=namespaces,
            category_filter=category_filter,
            memory_type_filter=memory_type_filter,
            include_archived=include_archived,
        )

        # A2: low-similarity cutoff before normalization
        candidates = [c for c in candidates if c.relevance >= SIM_FLOOR]
        if not candidates:
            return []

        raw = self._compute_raw_scores(candidates)

        if expand_links:
            seen_ids = {sm.record.id for sm, _, _, _ in raw}
            # Pre-score to find top-k for link expansion
            pre_scored = self._normalize_and_score(raw)
            pre_scored.sort(key=lambda s: s.combined_score, reverse=True)
            for sm in pre_scored[:k]:
                for linked in self._store.get_linked(sm.record.id):
                    if linked.id not in seen_ids and not linked.archived:
                        seen_ids.add(linked.id)
                        recency = self._compute_recency(linked.last_accessed)
                        raw.append((
                            ScoredMemory(
                                record=linked,
                                relevance=sm.relevance * 0.8,
                                importance_score=linked.importance,
                            ),
                            sm.relevance * 0.8,
                            recency,
                            linked.importance,
                        ))

        # Final normalization across all candidates including graph-expanded ones
        scored = self._normalize_and_score(raw)
        scored.sort(key=lambda s: s.combined_score, reverse=True)

        if track_access:
            self._store.batch_update_access([sm.record.id for sm in scored[:min(k, 5)]])

            # Log retrieval event for self-tuning feedback
            try:
                from memory.feedback import log_retrieval_event
                log_retrieval_event(query, scored[:k])
            except Exception:
                pass  # feedback logging is non-critical

        return scored[:k]


class HybridRetriever:
    """Combines vector search with BM25 keyword search via reciprocal rank fusion."""

    RRF_K = 60  # standard RRF constant

    def __init__(self, store: MemoryStore):
        self._store = store
        self._triple = TripleScoredRetriever(store)
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._bm25_records: dict[str, MemoryRecord] = {}
        self._bm25_namespaces: frozenset[str] | None = None

    def _build_bm25_index(self, namespaces: list[str] | None = None):
        """Build/rebuild the BM25 index from all active memories."""
        from rank_bm25 import BM25Okapi

        records = self._store.get_all(include_archived=False)
        if namespaces:
            records = [r for r in records if r.namespace in namespaces]
        self._bm25_records = {r.id: r for r in records}
        self._bm25_ids = [r.id for r in records]
        self._bm25_namespaces = frozenset(namespaces) if namespaces else None
        corpus = [r.content.lower().split() for r in records]
        if corpus:
            self._bm25 = BM25Okapi(corpus)
        else:
            self._bm25 = None

    def _bm25_search(self, query: str, k: int = 20, namespaces: list[str] | None = None) -> list[tuple[str, float]]:
        """Return top-k (id, score) pairs from BM25."""
        ns_key = frozenset(namespaces) if namespaces else None
        if self._bm25 is None or ns_key != self._bm25_namespaces:
            self._build_bm25_index(namespaces)
        if self._bm25 is None:
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(self._bm25_ids, scores), key=lambda x: x[1], reverse=True
        )
        return [(rid, s) for rid, s in ranked[:k] if s > 0]

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        namespaces: list[str] | None = None,
        category_filter: str | None = None,
        memory_type_filter: str | None = None,
        rebuild_bm25: bool = False,
    ) -> list[ScoredMemory]:
        """Hybrid retrieval: vector + BM25, merged via RRF, then triple-scored."""
        if rebuild_bm25 or self._bm25 is None:
            self._build_bm25_index(namespaces)

        # Vector search (via triple scorer for full scoring)
        vector_results = self._triple.retrieve(
            query,
            k=k * 3,
            namespaces=namespaces,
            category_filter=category_filter,
            memory_type_filter=memory_type_filter,
        )

        # BM25 search
        bm25_hits = self._bm25_search(query, k=k * 3, namespaces=namespaces)

        # Build RRF scores
        rrf_scores: dict[str, float] = {}

        for rank, sm in enumerate(vector_results):
            rrf_scores[sm.record.id] = rrf_scores.get(sm.record.id, 0) + 1.0 / (
                self.RRF_K + rank + 1
            )

        for rank, (rid, _) in enumerate(bm25_hits):
            rrf_scores[rid] = rrf_scores.get(rid, 0) + 1.0 / (
                self.RRF_K + rank + 1
            )

        # Build result map from vector results
        result_map: dict[str, ScoredMemory] = {
            sm.record.id: sm for sm in vector_results
        }

        # Add BM25-only hits
        for rid, _ in bm25_hits:
            if rid not in result_map and rid in self._bm25_records:
                record = self._bm25_records[rid]
                result_map[rid] = ScoredMemory(
                    record=record,
                    relevance=0.0,
                    importance_score=record.importance,
                )

        # Sort by RRF score and return top-k
        ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
        return [result_map[rid] for rid in ranked_ids if rid in result_map]

    def refresh_index(self):
        """Force rebuild of the BM25 index (call after inserts)."""
        self._build_bm25_index()
