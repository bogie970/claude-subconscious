"""MemoryStore — LanceDB-backed vector store for hierarchical agent memory.

Wraps LanceDB with domain-specific operations: insert, search, update,
archive, and tier management. Designed for embedded use (no server process).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import lancedb
import pyarrow as pa
from filelock import FileLock

from memory.config import (
    LANCEDB_PATH,
    MEMORY_DIR,
    TABLE_NAME,
    EMBEDDING_DIM,
    DEFAULT_TOP_K,
)
from memory.embeddings import EmbeddingService
from memory.enrichment import enrich_memory, build_enriched_text
from memory.schema import MemoryRecord, ScoredMemory

def _sanitize(value: str) -> str:
    """Escape single quotes for LanceDB SQL where clauses."""
    return value.replace("'", "''")


LANCE_SCHEMA = pa.schema(
    [
        pa.field("id", pa.utf8()),
        pa.field("content", pa.utf8()),
        pa.field("memory_type", pa.utf8()),
        pa.field("category", pa.utf8()),
        pa.field("source", pa.utf8()),
        pa.field("namespace", pa.utf8()),
        pa.field("created_at", pa.utf8()),
        pa.field("last_accessed", pa.utf8()),
        pa.field("access_count", pa.int32()),
        pa.field("importance", pa.float32()),
        pa.field("tags", pa.utf8()),
        pa.field("links", pa.utf8()),
        pa.field("session_id", pa.utf8()),
        pa.field("metadata_json", pa.utf8()),
        pa.field("archived", pa.bool_()),
        pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    ]
)

# v2 schema additions: provenance, tier, bitemporal, contradiction, embedding versioning.
# Added by migrate_v2.backfill_v2() via LanceDB add_columns (metadata-only for new cols).
# New v2 records via insert() should populate all these fields with defaults.
V2_FIELDS = [
    pa.field("tier", pa.utf8()),                        # candidate | probationary | verified | tombstoned
    pa.field("provenance", pa.utf8()),                  # user_stated | llm_inferred | tool_observed | file_grounded
    pa.field("source_ref", pa.utf8()),                  # transcript_id:turn | file:line | tool_call_id
    pa.field("writer", pa.utf8()),                      # user | subconscious_haiku | sonnet_promoter | etc
    pa.field("confidence", pa.float32()),               # 0.0 - 1.0
    pa.field("valid_from", pa.utf8()),                  # ISO timestamp; bitemporal valid time
    pa.field("valid_to", pa.utf8()),                    # ISO timestamp; "" = currently valid
    pa.field("supersedes", pa.utf8()),                  # nullable id of replaced memory ("" = none)
    pa.field("superseded_by", pa.utf8()),               # nullable id of replacing memory
    pa.field("contradiction_state", pa.utf8()),         # clean | flagged_haiku | adjudicated | user_review
    pa.field("conflict_with", pa.utf8()),               # JSON list of ids
    pa.field("seen_count", pa.int32()),                 # re-encounter counter
    pa.field("last_seen_at", pa.utf8()),                # ISO timestamp
    pa.field("promoted_at", pa.utf8()),                 # ISO timestamp; "" = not yet promoted
    pa.field("embedding_model", pa.utf8()),             # current model name
    pa.field("embedding_version", pa.int32()),          # version number
]

LANCE_SCHEMA_V2 = pa.schema(list(LANCE_SCHEMA) + V2_FIELDS)

# Audit log table — append-only, separate from memory table
LANCE_AUDIT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.utf8()),                      # UUID
        pa.field("memory_id", pa.utf8()),               # references memory.id
        pa.field("op", pa.utf8()),                      # create | update | promote | tombstone | supersede | backfill_v2
        pa.field("who", pa.utf8()),                     # writer agent
        pa.field("when", pa.utf8()),                    # ISO timestamp
        pa.field("why", pa.utf8()),                     # reason / context
        pa.field("before", pa.utf8()),                  # JSON snapshot before op (nullable)
        pa.field("after", pa.utf8()),                   # JSON snapshot after op
    ]
)

AUDIT_TABLE_NAME = "memory_audit"


class MemoryStore:
    """LanceDB-backed memory store with embedding integration."""

    def __init__(
        self,
        db_path: str = LANCEDB_PATH,
        embedder: EmbeddingService | None = None,
    ):
        os.makedirs(db_path, exist_ok=True)
        self._db = lancedb.connect(db_path)
        self._embedder = embedder or EmbeddingService()
        self._lock = FileLock(f"{db_path}.lock", timeout=60)
        self._table = self._get_or_create_table()
        self._ensure_audit_table()
        # A3 (Plan A Day 1): fail-loud if a v1-schema store slipped through.
        # TRACE-3 showed v1 fallback silently drops the tier column —
        # quarantine taxonomy invisible, every row injected as if verified.
        if not self._has_v2_schema():
            raise RuntimeError(
                f"v1 schema fallback at {self._db.uri}/{TABLE_NAME} — "
                "refusing to run. v2 columns missing. "
                "Run migrations/migrate_v2.py to upgrade, or delete the table to recreate."
            )

    def _ensure_audit_table(self):
        """Create memory_audit table eagerly. Prevents lazy-create race
        between concurrent _audit() calls."""
        if AUDIT_TABLE_NAME not in list(self._db.table_names()):
            self._db.create_table(AUDIT_TABLE_NAME, schema=LANCE_AUDIT_SCHEMA)

    def _get_or_create_table(self):
        try:
            return self._db.open_table(TABLE_NAME)
        except (FileNotFoundError, ValueError):
            # A3 (Plan A Day 1): fresh installs use v2 schema directly.
            return self._db.create_table(TABLE_NAME, schema=LANCE_SCHEMA_V2)

    def _has_v2_schema(self) -> bool:
        """Check if the underlying table has v2 columns (post-migration)."""
        v2_field_names = {f.name for f in V2_FIELDS}
        schema_field_names = {f.name for f in self._table.schema}
        return v2_field_names.issubset(schema_field_names)

    def insert(self, record: MemoryRecord, enrich: bool = True, evolve: bool = False) -> str:
        """Insert a single memory record. Returns the record ID.

        If enrich=True, uses LLM to generate keywords/context/tags (A-MEM
        pattern) and embeds over the enriched text for better retrieval.
        If evolve=True, checks nearest neighbors and evolves them based
        on the new memory (A-MEM memory evolution).
        """
        enrichment = {}
        embed_text = record.content

        if enrich:
            enrichment = enrich_memory(record.content, record.category)
            embed_text = build_enriched_text(record.content, enrichment)
            if enrichment.get("generated_tags"):
                record.tags = list(set(record.tags + enrichment["generated_tags"]))
            if enrichment.get("keywords") or enrichment.get("context"):
                record.metadata["enrichment"] = enrichment

        vector = self._embedder.embed_one(embed_text)
        row = record.to_lance_dict(vector, include_v2=self._has_v2_schema())
        with self._lock:
            self._table.add([row])

        if evolve:
            from memory.evolution import evolve_on_insert
            evolve_on_insert(self, record)

        return record.id

    def insert_many(self, records: list[MemoryRecord], enrich: bool = True) -> list[str]:
        """Insert multiple memory records. Returns list of IDs.

        If enrich=True, each record is LLM-enriched before embedding.
        """
        if not records:
            return []

        embed_texts = []
        for r in records:
            if enrich:
                enrichment = enrich_memory(r.content, r.category)
                embed_texts.append(build_enriched_text(r.content, enrichment))
                if enrichment.get("generated_tags"):
                    r.tags = list(set(r.tags + enrichment["generated_tags"]))
                if enrichment.get("keywords") or enrichment.get("context"):
                    r.metadata["enrichment"] = enrichment
            else:
                embed_texts.append(r.content)

        vectors = self._embedder.embed(embed_texts)
        include_v2 = self._has_v2_schema()
        rows = [r.to_lance_dict(v, include_v2=include_v2) for r, v in zip(records, vectors)]
        with self._lock:
            self._table.add(rows)
        return [r.id for r in records]

    def search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        namespaces: list[str] | None = None,
        category_filter: str | None = None,
        memory_type_filter: str | None = None,
        include_archived: bool = False,
        tier: tuple[str, ...] | None = ("verified", "probationary"),
    ) -> list[ScoredMemory]:
        """Vector similarity search with optional metadata filters.

        Returns ScoredMemory objects with only relevance (vector distance)
        populated. Use TripleScoredRetriever for full scoring.

        A1 (Plan A Day 1): default tier filter excludes candidate AND tombstoned.
        Pass tier=None to bypass (debug / memory_recall scope='archive' etc.).
        Closes TRACE-3 candidate-tier injection AND the 351 tombstone leak.
        """
        query_vector = self._embedder.embed_one(query)

        search_query = self._table.search(query_vector).limit(k)

        where_clauses = []
        if not include_archived:
            where_clauses.append("archived = false")
            # Bitemporal (#76): exclude SUPERSEDED rows (non-empty valid_to) from
            # default retrieval (mirrors aisys/memory/store.py — keep the two in sync).
            # Bypassed under include_archived so as-of/archive scope still sees the chain.
            if self._has_v2_schema():
                where_clauses.append("(valid_to = '' OR valid_to IS NULL)")
        if tier is not None:
            tier_list = ", ".join(f"'{_sanitize(t)}'" for t in tier)
            where_clauses.append(f"tier IN ({tier_list})")
        if namespaces:
            ns_list = ", ".join(f"'{_sanitize(n)}'" for n in namespaces)
            where_clauses.append(f"namespace IN ({ns_list})")
        if category_filter:
            where_clauses.append(f"category = '{_sanitize(category_filter)}'")
        if memory_type_filter:
            where_clauses.append(f"memory_type = '{_sanitize(memory_type_filter)}'")

        if where_clauses:
            search_query = search_query.where(" AND ".join(where_clauses))

        try:
            results = search_query.to_pandas()
        except FileNotFoundError:
            return []

        scored = []
        for _, row in results.iterrows():
            record = MemoryRecord.from_lance_row(row.to_dict())
            distance = float(row.get("_distance", 1.0))
            relevance = max(0.0, 1.0 - distance)
            scored.append(
                ScoredMemory(
                    record=record,
                    relevance=relevance,
                    importance_score=record.importance,
                    vector_distance=distance,
                )
            )

        scored.sort(key=lambda s: s.relevance, reverse=True)
        return scored[:k]

    def get_by_id(self, record_id: str) -> MemoryRecord | None:
        """Retrieve a single record by ID."""
        try:
            df = self._table.search().where(f"id = '{_sanitize(record_id)}'").limit(1).to_pandas()
        except FileNotFoundError:
            return None
        if df.empty:
            return None
        return MemoryRecord.from_lance_row(df.iloc[0].to_dict())

    def update_access(self, record_id: str) -> None:
        """Mark a memory as accessed (update last_accessed + increment count)."""
        self.batch_update_access([record_id])

    def batch_update_access(self, record_ids: list[str]) -> None:
        """Batch-update last_accessed and access_count for multiple records.

        Single SQL IN-clause update increments access_count in one pass,
        then compact_files() merges fragments to prevent version proliferation.
        """
        if not record_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        id_list = ", ".join(f"'{_sanitize(rid)}'" for rid in record_ids)
        with self._lock:
            # `updates` takes SQL expressions; access_count + 1 is evaluated per-row
            self._table.update(
                where=f"id IN ({id_list})",
                updates={
                    "last_accessed": f"'{now}'",
                    "access_count": "access_count + 1",
                },
            )
            try:
                self._table.compact_files()
            except Exception:
                pass

    def update_importance(self, record_id: str, new_importance: float) -> None:
        """Update the importance score of a memory."""
        with self._lock:
            self._table.update(
                where=f"id = '{_sanitize(record_id)}'",
                values={"importance": new_importance},
            )

    def update_content(self, record_id: str, new_content: str, re_embed: bool = True) -> None:
        """Update a memory's content, optionally recomputing its vector."""
        values: dict = {"content": new_content}
        if re_embed:
            vector = self._embedder.embed_one(new_content)
            values["vector"] = vector
        with self._lock:
            self._table.update(where=f"id = '{_sanitize(record_id)}'", values=values)

    def update_tags(self, record_id: str, tags: list[str]) -> None:
        """Replace the tags on a memory record."""
        with self._lock:
            self._table.update(
                where=f"id = '{_sanitize(record_id)}'",
                values={"tags": ",".join(tags)},
            )

    def update_links(self, record_id: str, link_ids: list[str]) -> None:
        """Replace the links on a memory record."""
        with self._lock:
            self._table.update(
                where=f"id = '{_sanitize(record_id)}'",
                values={"links": ",".join(link_ids)},
            )

    def update_vector(self, record_id: str, text: str) -> None:
        """Re-embed text and update the vector for a memory record."""
        vector = self._embedder.embed_one(text)
        with self._lock:
            self._table.update(
                where=f"id = '{_sanitize(record_id)}'",
                values={"vector": vector},
            )

    def get_linked(self, record_id: str) -> list[MemoryRecord]:
        """Get all memories linked from a given record."""
        record = self.get_by_id(record_id)
        if not record or not record.links:
            return []
        linked = []
        for lid in record.links:
            r = self.get_by_id(lid)
            if r:
                linked.append(r)
        return linked

    def archive(self, record_id: str) -> None:
        """Move a memory to the archived tier (L3)."""
        with self._lock:
            self._table.update(
                where=f"id = '{_sanitize(record_id)}'",
                values={"archived": True},
            )

    def unarchive(self, record_id: str) -> None:
        """Promote a memory back from archive to active tier."""
        with self._lock:
            self._table.update(
                where=f"id = '{_sanitize(record_id)}'",
                values={"archived": False},
            )

    def delete(self, record_id: str) -> None:
        """Permanently delete a memory record."""
        with self._lock:
            self._table.delete(f"id = '{_sanitize(record_id)}'")

    def count(self, include_archived: bool = False) -> int:
        """Count total memories using SQL pushdown (no full-table load)."""
        if include_archived:
            return self._table.count_rows()
        return self._table.count_rows("archived = false")

    def count_by_type(self, include_archived: bool = False) -> dict[str, int]:
        """Count memories grouped by memory_type using SQL pushdown."""
        from memory.schema import MemoryType
        counts = {}
        for mt in MemoryType:
            filt = f"memory_type = '{mt.value}'"
            if not include_archived:
                filt = f"archived = false AND {filt}"
            counts[mt.value] = self._table.count_rows(filt)
        return counts

    def get_recent(
        self,
        n: int = 5,
        source: str | None = None,
        memory_type: str | None = None,
        namespace: str | None = None,
    ) -> list[MemoryRecord]:
        """Get the N most recently created memories (filtered, not full-table)."""
        where_clauses = ["archived = false"]
        if source:
            where_clauses.append(f"source = '{_sanitize(source)}'")
        if memory_type:
            where_clauses.append(f"memory_type = '{_sanitize(memory_type)}'")
        if namespace:
            where_clauses.append(f"namespace = '{_sanitize(namespace)}'")
        filter_str = " AND ".join(where_clauses)
        try:
            df = self._table.search().where(filter_str).limit(n * 3).to_pandas()
        except FileNotFoundError:
            return []
        if df.empty:
            return []
        df = df.sort_values("created_at", ascending=False).head(n)
        return [MemoryRecord.from_lance_row(row.to_dict()) for _, row in df.iterrows()]

    def get_all(self, include_archived: bool = False, namespace: str | None = None) -> list[MemoryRecord]:
        """Get all memory records. Use sparingly — for consolidation only."""
        df = self._table.to_pandas()
        if not include_archived:
            df = df[~df["archived"]]
        if namespace:
            df = df[df["namespace"] == namespace]
        return [MemoryRecord.from_lance_row(row.to_dict()) for _, row in df.iterrows()]

    # ---------- v2 schema helpers (Phase B+) ----------

    def scan_v2(self, columns: list[str] | None = None) -> list[dict]:
        """Return all rows as dicts including v2 fields.

        columns: if provided, materialize ONLY these columns. Excluding
        the 'vector' column is the big win — it's 768 × float32 = ~3KB/row,
        so on a 100k-row store skipping it saves ~300MB allocation.
        Callers that only need to filter on tier/provenance/etc should
        pass an explicit columns list.
        """
        if columns is None:
            return self._table.to_pandas().to_dict(orient="records")
        # LanceDB pyarrow path: select columns then to_pandas
        try:
            arrow = self._table.search().select(columns).to_arrow()
            return arrow.to_pandas().to_dict(orient="records")
        except Exception:
            # Fallback: full scan and filter
            df = self._table.to_pandas()
            available = [c for c in columns if c in df.columns]
            return df[available].to_dict(orient="records") if available else []

    # Curated column subsets for callers that don't need the vector
    SCAN_LEAN_COLUMNS = [
        "id", "content", "tier", "provenance", "writer", "confidence",
        "category", "tags", "session_id", "source_ref",
        "created_at", "last_accessed", "last_seen_at", "promoted_at",
        "valid_from", "valid_to", "supersedes", "superseded_by",
        "contradiction_state", "conflict_with",
        "seen_count", "access_count", "importance",
        "embedding_model", "embedding_version", "archived",
    ]

    def scan_v2_lean(self) -> list[dict]:
        """scan_v2 without the heavy vector column."""
        return self.scan_v2(columns=self.SCAN_LEAN_COLUMNS)

    def list_tables(self) -> list[str]:
        """Return list of LanceDB table names in this database."""
        return list(self._db.table_names())

    def audit_scan(self) -> list[dict]:
        """Return all audit log entries. Returns empty list if audit table missing."""
        try:
            audit_table = self._db.open_table(AUDIT_TABLE_NAME)
            return audit_table.to_pandas().to_dict(orient="records")
        except (FileNotFoundError, ValueError):
            return []

    @property
    def db(self):
        """Expose the underlying LanceDB connection for migration scripts."""
        return self._db

    @property
    def table(self):
        """Expose the underlying LanceDB table for migration scripts."""
        return self._table

    @property
    def lock(self):
        """Expose the file lock for cross-module synchronized writes."""
        return self._lock
