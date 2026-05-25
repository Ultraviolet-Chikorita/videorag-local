from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from .models import Collection, QueryFilters, Segment, SegmentContext, SegmentLink, Video


_PHRASE_STOP_WORDS = {
    "all",
    "about",
    "actually",
    "after",
    "also",
    "and",
    "another",
    "any",
    "are",
    "but",
    "can",
    "called",
    "commonly",
    "each",
    "first",
    "for",
    "from",
    "get",
    "given",
    "give",
    "gives",
    "had",
    "has",
    "have",
    "here",
    "how",
    "into",
    "just",
    "kind",
    "last",
    "let",
    "like",
    "more",
    "once",
    "one",
    "other",
    "pretty",
    "same",
    "see",
    "should",
    "some",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "then",
    "tells",
    "this",
    "those",
    "what",
    "when",
    "where",
    "which",
    "video",
    "well",
    "way",
    "will",
    "with",
    "words",
    "would",
    "you",
    "your",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    external_id TEXT NOT NULL,
    title TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    sequence_index INTEGER,
    duration_ms INTEGER,
    tags_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    UNIQUE(collection_id, external_id)
);

CREATE TABLE IF NOT EXISTS segments (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    transcript TEXT NOT NULL,
    visual_caption TEXT NOT NULL DEFAULT '',
    frame_paths_json TEXT NOT NULL DEFAULT '[]',
    visual_embedding_json TEXT NOT NULL DEFAULT '[]',
    retrieval_text TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    UNIQUE(video_id, ordinal)
);

CREATE VIRTUAL TABLE IF NOT EXISTS segment_fts
USING fts5(segment_id UNINDEXED, content, tokenize='porter unicode61');

CREATE TABLE IF NOT EXISTS video_tags (
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY(video_id, tag)
);

CREATE TABLE IF NOT EXISTS segment_tags (
    segment_id TEXT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY(segment_id, tag)
);

CREATE TABLE IF NOT EXISTS video_facets (
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY(video_id, key, value)
);

CREATE TABLE IF NOT EXISTS segment_facets (
    segment_id TEXT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY(segment_id, key, value)
);

CREATE TABLE IF NOT EXISTS segment_links (
    source_segment_id TEXT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    target_segment_id TEXT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    relation_key TEXT NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY(source_segment_id, target_segment_id, relation_type, relation_key)
);

CREATE INDEX IF NOT EXISTS idx_segments_video_ordinal ON segments(video_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_segment_facets_lookup ON segment_facets(key, value);
CREATE INDEX IF NOT EXISTS idx_segment_links_source ON segment_links(source_segment_id);
"""

_CONTEXT_SELECT = """
SELECT
    c.id AS c_id, c.title AS c_title, c.metadata_json AS c_metadata,
    v.id AS v_id, v.collection_id AS v_collection_id, v.external_id AS v_external_id,
    v.title AS v_title, v.source_uri AS v_source_uri, v.sequence_index AS v_sequence_index,
    v.duration_ms AS v_duration_ms, v.tags_json AS v_tags, v.metadata_json AS v_metadata,
    s.id AS s_id, s.video_id AS s_video_id, s.ordinal AS s_ordinal,
    s.start_ms AS s_start_ms, s.end_ms AS s_end_ms, s.transcript AS s_transcript,
    s.visual_caption AS s_visual_caption, s.frame_paths_json AS s_frame_paths,
    s.visual_embedding_json AS s_visual_embedding,
    s.tags_json AS s_tags, s.metadata_json AS s_metadata
FROM segments s
JOIN videos v ON v.id = s.video_id
JOIN collections c ON c.id = v.collection_id
"""


class SQLiteIndex:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(_SCHEMA)
        self._migrate_segments()
        self.connection.commit()

    def __enter__(self) -> "SQLiteIndex":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def upsert_collection(self, collection: Collection) -> None:
        self.connection.execute(
            """
            INSERT INTO collections(id, title, metadata_json) VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET title=excluded.title, metadata_json=excluded.metadata_json
            """,
            (collection.id, collection.title, _json(collection.metadata)),
        )
        self.connection.commit()

    def upsert_video(self, video: Video) -> None:
        self.connection.execute(
            """
            INSERT INTO videos(
                id, collection_id, external_id, title, source_uri, sequence_index,
                duration_ms, tags_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                collection_id=excluded.collection_id,
                external_id=excluded.external_id,
                title=excluded.title,
                source_uri=excluded.source_uri,
                sequence_index=excluded.sequence_index,
                duration_ms=excluded.duration_ms,
                tags_json=excluded.tags_json,
                metadata_json=excluded.metadata_json
            """,
            (
                video.id,
                video.collection_id,
                video.external_id,
                video.title,
                video.source_uri,
                video.sequence_index,
                video.duration_ms,
                _json(video.tags),
                _json(video.metadata),
            ),
        )
        self.connection.execute("DELETE FROM video_tags WHERE video_id = ?", (video.id,))
        self.connection.execute("DELETE FROM video_facets WHERE video_id = ?", (video.id,))
        self.connection.executemany(
            "INSERT OR IGNORE INTO video_tags(video_id, tag) VALUES (?, ?)",
            [(video.id, _norm(tag)) for tag in video.tags if str(tag).strip()],
        )
        self.connection.executemany(
            "INSERT OR IGNORE INTO video_facets(video_id, key, value) VALUES (?, ?, ?)",
            [(video.id, key, value) for key, value in _facets(video.metadata)],
        )
        self.connection.commit()

    def upsert_segment(self, segment: Segment) -> None:
        if segment.end_ms <= segment.start_ms:
            raise ValueError(f"Segment {segment.id} has invalid timestamps.")
        self.connection.execute(
            """
            INSERT INTO segments(
                id, video_id, ordinal, start_ms, end_ms, transcript,
                visual_caption, frame_paths_json, visual_embedding_json,
                retrieval_text, tags_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                video_id=excluded.video_id,
                ordinal=excluded.ordinal,
                start_ms=excluded.start_ms,
                end_ms=excluded.end_ms,
                transcript=excluded.transcript,
                visual_caption=excluded.visual_caption,
                frame_paths_json=excluded.frame_paths_json,
                visual_embedding_json=excluded.visual_embedding_json,
                retrieval_text=excluded.retrieval_text,
                tags_json=excluded.tags_json,
                metadata_json=excluded.metadata_json
            """,
            (
                segment.id,
                segment.video_id,
                segment.ordinal,
                segment.start_ms,
                segment.end_ms,
                segment.transcript,
                segment.visual_caption,
                _json(segment.frame_paths),
                _json(segment.visual_embedding),
                segment.retrieval_text,
                _json(segment.tags),
                _json(segment.metadata),
            ),
        )
        self.connection.execute("DELETE FROM segment_fts WHERE segment_id = ?", (segment.id,))
        self.connection.execute(
            "INSERT INTO segment_fts(segment_id, content) VALUES (?, ?)",
            (segment.id, segment.retrieval_text),
        )
        self.connection.execute("DELETE FROM segment_tags WHERE segment_id = ?", (segment.id,))
        self.connection.execute("DELETE FROM segment_facets WHERE segment_id = ?", (segment.id,))
        self.connection.executemany(
            "INSERT OR IGNORE INTO segment_tags(segment_id, tag) VALUES (?, ?)",
            [(segment.id, _norm(tag)) for tag in segment.tags if str(tag).strip()],
        )
        self.connection.executemany(
            "INSERT OR IGNORE INTO segment_facets(segment_id, key, value) VALUES (?, ?, ?)",
            [(segment.id, key, value) for key, value in _facets(segment.metadata)],
        )
        self.connection.commit()

    def rebuild_entity_links(self, max_segments_per_entity: int = 32) -> int:
        self.connection.execute(
            "DELETE FROM segment_links WHERE relation_type IN ('shared_entity', 'shared_phrase')"
        )
        rows = self.connection.execute(
            """
            SELECT f.value AS entity, f.segment_id, s.video_id
            FROM segment_facets f
            JOIN segments s ON s.id = f.segment_id
            WHERE f.key = 'entity'
            ORDER BY f.value, s.video_id, s.ordinal
            """
        ).fetchall()
        groups: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
        for row in rows:
            groups[("shared_entity", row["entity"])].add((row["segment_id"], row["video_id"]))
        transcript_rows = self.connection.execute(
            "SELECT id AS segment_id, video_id, transcript, visual_caption FROM segments"
        ).fetchall()
        for row in transcript_rows:
            source_text = f"{row['transcript']} {row['visual_caption']}"
            for phrase in _phrase_candidates(source_text):
                groups[("shared_phrase", phrase)].add((row["segment_id"], row["video_id"]))

        inserted = 0
        for (relation_type, key), member_set in groups.items():
            members = sorted(member_set)
            if len(members) > max_segments_per_entity:
                continue
            weight = 0.8 if relation_type == "shared_entity" else 0.45
            for left, right in combinations(members, 2):
                if left[1] == right[1]:
                    continue
                self.connection.executemany(
                    """
                    INSERT OR REPLACE INTO segment_links(
                        source_segment_id, target_segment_id, relation_type, relation_key, weight
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (left[0], right[0], relation_type, key, weight),
                        (right[0], left[0], relation_type, key, weight),
                    ],
                )
                inserted += 1
        self.connection.commit()
        return inserted

    def search_lexical(
        self, fts_query: str, filters: QueryFilters, limit: int
    ) -> list[SegmentContext]:
        clauses, params = self._filter_clauses(filters)
        sql = (
            _CONTEXT_SELECT
            + " JOIN segment_fts ON segment_fts.segment_id = s.id"
            + " WHERE segment_fts MATCH ?"
            + clauses
            + " ORDER BY bm25(segment_fts) LIMIT ?"
        )
        rows = self.connection.execute(sql, [fts_query, *params, limit]).fetchall()
        return [_context(row) for row in rows]

    def search_entity_mentions(
        self, query: str, filters: QueryFilters, limit: int
    ) -> list[tuple[SegmentContext, str]]:
        clauses, params = self._filter_clauses(filters)
        normalized_query = _norm(query)
        sql = (
            _CONTEXT_SELECT
            + " JOIN segment_facets sf ON sf.segment_id = s.id"
            + " WHERE sf.key = 'entity' AND instr(?, sf.value) > 0"
            + clauses
            + " ORDER BY length(sf.value) DESC, s.ordinal LIMIT ?"
        )
        rows = self.connection.execute(sql, [normalized_query, *params, limit]).fetchall()
        return [(_context(row), self._entity_for_segment(row["s_id"], normalized_query)) for row in rows]

    def search_visual(
        self, query_embedding: tuple[float, ...], filters: QueryFilters, limit: int
    ) -> list[tuple[SegmentContext, float]]:
        if not query_embedding:
            return []
        clauses, params = self._filter_clauses(filters)
        sql = (
            _CONTEXT_SELECT
            + " WHERE s.visual_embedding_json != '[]'"
            + clauses
        )
        rows = self.connection.execute(sql, params).fetchall()
        matches = []
        for row in rows:
            vector = tuple(float(value) for value in json.loads(row["s_visual_embedding"]))
            similarity = _cosine_similarity(query_embedding, vector)
            matches.append((_context(row), similarity))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:limit]

    def linked_contexts(
        self, source_ids: Iterable[str], filters: QueryFilters
    ) -> list[tuple[SegmentContext, SegmentLink]]:
        source_ids = tuple(source_ids)
        if not source_ids:
            return []
        placeholders = ",".join("?" for _ in source_ids)
        clauses, params = self._filter_clauses(filters)
        linked_select = _CONTEXT_SELECT.replace(
            "\nFROM segments s",
            """,
    sl.source_segment_id AS link_source_segment_id,
    sl.target_segment_id AS link_target_segment_id,
    sl.relation_type AS link_relation_type,
    sl.relation_key AS link_relation_key,
    sl.weight AS link_weight
FROM segments s""",
            1,
        )
        sql = (
            linked_select
            + """
            JOIN (
                SELECT source_segment_id, target_segment_id, relation_type, relation_key, weight,
                       row_number() OVER (
                           PARTITION BY target_segment_id ORDER BY weight DESC, relation_key
                       ) AS preference
                FROM segment_links
                WHERE source_segment_id IN ({})
            ) sl ON sl.target_segment_id = s.id AND sl.preference = 1
            """.format(placeholders)
            + " WHERE 1 = 1"
            + clauses
            + " ORDER BY sl.weight DESC"
        )
        rows = self.connection.execute(sql, [*source_ids, *params]).fetchall()
        return [
            (
                _context(row),
                SegmentLink(
                    source_segment_id=row["link_source_segment_id"],
                    target_segment_id=row["link_target_segment_id"],
                    relation_type=row["link_relation_type"],
                    relation_key=row["link_relation_key"],
                    weight=row["link_weight"],
                ),
            )
            for row in rows
        ]

    def neighbor_contexts(
        self, context: SegmentContext, window: int, filters: QueryFilters
    ) -> list[tuple[SegmentContext, int]]:
        if window < 1:
            return []
        clauses, params = self._filter_clauses(filters)
        sql = (
            _CONTEXT_SELECT
            + " WHERE s.video_id = ? AND s.id != ? AND s.ordinal BETWEEN ? AND ?"
            + clauses
            + " ORDER BY abs(s.ordinal - ?), s.ordinal"
        )
        values = [
            context.video.id,
            context.segment.id,
            context.segment.ordinal - window,
            context.segment.ordinal + window,
            *params,
            context.segment.ordinal,
        ]
        rows = self.connection.execute(sql, values).fetchall()
        return [
            (_context(row), abs(row["s_ordinal"] - context.segment.ordinal)) for row in rows
        ]

    def get_video(self, video_id: str) -> Video | None:
        row = self.connection.execute(
            """
            SELECT
                id, collection_id, external_id, title, source_uri, sequence_index,
                duration_ms, tags_json, metadata_json
            FROM videos
            WHERE id = ?
            """,
            (video_id,),
        ).fetchone()
        if row is None:
            return None
        return Video(
            id=row["id"],
            collection_id=row["collection_id"],
            external_id=row["external_id"],
            title=row["title"],
            source_uri=row["source_uri"],
            sequence_index=row["sequence_index"],
            duration_ms=row["duration_ms"],
            tags=tuple(json.loads(row["tags_json"])),
            metadata=json.loads(row["metadata_json"]),
        )

    def stats(self) -> dict[str, int]:
        result = {}
        for label, table in (
            ("collections", "collections"),
            ("videos", "videos"),
            ("segments", "segments"),
            ("links", "segment_links"),
        ):
            result[label] = self.connection.execute(
                f"SELECT count(*) FROM {table}"
            ).fetchone()[0]
        result["visual_segments"] = self.connection.execute(
            """
            SELECT count(*) FROM segments
            WHERE visual_caption != '' OR visual_embedding_json != '[]'
            """
        ).fetchone()[0]
        return result

    def _filter_clauses(self, filters: QueryFilters) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if filters.collection_id:
            clauses.append(" AND v.collection_id = ?")
            params.append(filters.collection_id)
        if filters.video_ids:
            placeholders = ",".join("?" for _ in filters.video_ids)
            clauses.append(f" AND v.id IN ({placeholders})")
            params.extend(filters.video_ids)
        for tag in filters.tags:
            clauses.append(
                """
                AND (
                    EXISTS (SELECT 1 FROM segment_tags st WHERE st.segment_id = s.id AND st.tag = ?)
                    OR EXISTS (SELECT 1 FROM video_tags vt WHERE vt.video_id = v.id AND vt.tag = ?)
                )
                """
            )
            normalized = _norm(tag)
            params.extend([normalized, normalized])
        for key, value in filters.facets.items():
            clauses.append(
                """
                AND (
                    EXISTS (
                        SELECT 1 FROM segment_facets sf2
                        WHERE sf2.segment_id = s.id AND sf2.key = ? AND sf2.value = ?
                    )
                    OR EXISTS (
                        SELECT 1 FROM video_facets vf
                        WHERE vf.video_id = v.id AND vf.key = ? AND vf.value = ?
                    )
                )
                """
            )
            normalized_key = _facet_key(key)
            normalized_value = _norm(value)
            params.extend([normalized_key, normalized_value, normalized_key, normalized_value])
        return "".join(clauses), params

    def _entity_for_segment(self, segment_id: str, query: str) -> str:
        rows = self.connection.execute(
            "SELECT value FROM segment_facets WHERE segment_id = ? AND key = 'entity'",
            (segment_id,),
        ).fetchall()
        matches = [row["value"] for row in rows if row["value"] in query]
        return max(matches, key=len, default="entity")

    def _migrate_segments(self) -> None:
        existing = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(segments)").fetchall()
        }
        for column, definition in (
            ("visual_caption", "TEXT NOT NULL DEFAULT ''"),
            ("frame_paths_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("visual_embedding_json", "TEXT NOT NULL DEFAULT '[]'"),
        ):
            if column not in existing:
                self.connection.execute(f"ALTER TABLE segments ADD COLUMN {column} {definition}")


def _context(row: sqlite3.Row) -> SegmentContext:
    return SegmentContext(
        collection=Collection(
            id=row["c_id"],
            title=row["c_title"],
            metadata=json.loads(row["c_metadata"]),
        ),
        video=Video(
            id=row["v_id"],
            collection_id=row["v_collection_id"],
            external_id=row["v_external_id"],
            title=row["v_title"],
            source_uri=row["v_source_uri"],
            sequence_index=row["v_sequence_index"],
            duration_ms=row["v_duration_ms"],
            tags=tuple(json.loads(row["v_tags"])),
            metadata=json.loads(row["v_metadata"]),
        ),
        segment=Segment(
            id=row["s_id"],
            video_id=row["s_video_id"],
            ordinal=row["s_ordinal"],
            start_ms=row["s_start_ms"],
            end_ms=row["s_end_ms"],
            transcript=row["s_transcript"],
            visual_caption=row["s_visual_caption"],
            frame_paths=tuple(json.loads(row["s_frame_paths"])),
            visual_embedding=tuple(json.loads(row["s_visual_embedding"])),
            tags=tuple(json.loads(row["s_tags"])),
            metadata=json.loads(row["s_metadata"]),
        ),
    )


def _facets(metadata: dict[str, Any]) -> list[tuple[str, str]]:
    pairs = []
    for key, value in metadata.items():
        values = value if isinstance(value, (list, tuple, set)) else [value]
        for item in values:
            if isinstance(item, (str, int, float, bool)) and str(item).strip():
                pairs.append((_facet_key(key), _norm(item)))
    return pairs


def _facet_key(key: Any) -> str:
    normalized = _norm(key)
    return "entity" if normalized in {"entity", "entities"} else normalized


def _norm(value: Any) -> str:
    return str(value).strip().casefold()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def _phrase_candidates(transcript: str) -> set[str]:
    words = re.findall(r"[a-z][a-z-]+", transcript.casefold())
    phrases = set()
    for size in (2, 3):
        for index in range(len(words) - size + 1):
            candidate = words[index : index + size]
            if all(len(word) >= 3 and word not in _PHRASE_STOP_WORDS for word in candidate):
                phrases.add(" ".join(candidate))
    return phrases
