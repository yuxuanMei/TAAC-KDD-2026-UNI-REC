"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""
# https://huggingface.co/datasets/TAAC2026/data_sample_1000

import os
import logging
import random
import json
import gc
import datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # Ordered list of (feature_id, offset, length).
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # Quick lookup from fid to its (offset, length).
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy('file_system')


def _format_wall_time(ts: int, offset_hours: float = 0.0) -> str:
    offset_seconds = int(round(offset_hours * 3600.0))
    wall_ts = int(ts) + offset_seconds
    dt = datetime.datetime.utcfromtimestamp(wall_ts)
    if abs(offset_hours) < 1e-9:
        suffix = "UTC"
    else:
        suffix = f"UTC{offset_hours:+g}"
    return f"{dt:%Y-%m-%d %H:%M:%S} {suffix}"


def _summarize_wall_counts(
    timestamps: np.ndarray,
    offset_hours: float,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    if timestamps.size == 0:
        return [], []
    wall = timestamps.astype(np.float64) + offset_hours * 3600.0
    day_index = np.floor(wall / 86400.0).astype(np.int64)
    hour = (np.floor(np.remainder(wall, 86400.0) / 3600.0)
            .astype(np.int64))

    day_counts: List[Tuple[str, int]] = []
    for day, cnt in zip(*np.unique(day_index, return_counts=True)):
        label = datetime.datetime.utcfromtimestamp(int(day) * 86400).strftime(
            "%Y-%m-%d")
        day_counts.append((label, int(cnt)))

    hour_counts = [
        (f"{int(h):02d}:00", int(cnt))
        for h, cnt in zip(*np.unique(hour, return_counts=True))
    ]
    return day_counts, hour_counts


def _log_timestamp_block(
    name: str,
    timestamps: np.ndarray,
    offset_hours: float,
) -> None:
    if timestamps.size == 0:
        logging.info(f"[SplitTimeProbe] {name}: empty")
        return
    ts_min = int(timestamps.min())
    ts_max = int(timestamps.max())
    logging.info(
        f"[SplitTimeProbe] {name}: rows={timestamps.size}, "
        f"UTC=[{_format_wall_time(ts_min, 0.0)} -> "
        f"{_format_wall_time(ts_max, 0.0)}], "
        f"wall=[{_format_wall_time(ts_min, offset_hours)} -> "
        f"{_format_wall_time(ts_max, offset_hours)}]"
    )


def _log_split_time_probe(
    rg_info: List[Tuple[str, int, int]],
    n_train_rgs: int,
    offset_hours: float,
) -> None:
    """Log exact timestamp ranges for the current RowGroup train/valid split."""
    try:
        logging.info("[SplitTimeProbe] start: reading timestamp column only")
        pf_cache: Dict[str, pq.ParquetFile] = {}
        per_rg: List[Tuple[int, int, int, int]] = []
        rg_parts: List[np.ndarray] = []

        for global_idx, (file_path, rg_idx, _) in enumerate(rg_info):
            pf = pf_cache.get(file_path)
            if pf is None:
                pf = pq.ParquetFile(file_path)
                pf_cache[file_path] = pf
            table = pf.read_row_group(rg_idx, columns=['timestamp'])
            arr = table.column('timestamp').to_numpy(
                zero_copy_only=False).astype(np.int64)
            rg_parts.append(arr)
            if arr.size == 0:
                per_rg.append((global_idx, 0, 0, 0))
                continue
            per_rg.append((global_idx, int(arr.size), int(arr.min()),
                           int(arr.max())))

        non_empty_parts = [arr for arr in rg_parts if arr.size > 0]
        if not non_empty_parts:
            logging.warning("[SplitTimeProbe] no timestamps found")
            return

        all_ts = np.concatenate(non_empty_parts)
        train_parts = [arr for arr in rg_parts[:n_train_rgs] if arr.size > 0]
        valid_parts = [arr for arr in rg_parts[n_train_rgs:] if arr.size > 0]
        train_ts = np.concatenate(train_parts) if train_parts else np.array(
            [], dtype=np.int64)
        valid_ts = np.concatenate(valid_parts) if valid_parts else np.array(
            [], dtype=np.int64)

        _log_timestamp_block("all", all_ts, offset_hours)
        _log_timestamp_block("train", train_ts, offset_hours)
        _log_timestamp_block("valid", valid_ts, offset_hours)

        if train_ts.size and valid_ts.size:
            gap = int(valid_ts.min()) - int(train_ts.max())
            logging.info(
                "[SplitTimeProbe] train_max_to_valid_min_gap_seconds="
                f"{gap}")

        day_counts, hour_counts = _summarize_wall_counts(
            valid_ts, offset_hours)
        day_msg = ", ".join(f"{k}={v}" for k, v in day_counts)
        hour_msg = ", ".join(f"{k}={v}" for k, v in hour_counts)
        logging.info(f"[SplitTimeProbe] valid_wall_date_counts: {day_msg}")
        logging.info(f"[SplitTimeProbe] valid_wall_hour_counts: {hour_msg}")

        valid_rg = per_rg[n_train_rgs:]
        if valid_rg:
            bins = min(5, len(valid_rg))
            bin_size = max(1, int(np.ceil(len(valid_rg) / bins)))
            for start in range(0, len(valid_rg), bin_size):
                block = valid_rg[start:start + bin_size]
                rows = sum(x[1] for x in block)
                non_empty = [x for x in block if x[1] > 0]
                if not non_empty:
                    continue
                rg_start = block[0][0]
                rg_end = block[-1][0]
                ts_min = min(x[2] for x in non_empty)
                ts_max = max(x[3] for x in non_empty)
                logging.info(
                    "[SplitTimeProbe] valid_rg_block "
                    f"{rg_start}-{rg_end}: rows={rows}, "
                    f"wall=[{_format_wall_time(ts_min, offset_hours)} -> "
                    f"{_format_wall_time(ts_max, offset_hours)}]"
                )
    except Exception as exc:
        logging.warning(f"[SplitTimeProbe] failed: {exc}")

# Time-delta bucket boundaries (64 edges -> 65 buckets: 0=padding, 1..64).
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# Total number of time-bucket embedding slots (= number of boundaries + 1, with
# padding=0 included).
#
# This constant is uniquely determined by the length of BUCKET_BOUNDARIES; on
# the model side, ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` must match
# this value exactly, otherwise an IndexError may be raised at runtime.
#
# That is why ``train.py`` / ``infer.py`` only expose the boolean flag
# ``--use_time_buckets`` and derive the concrete bucket count from here.
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1


DEFAULT_TIME_ZONE_OFFSET_HOURS = 8.0
DEFAULT_TIME_ZONE_OFFSET_SECONDS = DEFAULT_TIME_ZONE_OFFSET_HOURS * 3600.0
OOV_UNK_MAP_FILENAME = "oov_unk_maps.npz"
DEFAULT_OOV_MAX_VOCAB_SIZE = 1_000_000


def _oov_unk_key(kind: str, fid: int, domain: Optional[str] = None) -> str:
    if kind == "seq":
        return f"seq_{domain}_f{int(fid)}"
    return f"{kind}_f{int(fid)}"


def _update_oov_counts(
    counts: "npt.NDArray[np.uint8]",
    values: "npt.NDArray[np.int64]",
    vocab_size: int,
    min_count: int,
) -> None:
    if values.size == 0:
        return
    vals = values.astype(np.int64, copy=False)
    vals = vals[(vals > 0) & (vals < int(vocab_size))]
    if vals.size == 0:
        return
    uniq, cnt = np.unique(vals, return_counts=True)
    current = counts[uniq].astype(np.int16, copy=False)
    updated = current + np.minimum(cnt, int(min_count)).astype(np.int16)
    counts[uniq] = np.minimum(updated, int(min_count)).astype(np.uint8)


def build_oov_unk_maps(
    rg_info: List[Tuple[str, int, int]],
    schema_path: str,
    seq_max_lens: Optional[Dict[str, int]] = None,
    min_count: int = 2,
    max_vocab_size: int = DEFAULT_OOV_MAX_VOCAB_SIZE,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Build frequent-ID masks from the training RowGroups.

    Each tracked sparse feature gets a boolean mask of shape ``[vocab_size]``.
    IDs with count >= ``min_count`` stay as-is; IDs absent or below the
    threshold are mapped to the dedicated UNK row ``vocab_size`` at dataset
    conversion time. Very large vocabularies can be skipped because the model
    may also skip their embeddings via ``emb_skip_threshold``.
    """
    del seq_max_lens  # Counts use all stored list values; model truncation is unchanged.
    min_count = max(1, int(min_count))
    max_vocab_size = int(max_vocab_size) if max_vocab_size else 0
    with open(schema_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    plans: List[Tuple[str, str, int]] = []

    def _maybe_add(kind: str, fid: int, vocab_size: int, col_name: str,
                   domain: Optional[str] = None) -> None:
        vs = int(vocab_size)
        if vs <= 1:
            return
        if max_vocab_size > 0 and vs > max_vocab_size:
            return
        plans.append((_oov_unk_key(kind, fid, domain), col_name, vs))

    for fid, vs, _dim in raw.get('user_int', []):
        _maybe_add('user_int', fid, vs, f'user_int_feats_{fid}')
    for fid, vs, _dim in raw.get('item_int', []):
        _maybe_add('item_int', fid, vs, f'item_int_feats_{fid}')
    for domain, cfg in sorted(raw.get('seq', {}).items()):
        prefix = cfg['prefix']
        ts_fid = cfg.get('ts_fid')
        for fid, vs in cfg.get('features', []):
            if fid == ts_fid:
                continue
            _maybe_add('seq', fid, vs, f'{prefix}_{fid}', domain)

    if not plans:
        logging.info("[OOV_UNK] no sparse features selected for UNK mapping")
        return {}, {
            "min_count": min_count,
            "max_vocab_size": max_vocab_size,
            "features": {},
        }

    counts: Dict[str, np.ndarray] = {
        key: np.zeros(vs, dtype=np.uint8)
        for key, _col_name, vs in plans
    }
    vocab_by_key = {key: vs for key, _col_name, vs in plans}
    needed_cols = sorted({col_name for _key, col_name, _vs in plans})
    pf_cache: Dict[str, pq.ParquetFile] = {}
    scanned_rows = 0

    logging.info(
        f"[OOV_UNK] scanning {len(rg_info)} train RowGroups, "
        f"features={len(plans)}, min_count={min_count}, "
        f"max_vocab_size={max_vocab_size or 'none'}")
    for rg_pos, (file_path, rg_idx, row_count) in enumerate(rg_info, start=1):
        pf = pf_cache.get(file_path)
        if pf is None:
            pf = pq.ParquetFile(file_path)
            pf_cache[file_path] = pf
        for batch in pf.iter_batches(
            batch_size=65536,
            row_groups=[rg_idx],
            columns=needed_cols,
        ):
            local_idx = {name: i for i, name in enumerate(batch.schema.names)}
            for key, col_name, vs in plans:
                ci = local_idx.get(col_name)
                if ci is None:
                    continue
                col = batch.column(ci)
                if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
                    values = col.values.to_numpy(zero_copy_only=False)
                else:
                    values = col.fill_null(0).to_numpy(
                        zero_copy_only=False)
                _update_oov_counts(counts[key], values, vs, min_count)
        scanned_rows += int(row_count)
        if rg_pos % 100 == 0:
            logging.info(
                f"[OOV_UNK] scanned {rg_pos}/{len(rg_info)} RowGroups, "
                f"rows={scanned_rows}")

    maps = {
        key: (arr >= min_count)
        for key, arr in counts.items()
    }
    feature_meta = {
        key: {
            "vocab_size": int(vocab_by_key[key]),
            "kept_ids": int(mask.sum()),
            "unk_ids": int(vocab_by_key[key] - int(mask.sum())),
        }
        for key, mask in maps.items()
    }
    logging.info(
        "[OOV_UNK] built maps: "
        f"features={len(maps)}, rows={scanned_rows}, "
        f"kept_ids={sum(x['kept_ids'] for x in feature_meta.values())}, "
        f"unk_ids={sum(x['unk_ids'] for x in feature_meta.values())}")
    meta = {
        "min_count": min_count,
        "max_vocab_size": max_vocab_size,
        "scanned_rows": scanned_rows,
        "features": feature_meta,
    }
    return maps, meta


def save_oov_unk_maps(
    maps: Dict[str, np.ndarray],
    path: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays: Dict[str, Any] = {
        key: value.astype(np.bool_, copy=False)
        for key, value in maps.items()
    }
    arrays["__meta__"] = np.array(json.dumps(meta or {}, ensure_ascii=False))
    np.savez_compressed(path, **arrays)
    logging.info(f"[OOV_UNK] saved maps to {path}")


def load_oov_unk_maps(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        maps = {
            key: data[key].astype(np.bool_, copy=False)
            for key in data.files
            if key != "__meta__"
        }
        meta: Dict[str, Any] = {}
        if "__meta__" in data.files:
            raw_meta = str(data["__meta__"].item())
            if raw_meta:
                meta = json.loads(raw_meta)
    logging.info(f"[OOV_UNK] loaded {len(maps)} maps from {path}")
    return maps, meta


# History-match item dense features are deterministic statistics derived from
# item fields and A/B/C/D behavior sequences. They occupy the otherwise-empty
# item_dense side only when --history_match_dense is enabled.
HISTORY_MATCH_DENSE_FID = 10001
HISTORY_MATCH_DENSE_DIM = 64
ITEM_DENSE_BASE_FIDS = [5, 6, 7, 8, 9, 10, 12, 13, 16, 81, 83, 84, 85]
HISTORY_MATCH_PAIR_FIDS = [
    # (target item fid, sequence domain key, sequence fid).  These curated
    # pairs avoid cross-slot "feature soup" matches such as comparing an item
    # category id with a sequence action id that merely shares the same value.
    (5, 'a', 42),
    (5, 'b', 70),
    (5, 'b', 78),
    (5, 'c', 30),
    (5, 'd', 21),
    (6, 'b', 71),
    (6, 'b', 78),
    (83, 'b', 68),
    (83, 'b', 75),
    (83, 'd', 21),
    (84, 'b', 70),
    (84, 'd', 21),
    (85, 'b', 78),
    (85, 'b', 79),
]
HISTORY_MATCH_DETAIL_OFFSET = 22
HISTORY_MATCH_AGG_OFFSET = HISTORY_MATCH_DETAIL_OFFSET + len(HISTORY_MATCH_PAIR_FIDS) * 2
HISTORY_MATCH_GLOBAL_OFFSET = HISTORY_MATCH_AGG_OFFSET + 4 * 2
SEQ_ACTION_FIDS = {
    'a': [40, 41, 46],
    'b': [68, 75, 77],
    'c': [28, 32, 33],
    'd': [17, 24, 25],
}

# Rare16 profile features are a small, deterministic user_dense extension focused
# on the observed online weak spot: item_int fid16 being train-rare/unseen
# while fid5/fid6 still describe a meaningful item group.  This keeps the
# single shared UNK embedding behavior intact and only gives the final model a
# light dense side channel. It is appended to the already-existing user_dense
# token to avoid adding a new HyFormer token. Keep it deliberately small: the
# previous bucketized version over-moved high-score OOV rows online.
RARE16_PROFILE_DENSE_FID = 10002
RARE16_PROFILE_DENSE_DIM = 24
RARE16_PROFILE_RARE_FIDS = [5, 6, 7, 8, 10, 12, 16, 84, 85]
RARE16_PROFILE_HOUR_BINS = [
    (7 * 60 + 40, 8 * 60),       # 07:40-08:00
    (8 * 60, 8 * 60 + 30),       # 08:00-08:30
    (8 * 60 + 30, 9 * 60),       # 08:30-09:00
    (9 * 60, 9 * 60 + 15),       # 09:00-09:15
]


def _domain_key(domain: str) -> str:
    d = domain.lower()
    for key in ('a', 'b', 'c', 'd'):
        if d.endswith(key) or f'_{key}' in d:
            return key
    return d


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        seq_time_feature_dim: int = 4,
        history_match_dense: bool = False,
        rare16_profile: bool = False,
        time_zone_offset_hours: float = DEFAULT_TIME_ZONE_OFFSET_HOURS,
        oov_unk_maps: Optional[Dict[str, np.ndarray]] = None,
        oov_unk_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
            seq_time_feature_dim: 4 keeps the legacy cyclic time features;
                7 adds compact recency/context features; 12 keeps the
                previous richer per-step time features.
            history_match_dense: if True, build deterministic item-history
                overlap statistics into the item_dense side.
            rare16_profile: if True, append lightweight Rare16/OOV profile
                features to the existing user_dense token.
            time_zone_offset_hours: wall-clock offset used for hour/weekday
                features. 8.0 maps UTC timestamps to Beijing time.
            oov_unk_maps: optional frequent-ID masks built from training data.
                Values absent from a mask are mapped to the feature's UNK row
                ``vocab_size`` instead of padding 0.
        """
        super().__init__()

        # Accept either a directory or a single file path.
        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        _time_dim = int(seq_time_feature_dim)
        self.seq_time_feature_dim = _time_dim if _time_dim in (4, 7, 12) else 4
        self.history_match_dense = bool(history_match_dense)
        self.rare16_profile = bool(rare16_profile)
        self.time_zone_offset_hours = float(time_zone_offset_hours)
        self.time_zone_offset_seconds = self.time_zone_offset_hours * 3600.0
        self.oov_unk_maps = oov_unk_maps or {}
        self.oov_unk_meta = oov_unk_meta or {}
        self.use_oov_unk = bool(self.oov_unk_maps)
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build the list of Row Groups.
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_item_dense = np.zeros((B, self.item_dense_schema.total_dim), dtype=np.float32)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_session = {}
        self._buf_seq_cross_day = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_session[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_cross_day[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self._user_int_plan = []  # [(col_idx, fid, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, fid, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, fid, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        self._seq_slot_by_fid = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            self._seq_slot_by_fid[domain] = {
                fid: slot for slot, fid in enumerate(sideinfo_fids)
            }
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, fid, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}, "
            f"seq_time_feature_dim={self.seq_time_feature_dim}, "
            f"history_match_dense={self.history_match_dense}, "
            f"rare16_profile={self.rare16_profile}, "
            f"time_zone_offset_hours={self.time_zone_offset_hours}, "
            f"use_oov_unk={self.use_oov_unk}, "
            f"user_dense_dim={self.user_dense_schema.total_dim}, "
            f"rare16_profile_offset={self.rare16_profile_offset}, "
            f"item_dense_dim={self.item_dense_schema.total_dim}, "
            f"rare16_profile_dim={RARE16_PROFILE_DENSE_DIM}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        self.user_int_vocab_by_fid: Dict[int, int] = {}
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)
            self.user_int_vocab_by_fid[int(fid)] = int(vs)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        self.item_int_vocab_by_fid: Dict[int, int] = {}
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)
            self.item_int_vocab_by_fid[int(fid)] = int(vs)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)
        self.rare16_profile_offset = self.user_dense_schema.total_dim
        if self.rare16_profile:
            self.user_dense_schema.add(RARE16_PROFILE_DENSE_FID, RARE16_PROFILE_DENSE_DIM)

        # ---- item_dense (optional deterministic history-match statistics) ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()
        if self.history_match_dense:
            self.item_dense_schema.add(HISTORY_MATCH_DENSE_FID, HISTORY_MATCH_DENSE_DIM)

        # ---- sequence domains ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        merged_lists: Dict[str, List[Any]] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            elif (
                isinstance(buffer[0][k], list)
                and k == 'user_id'
                and len(buffer[0][k]) == buffer[0]['label'].shape[0]
            ):
                merged_lists[k] = [
                    item
                    for b in buffer
                    for item in b[k]
                ]
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        rand_idx_list = rand_idx.tolist()
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            for k, values in merged_lists.items():
                batch[k] = [values[j] for j in rand_idx_list[i:end]]
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = vocab_size if self.use_oov_unk else 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def _apply_oov_unk(
        self,
        key: str,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Map train-rare or train-unseen IDs to the feature-specific UNK row."""
        if not self.use_oov_unk or vocab_size <= 0:
            return
        frequent = self.oov_unk_maps.get(key)
        if frequent is None:
            return
        if frequent.shape[0] != int(vocab_size):
            logging.warning(
                f"[OOV_UNK] map shape mismatch for {key}: "
                f"mask={frequent.shape[0]}, vocab={vocab_size}; skipping")
            return
        valid_mask = (arr > 0) & (arr < vocab_size)
        if not valid_mask.any():
            return
        vals = arr[valid_mask]
        rare_mask = ~frequent[vals]
        if rare_mask.any():
            vals[rare_mask] = int(vocab_size)
            arr[valid_mask] = vals

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 (padding). Note: the raw data contains -1
        (missing); currently treated the same way as 0 (padding).

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start:start + use_len]
            lengths[i] = use_len

        padded[padded <= 0] = 0
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start:start + use_len]

        return padded

    def _item_scalar(self, item_int: "npt.NDArray[np.int64]", fid: int) -> "npt.NDArray[np.int64]":
        try:
            offset, length = self.item_int_schema.get_offset_length(fid)
        except KeyError:
            return np.zeros(item_int.shape[0], dtype=np.int64)
        if length <= 0:
            return np.zeros(item_int.shape[0], dtype=np.int64)
        return item_int[:, offset]

    def _feature_any_rare_from_mapped(
        self,
        values: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> "npt.NDArray[np.bool_]":
        """Return rows mapped to the feature-specific UNK row.

        OOV/rare values are mapped to ``vocab_size`` by ``_apply_oov_unk``.
        For multi-value features, a row is rare when any non-padding position
        hits that UNK row.
        """
        if not self.use_oov_unk or int(vocab_size) <= 0:
            return np.zeros(values.shape[0], dtype=np.bool_)
        rare = values >= int(vocab_size)
        if rare.ndim == 1:
            return rare
        return rare.any(axis=1)

    def _item_any_rare(
        self,
        item_int: "npt.NDArray[np.int64]",
        fid: int,
    ) -> "npt.NDArray[np.bool_]":
        try:
            offset, length = self.item_int_schema.get_offset_length(fid)
        except KeyError:
            return np.zeros(item_int.shape[0], dtype=np.bool_)
        vs = int(self.item_int_vocab_by_fid.get(int(fid), 0))
        return self._feature_any_rare_from_mapped(
            item_int[:, offset:offset + length], vs)

    def _user_any_rare(
        self,
        user_int: "npt.NDArray[np.int64]",
        fid: int,
    ) -> "npt.NDArray[np.bool_]":
        try:
            offset, length = self.user_int_schema.get_offset_length(fid)
        except KeyError:
            return np.zeros(user_int.shape[0], dtype=np.bool_)
        vs = int(self.user_int_vocab_by_fid.get(int(fid), 0))
        return self._feature_any_rare_from_mapped(
            user_int[:, offset:offset + length], vs)

    def _fill_rare16_profile_dense(
        self,
        user_dense: "npt.NDArray[np.float32]",
        item_int: "npt.NDArray[np.int64]",
        user_int: "npt.NDArray[np.int64]",
        timestamps: "npt.NDArray[np.int64]",
    ) -> None:
        """Fill the lightweight Rare16 Group Profile block.

        The block intentionally uses only point-wise metadata and the OOV maps
        built from training data, so train/inference semantics stay aligned.
        """
        if user_dense is None or RARE16_PROFILE_DENSE_DIM <= 0:
            return
        base = int(self.rare16_profile_offset)
        end = base + RARE16_PROFILE_DENSE_DIM
        if user_dense.shape[1] < end:
            return

        B = item_int.shape[0]
        block = user_dense[:, base:end]
        block[:] = 0.0

        rare_by_fid = {
            fid: self._item_any_rare(item_int, fid)
            for fid in RARE16_PROFILE_RARE_FIDS
        }
        item_rare_count = np.zeros(B, dtype=np.float32)
        for mask in rare_by_fid.values():
            item_rare_count += mask.astype(np.float32)

        user_rare_count = np.zeros(B, dtype=np.float32)
        if self.use_oov_unk:
            for fid, _offset, _length in self.user_int_schema.entries:
                user_rare_count += self._user_any_rare(user_int, fid).astype(np.float32)

        rare16 = rare_by_fid.get(16, np.zeros(B, dtype=np.bool_))
        item83 = self._item_scalar(item_int, 83)
        item84 = self._item_scalar(item_int, 84)
        item85 = self._item_scalar(item_int, 85)
        tag_zero_83 = item83 == 0
        tag_zero_84 = item84 == 0
        tag_zero_85 = item85 == 0
        tag_all_zero = tag_zero_83 & tag_zero_84 & tag_zero_85

        block[:, 0] = np.minimum(item_rare_count, 6.0) / 6.0
        block[:, 1] = np.minimum(user_rare_count, 6.0) / 6.0
        block[:, 2] = rare16.astype(np.float32)
        for j, fid in enumerate([5, 6, 7, 8, 10, 12, 84, 85], start=3):
            block[:, j] = rare_by_fid.get(fid, np.zeros(B, dtype=np.bool_)).astype(np.float32)

        block[:, 11] = tag_zero_83.astype(np.float32)
        block[:, 12] = tag_zero_84.astype(np.float32)
        block[:, 13] = tag_zero_85.astype(np.float32)
        block[:, 14] = tag_all_zero.astype(np.float32)
        block[:, 15] = (rare16 & tag_all_zero).astype(np.float32)

        wall = timestamps.astype(np.float64) + self.time_zone_offset_seconds
        minute = np.remainder(wall, 86400.0) / 60.0
        hour_flags = []
        for lo, hi in RARE16_PROFILE_HOUR_BINS:
            flag = (minute >= float(lo)) & (minute < float(hi))
            hour_flags.append(flag)
        for j, flag in enumerate(hour_flags):
            block[:, 16 + j] = flag.astype(np.float32)
            block[:, 20 + j] = (rare16 & flag).astype(np.float32)

    def _fill_history_match_dense_base(
        self,
        item_dense: "npt.NDArray[np.float32]",
        item_int: "npt.NDArray[np.int64]",
    ) -> None:
        # 0..12: compact item-side values. These give the overlap statistics a
        # coarse item profile without adding new sparse tables.
        for j, fid in enumerate(ITEM_DENSE_BASE_FIDS):
            vals = self._item_scalar(item_int, fid).astype(np.float32)
            item_dense[:, j] = np.log1p(np.maximum(vals, 0.0)) / 12.0

        # 13: density of the multi-value item tag list.
        try:
            offset, length = self.item_int_schema.get_offset_length(11)
            tags = item_int[:, offset:offset + length]
            item_dense[:, 13] = (tags > 0).sum(axis=1).astype(np.float32) / max(length, 1)
        except KeyError:
            pass

    def _write_domain_history_match_dense(
        self,
        item_dense: "npt.NDArray[np.float32]",
        item_int: "npt.NDArray[np.int64]",
        domain: str,
        seq_values: "npt.NDArray[np.int64]",
        seq_lens: "npt.NDArray[np.int64]",
        max_len: int,
        global_count: "npt.NDArray[np.float32]",
        global_recency: "npt.NDArray[np.float32]",
    ) -> None:
        key = _domain_key(domain)
        domain_pos = {'a': 0, 'b': 1, 'c': 2, 'd': 3}.get(key)
        if domain_pos is None:
            return

        # 14..17: per-domain sequence length strength.
        item_dense[:, 14 + domain_pos] = (
            np.log1p(seq_lens.astype(np.float32)) / np.log1p(float(max_len + 1))
        )

        slot_by_fid = self._seq_slot_by_fid.get(domain, {})
        action_slots = [
            slot_by_fid[fid] for fid in SEQ_ACTION_FIDS.get(key, [])
            if fid in slot_by_fid
        ]
        if action_slots:
            actions = seq_values[:, action_slots, :]
            denom = np.maximum(seq_lens.astype(np.float32) * len(action_slots), 1.0)
            item_dense[:, 18 + domain_pos] = (actions > 0).sum(axis=(1, 2)) / denom

        domain_hit_total = np.zeros(item_dense.shape[0], dtype=np.float32)
        domain_recent_hits = np.zeros(item_dense.shape[0], dtype=np.float32)
        recent_window = min(20, max_len)
        pair_count = 0

        for pair_pos, (item_fid, pair_domain, seq_fid) in enumerate(HISTORY_MATCH_PAIR_FIDS):
            if pair_domain != key:
                continue
            seq_slot = slot_by_fid.get(seq_fid)
            if seq_slot is None:
                continue
            pair_count += 1
            out_offset = HISTORY_MATCH_DETAIL_OFFSET + pair_pos * 2
            target = self._item_scalar(item_int, item_fid)
            valid_target = target > 0
            if not valid_target.any():
                continue

            seq_attr_values = seq_values[:, seq_slot, :]
            hit_by_pos = (
                seq_attr_values == target.reshape(-1, 1)
            ) & valid_target.reshape(-1, 1)
            counts = hit_by_pos.sum(axis=1).astype(np.float32)
            item_dense[:, out_offset] = np.log1p(counts) / np.log1p(float(max_len + 1))

            has_hit = counts > 0
            first_pos = np.argmax(hit_by_pos, axis=1)
            recency = np.zeros_like(counts, dtype=np.float32)
            recency[has_hit] = 1.0 / (1.0 + first_pos[has_hit].astype(np.float32))
            item_dense[:, out_offset + 1] = recency

            domain_hit_total += counts
            if recent_window > 0:
                domain_recent_hits += hit_by_pos[:, :recent_window].sum(axis=1).astype(np.float32)
            global_count += counts
            global_recency[:] = np.maximum(global_recency, recency)

        # Per-domain aggregate overlap strength. Placed after all curated
        # detail pairs so it cannot overwrite pair-level features.
        agg_offset = HISTORY_MATCH_AGG_OFFSET + domain_pos * 2
        item_dense[:, agg_offset] = np.log1p(domain_hit_total) / np.log1p(float(max_len + 1))
        if recent_window > 0 and pair_count > 0:
            item_dense[:, agg_offset + 1] = domain_recent_hits / float(recent_window * pair_count)

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""
        B = batch.num_rows

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()

        # ---- user_int: write into pre-allocated buffer ----
        # Note: null -> 0 (via fill_null), -1 -> 0 (via arr<=0); missing values
        # are treated the same as padding. Features with vs==0 have no vocab
        # information and are forced to 0 on the dataset side so that the
        # model's 1-slot Embedding (created for vs=0) is never indexed out of
        # range.
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, fid, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                    self._apply_oov_unk(_oov_unk_key('user_int', fid), arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('user_int', ci, padded, vs)
                    self._apply_oov_unk(_oov_unk_key('user_int', fid), padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, fid, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                    self._apply_oov_unk(_oov_unk_key('item_int', fid), arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('item_int', ci, padded, vs)
                    self._apply_oov_unk(_oov_unk_key('item_int', fid), padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded

        # ---- user_dense ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset:offset + dim] = padded
        if self.rare16_profile:
            self._fill_rare16_profile_dense(
                user_dense=user_dense,
                item_int=item_int,
                user_int=user_int,
                timestamps=timestamps,
            )

        # ---- deterministic item_dense side channels ----
        if self.history_match_dense and self.item_dense_schema.total_dim > 0:
            item_dense = self._buf_item_dense[:B]
            item_dense[:] = 0.0
        else:
            item_dense = None

        # ---- optional item_dense history-match statistics ----
        if self.history_match_dense and item_dense is not None:
            self._fill_history_match_dense_base(item_dense, item_int)
            item_dense_global_count = np.zeros(B, dtype=np.float32)
            item_dense_global_recency = np.zeros(B, dtype=np.float32)
        else:
            item_dense_global_count = None
            item_dense_global_recency = None

        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'user_dense_feats': torch.from_numpy(user_dense.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': (
                torch.from_numpy(item_dense.copy())
                if item_dense is not None
                else torch.zeros(B, self.item_dense_schema.total_dim, dtype=torch.float32)
            ),
            'label': torch.from_numpy(labels),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
        }

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # Fused path: first collect (offsets, values, vocab_size, col_idx)
            # for every side-info column, then fill the buffer in a single pass.
            col_data = []
            for ci, fid, slot, vs in side_plan:
                col = batch.column(ci)
                col_data.append((col.offsets.to_numpy(), col.values.to_numpy(), vs, ci, fid))

            for c, (offs, vals, vs, ci, fid) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    out[i, c, :ul] = vals[s:s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul

            # Values <= 0 -> 0.
            out[out <= 0] = 0

            # Check out-of-bound values per feature's vocab_size.
            # vs==0 means no vocab info; force the whole slice to 0 so that
            # the model's 1-slot Embedding is never indexed out of range.
            for c, (_, _, vs, ci, fid) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                    self._apply_oov_unk(
                        _oov_unk_key('seq', fid, domain), slice_c, vs)
                else:
                    slice_c[:] = 0

            if self.history_match_dense:
                self._write_domain_history_match_dense(
                    item_dense=item_dense,
                    item_int=item_int,
                    domain=domain,
                    seq_values=out,
                    seq_lens=lengths,
                    max_len=max_len,
                    global_count=item_dense_global_count,
                    global_recency=item_dense_global_recency,
                )

            # Sequence data augmentation: randomly mask 25% of tokens during
            # training to simulate OOB / new-item scenarios in the test set
            # and reduce overfitting to specific item IDs.
            if self.is_training and self.shuffle:
                mask = np.random.random(out.shape) < 0.25
                out[mask] = 0

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # Time bucketing.
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            session_bucket = self._buf_seq_session[domain][:B]
            session_bucket[:] = 0
            cross_day_bucket = self._buf_seq_cross_day[domain][:B]
            cross_day_bucket[:] = 0
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                # Pad timestamps into shape (B, max_len).
                ts_padded = np.zeros((B, max_len), dtype=np.int64)
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    ts_padded[i, :ul] = ts_vals[s:s + ul]

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                # np.searchsorted returns values in [0, len(BUCKET_BOUNDARIES)].
                # After +1 the nominal range is [1, len(BUCKET_BOUNDARIES)+1];
                # the upper bound only appears when time_diff exceeds the
                # largest boundary (~1 year) and would index past
                # nn.Embedding(NUM_TIME_BUCKETS=len(BUCKET_BOUNDARIES)+1).
                # Clip raw result to [0, len(BUCKET_BOUNDARIES)-1] so the final
                # bucket id (after +1) stays within [1, len(BUCKET_BOUNDARIES)]
                # and is always a valid Embedding index. Time-diffs beyond the
                # largest boundary collapse into the last bucket.
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets

                valid_ts = ts_padded > 0
                gaps = np.zeros_like(ts_padded, dtype=np.int64)
                if max_len > 1:
                    gaps[:, 1:] = np.maximum(ts_padded[:, :-1] - ts_padded[:, 1:], 0)
                new_session = valid_ts & (gaps > 30 * 60)
                new_session[:, 0] = valid_ts[:, 0]
                sessions = np.clip(np.cumsum(new_session, axis=1), 0, 16)
                sessions[~valid_ts] = 0
                session_bucket[:] = sessions

                target_wall_day = (
                    (timestamps.astype(np.float64) + self.time_zone_offset_seconds)
                    // 86400.0
                ).reshape(-1, 1)
                event_wall_day = (
                    (ts_padded.astype(np.float64) + self.time_zone_offset_seconds)
                    // 86400.0
                )
                day_delta = np.maximum(target_wall_day - event_wall_day, 0)
                cross = np.zeros_like(ts_padded, dtype=np.int64)
                cross[valid_ts & (day_delta == 0)] = 1
                cross[valid_ts & (day_delta == 1)] = 2
                cross[valid_ts & (day_delta == 2)] = 3
                cross[valid_ts & (day_delta >= 3) & (day_delta <= 6)] = 4
                cross[valid_ts & (day_delta >= 7) & (day_delta <= 13)] = 5
                cross[valid_ts & (day_delta >= 14) & (day_delta <= 29)] = 6
                cross[valid_ts & (day_delta >= 30)] = 7
                cross_day_bucket[:] = cross

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())
            result[f'{domain}_session_bucket'] = torch.from_numpy(session_bucket.copy())
            result[f'{domain}_cross_day_bucket'] = torch.from_numpy(cross_day_bucket.copy())

            # Wall-clock time encoding. Raw timestamps are UTC; the configured
            # offset maps hour/weekday features to the local business context.
            if ts_ci is not None:
                TWO_PI = 2.0 * np.pi
                ts_float = ts_padded.astype(np.float64)
                ts_wall = ts_float + self.time_zone_offset_seconds
                target_wall_ts = (
                    timestamps.astype(np.float64) + self.time_zone_offset_seconds
                )
                # hour ∈ [0, 24), weekday ∈ [0, 7)  (Unix epoch is Thursday → +4)
                hour_of_day = np.remainder(ts_wall, 86400.0) / 3600.0
                day_index = np.floor(ts_wall / 86400.0)
                weekday = np.remainder(day_index + 4, 7)  # Monday=0
                if self.seq_time_feature_dim == 12:
                    target_hour = np.remainder(target_wall_ts, 86400.0) / 3600.0
                    target_hour_2d = np.repeat(target_hour.reshape(-1, 1), max_len, axis=1)
                    target_day_index = np.floor(target_wall_ts / 86400.0).reshape(-1, 1)
                    hist_hour_int = np.floor(hour_of_day).astype(np.int64)
                    target_hour_int = np.floor(target_hour_2d).astype(np.int64)
                    log_age = (
                        np.log1p(time_diff.astype(np.float64))
                        / np.log1p(31536000.0)
                    )
                    inv_age_hour = 1.0 / (1.0 + time_diff.astype(np.float64) / 3600.0)
                    inv_age_day = 1.0 / (1.0 + time_diff.astype(np.float64) / 86400.0)
                    same_day = (day_index == target_day_index).astype(np.float64)
                    same_hour = (hist_hour_int == target_hour_int).astype(np.float64)
                    target_is_morning_rush = (
                        (target_hour_2d >= 7.0) & (target_hour_2d < 10.0)
                    ).astype(np.float64)
                    cyclic = np.stack([
                        log_age,
                        inv_age_hour,
                        inv_age_day,
                        same_day,
                        same_hour,
                        np.sin(hour_of_day * (TWO_PI / 24.0)),
                        np.cos(hour_of_day * (TWO_PI / 24.0)),
                        np.sin(weekday * (TWO_PI / 7.0)),
                        np.cos(weekday * (TWO_PI / 7.0)),
                        np.sin(target_hour_2d * (TWO_PI / 24.0)),
                        np.cos(target_hour_2d * (TWO_PI / 24.0)),
                        target_is_morning_rush,
                    ], axis=-1).astype(np.float32)
                elif self.seq_time_feature_dim == 7:
                    target_hour = np.remainder(target_wall_ts, 86400.0) / 3600.0
                    target_hour_2d = np.repeat(
                        target_hour.reshape(-1, 1), max_len, axis=1
                    )
                    target_day_index = np.floor(
                        target_wall_ts / 86400.0
                    ).reshape(-1, 1)
                    log_age = (
                        np.log1p(time_diff.astype(np.float64))
                        / np.log1p(31536000.0)
                    )
                    same_day = (day_index == target_day_index).astype(np.float64)
                    hour_phase_sim = np.cos(
                        (hour_of_day - target_hour_2d) * (TWO_PI / 24.0)
                    )
                    cyclic = np.stack([
                        log_age,
                        np.sin(hour_of_day * (TWO_PI / 24.0)),
                        np.cos(hour_of_day * (TWO_PI / 24.0)),
                        np.sin(weekday * (TWO_PI / 7.0)),
                        np.cos(weekday * (TWO_PI / 7.0)),
                        same_day,
                        hour_phase_sim,
                    ], axis=-1).astype(np.float32)
                else:
                    cyclic = np.stack([
                        np.sin(hour_of_day * (TWO_PI / 24.0)),
                        np.cos(hour_of_day * (TWO_PI / 24.0)),
                        np.sin(weekday * (TWO_PI / 7.0)),
                        np.cos(weekday * (TWO_PI / 7.0)),
                    ], axis=-1).astype(np.float32)
                # Zero out padded positions (ts_padded == 0 means no timestamp)
                cyclic[ts_padded == 0] = 0.0
                result[f'{domain}_cyclic_time'] = torch.from_numpy(cyclic)
            else:
                # No timestamp column for this domain: output zeros
                result[f'{domain}_cyclic_time'] = torch.zeros(
                    B, max_len, self.seq_time_feature_dim, dtype=torch.float32)

        if self.history_match_dense and item_dense is not None:
            item_dense[:, HISTORY_MATCH_GLOBAL_OFFSET] = (
                np.log1p(item_dense_global_count)
                / np.log1p(float(sum(self._seq_maxlen.values()) + 1))
            )
            item_dense[:, HISTORY_MATCH_GLOBAL_OFFSET + 1] = item_dense_global_recency
            result['item_dense_feats'] = torch.from_numpy(item_dense.copy())

        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    seq_time_feature_dim: int = 4,
    history_match_dense: bool = False,
    rare16_profile: bool = False,
    time_zone_offset_hours: float = DEFAULT_TIME_ZONE_OFFSET_HOURS,
    split_time_probe: bool = False,
    use_oov_unk: bool = False,
    oov_min_count: int = 2,
    oov_max_vocab_size: int = DEFAULT_OOV_MAX_VOCAB_SIZE,
    full_train: bool = False,
    **kwargs: Any,
) -> Tuple[DataLoader, Optional[DataLoader], PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    The validation split is taken as the last ``valid_ratio`` fraction of Row
    Groups (in the file order returned by ``glob``).

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    random.seed(seed)

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    if full_train:
        n_valid_rgs = 0
        n_train_rgs = total_rgs
        logging.info("full_train=True: using all Row Groups for training; validation disabled")
    else:
        n_valid_rgs = max(1, int(total_rgs * valid_ratio))
        n_train_rgs = total_rgs - n_valid_rgs

    # train_ratio: use only the first N% of the training Row Groups.
    if train_ratio < 1.0 and not full_train:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")
    elif train_ratio < 1.0 and full_train:
        logging.info("full_train=True ignores train_ratio so the full dataset is used")

    train_rows = sum(r[2] for r in rg_info[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])

    logging.info(f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
                 f"{n_valid_rgs} valid ({valid_rows} rows)")
    if split_time_probe:
        _log_split_time_probe(
            rg_info=rg_info,
            n_train_rgs=n_train_rgs,
            offset_hours=time_zone_offset_hours,
        )

    oov_unk_maps: Optional[Dict[str, np.ndarray]] = None
    oov_unk_meta: Optional[Dict[str, Any]] = None
    if use_oov_unk:
        oov_unk_maps, oov_unk_meta = build_oov_unk_maps(
            rg_info=rg_info[:n_train_rgs],
            schema_path=schema_path,
            seq_max_lens=seq_max_lens or {},
            min_count=oov_min_count,
            max_vocab_size=oov_max_vocab_size,
        )

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=(0, n_train_rgs),
        clip_vocab=clip_vocab,
        seq_time_feature_dim=seq_time_feature_dim,
        history_match_dense=history_match_dense,
        rare16_profile=rare16_profile,
        time_zone_offset_hours=time_zone_offset_hours,
        oov_unk_maps=oov_unk_maps,
        oov_unk_meta=oov_unk_meta,
    )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    valid_loader: Optional[DataLoader]
    if full_train:
        valid_loader = None
    else:
        valid_dataset = PCVRParquetDataset(
            parquet_path=data_dir,
            schema_path=schema_path,
            batch_size=batch_size,
            seq_max_lens=seq_max_lens,
            shuffle=False,
            buffer_batches=0,
            row_group_range=(n_train_rgs, total_rgs),
            clip_vocab=clip_vocab,
            seq_time_feature_dim=seq_time_feature_dim,
            history_match_dense=history_match_dense,
            rare16_profile=rare16_profile,
            time_zone_offset_hours=time_zone_offset_hours,
            oov_unk_maps=oov_unk_maps,
            oov_unk_meta=oov_unk_meta,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=None,
            num_workers=0, pin_memory=use_cuda,
        )

    logging.info(f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}")

    return train_loader, valid_loader, train_dataset
