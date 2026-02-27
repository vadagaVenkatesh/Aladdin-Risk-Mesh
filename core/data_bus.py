"""
Aladdin Risk Mesh — Parquet Data Bus
Copyright (c) 2025 VDG Venkatesh. All Rights Reserved.

PROPRIETARY AND CONFIDENTIAL
This source code is the exclusive intellectual property of VDG Venkatesh.
Unauthorized use, reproduction, distribution, or modification of this code,
in whole or in part, without the express written consent of VDG Venkatesh
is strictly prohibited and may result in civil and criminal penalties.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from environment.env_config import (
    PARQUET_BASE_DIR,
    PARQUET_SIGNALS_PATH,
    PARQUET_RISK_PATH,
    PARQUET_PORTFOLIO_PATH,
    PARQUET_REGIME_PATH,
    PARQUET_EXECUTION_PATH,
    PARQUET_BACKTEST_PATH,
    PARQUET_MVO_PATH,
    PARQUET_KELLY_PATH,
    PARQUET_COMPRESSION,
)

log = logging.getLogger(__name__)

# ============================================================
# CHANNEL REGISTRY
# ============================================================
CHANNELS = {
    "signals":   PARQUET_SIGNALS_PATH,
    "risk":      PARQUET_RISK_PATH,
    "portfolio": PARQUET_PORTFOLIO_PATH,
    "regime":    PARQUET_REGIME_PATH,
    "execution": PARQUET_EXECUTION_PATH,
    "backtest":  PARQUET_BACKTEST_PATH,
    "mvo":       PARQUET_MVO_PATH,      # persisted between runs
    "kelly":     PARQUET_KELLY_PATH,    # persisted between runs
}


def _ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def init_bus() -> None:
    """Initialize all Parquet channel directories."""
    for channel, path in CHANNELS.items():
        _ensure_dir(path)
        log.debug(f"[DataBus] Channel '{channel}' ready at {path}")
    log.info(f"[DataBus] Initialized {len(CHANNELS)} channels at {PARQUET_BASE_DIR}")


# ============================================================
# WRITE
# ============================================================
def write(
    channel: str,
    df: pd.DataFrame,
    partition_cols: Optional[List[str]] = None,
    filename: str = "data.parquet",
) -> str:
    """
    Write a DataFrame to the specified channel.

    Parameters
    ----------
    channel      : One of CHANNELS keys.
    df           : DataFrame to persist.
    partition_cols: Optional Hive-style partition columns.
    filename     : File name when not using partitioning.

    Returns
    -------
    str : Full path written.
    """
    if channel not in CHANNELS:
        raise KeyError(f"[DataBus] Unknown channel '{channel}'. Valid: {list(CHANNELS)}")

    path = CHANNELS[channel]
    _ensure_dir(path)

    table = pa.Table.from_pandas(df, preserve_index=True)

    if partition_cols:
        pq.write_to_dataset(
            table,
            root_path=path,
            partition_cols=partition_cols,
            compression=PARQUET_COMPRESSION,
            use_legacy_dataset=False,
        )
        full_path = path
    else:
        full_path = os.path.join(path, filename)
        pq.write_table(table, full_path, compression=PARQUET_COMPRESSION)

    log.debug(f"[DataBus] write channel='{channel}' rows={len(df)} path={full_path}")
    return full_path


# ============================================================
# READ
# ============================================================
def read(
    channel: str,
    filename: str = "data.parquet",
    filters: Optional[list] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Read a DataFrame from the specified channel.

    Parameters
    ----------
    channel  : One of CHANNELS keys.
    filename : File name (ignored when reading partitioned datasets).
    filters  : PyArrow DNF filter list.
    columns  : Column projection.

    Returns
    -------
    pd.DataFrame
    """
    if channel not in CHANNELS:
        raise KeyError(f"[DataBus] Unknown channel '{channel}'. Valid: {list(CHANNELS)}")

    path = CHANNELS[channel]
    full_path = os.path.join(path, filename)

    # Try single-file first, fall back to dataset (partitioned)
    if os.path.isfile(full_path):
        table = pq.read_table(full_path, columns=columns, filters=filters)
    elif os.path.isdir(path) and any(Path(path).rglob("*.parquet")):
        table = pq.read_table(path, columns=columns, filters=filters)
    else:
        log.warning(f"[DataBus] No data found for channel='{channel}' at {path}")
        return pd.DataFrame()

    df = table.to_pandas()
    log.debug(f"[DataBus] read channel='{channel}' rows={len(df)}")
    return df


# ============================================================
# APPEND (streaming intraday writes)
# ============================================================
def append(
    channel: str,
    df: pd.DataFrame,
    filename: str = "data.parquet",
) -> str:
    """
    Append rows to an existing Parquet file (read-modify-write).
    For high-frequency intraday appends consider using partitioned writes.

    Returns
    -------
    str : Full path written.
    """
    existing = read(channel, filename=filename)
    if not existing.empty:
        combined = pd.concat([existing, df], ignore_index=False)
        combined = combined[~combined.index.duplicated(keep="last")]  # dedup by index
    else:
        combined = df
    return write(channel, combined, filename=filename)


# ============================================================
# PERSISTENCE HELPERS (MVO / Kelly — survive restarts)
# ============================================================
def persist_mvo(weights: pd.Series) -> None:
    """Persist MVO weights to disk (survives restarts)."""
    df = weights.to_frame(name="weight")
    write("mvo", df, filename="mvo_weights.parquet")
    log.info("[DataBus] MVO weights persisted.")


def load_mvo() -> pd.Series:
    """Load persisted MVO weights, returns empty Series if not found."""
    df = read("mvo", filename="mvo_weights.parquet")
    if df.empty:
        return pd.Series(dtype=float)
    return df["weight"]


def persist_kelly(sizes: pd.Series) -> None:
    """Persist Kelly position sizes to disk (survives restarts)."""
    df = sizes.to_frame(name="kelly_size")
    write("kelly", df, filename="kelly_sizes.parquet")
    log.info("[DataBus] Kelly sizes persisted.")


def load_kelly() -> pd.Series:
    """Load persisted Kelly sizes, returns empty Series if not found."""
    df = read("kelly", filename="kelly_sizes.parquet")
    if df.empty:
        return pd.Series(dtype=float)
    return df["kelly_size"]


# ============================================================
# CHANNEL STATUS
# ============================================================
def channel_status() -> dict:
    """Return dict of channel name -> file count / row count summary."""
    status = {}
    for channel, path in CHANNELS.items():
        files = list(Path(path).rglob("*.parquet")) if Path(path).exists() else []
        status[channel] = {
            "path": path,
            "files": len(files),
            "exists": Path(path).exists(),
        }
    return status


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    init_bus()
    print(channel_status())
