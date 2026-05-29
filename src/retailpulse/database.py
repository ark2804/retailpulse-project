"""
Optional database layer – PostgreSQL for storage, Redis for caching.
All functions degrade gracefully when services are not available.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


POSTGRES_URL = os.environ.get("POSTGRES_URL", "")
REDIS_URL = os.environ.get("REDIS_URL", "")


def get_engine():
    if not POSTGRES_URL:
        return None
    try:
        from sqlalchemy import create_engine
        return create_engine(POSTGRES_URL)
    except Exception as e:
        print(f"   PostgreSQL connection failed: {e}")
        return None


def get_redis():
    if not REDIS_URL:
        return None
    try:
        import redis
        return redis.from_url(REDIS_URL)
    except Exception as e:
        print(f"   Redis connection failed: {e}")
        return None


def save_to_db(df: pd.DataFrame, table: str) -> bool:
    engine = get_engine()
    if engine is None:
        return False
    try:
        df.to_sql(table, engine, if_exists="replace", index=False, chunksize=1000)
        print(f"   Saved {len(df)} rows to PostgreSQL table '{table}'")
        return True
    except Exception as e:
        print(f"   DB write failed for {table}: {e}")
        return False


def cache_set(key: str, value: str, ttl: int = 3600) -> bool:
    r = get_redis()
    if r is None:
        return False
    try:
        r.setex(key, ttl, value)
        return True
    except Exception:
        return False


def cache_get(key: str) -> str | None:
    r = get_redis()
    if r is None:
        return None
    try:
        val = r.get(key)
        return val.decode() if val else None
    except Exception:
        return None
