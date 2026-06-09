"""Async MySQL connection pool wrapper (engine service).

Mirrors common.db.create_pool semantics:
- charset utf8mb4, autocommit=True
- minsize/maxsize pool bounds

Engine manages two pools (catalog + metrics). Each pool is a separate MySQLAdapter
instance created at startup in server.py lifespan.

Usage:
    adapter = MySQLAdapter(host=..., port=..., user=..., password=..., db=...)
    await adapter.connect()
    async with adapter.cursor() as cur:
        await cur.execute("SELECT ...")
        rows = await cur.fetchall()
    await adapter.close()
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiomysql

log = logging.getLogger(__name__)


class MySQLAdapter:
    """Async aiomysql pool wrapper.

    Args:
        host: MySQL host.
        port: MySQL port.
        user: MySQL user.
        password: MySQL password (plain string).
        db: Database name.
        minsize: Minimum pool size.
        maxsize: Maximum pool size.
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        db: str,
        minsize: int = 1,
        maxsize: int = 10,
    ) -> None:
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._db = db
        self._minsize = minsize
        self._maxsize = maxsize
        self._pool: aiomysql.Pool | None = None

    async def connect(self) -> None:
        """Create the aiomysql connection pool."""
        if self._pool is None:
            self._pool = await aiomysql.create_pool(
                host=self._host,
                port=self._port,
                user=self._user,
                password=self._password,
                db=self._db,
                minsize=self._minsize,
                maxsize=self._maxsize,
                charset="utf8mb4",
                autocommit=True,
            )
            log.info("pool created host=%s db=%s", self._host, self._db)

    @asynccontextmanager
    async def cursor(self, dict_cursor: bool = True) -> AsyncIterator[aiomysql.Cursor]:
        """Yield a cursor from the pool.

        Args:
            dict_cursor: If True, rows are returned as dicts (default).

        Yields:
            An aiomysql cursor.

        Raises:
            RuntimeError: If connect() has not been called yet.
        """
        if self._pool is None:
            raise RuntimeError("MySQLAdapter.connect() must be called before cursor()")
        cursor_class = aiomysql.DictCursor if dict_cursor else aiomysql.Cursor
        async with self._pool.acquire() as conn:
            async with conn.cursor(cursor_class) as cur:
                yield cur

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None
            log.info("pool closed host=%s db=%s", self._host, self._db)
