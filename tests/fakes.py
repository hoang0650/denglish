"""In-memory stand-ins for Redis and a psycopg connection."""

from __future__ import annotations


class FakeRedis:
    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.ttl: dict[str, int] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.sets: dict[str, set[str]] = {}

    def set(self, key, value, ex=None):
        self.kv[key] = value if isinstance(value, str) else value.decode()
        if ex is not None:
            self.ttl[key] = int(ex)

    def setex(self, key, ttl, value):
        self.set(key, value, ex=ttl)

    def get(self, key):
        return self.kv.get(key)

    def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update({str(k): float(v) for k, v in mapping.items()})

    def zrevrange(self, key, start, end):
        items = sorted(self.zsets.get(key, {}).items(), key=lambda item: -item[1])
        return [name for name, _score in items[start : end + 1]]

    def zrem(self, key, member):
        self.zsets.get(key, {}).pop(str(member), None)

    def sadd(self, key, member):
        self.sets.setdefault(key, set()).add(str(member))

    def srem(self, key, member):
        self.sets.get(key, set()).discard(str(member))

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def ping(self):
        return True


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if self.conn.fail:
            raise RuntimeError("boom postgresql://user:hunter2@db.example:5432/x password=hunter2")
        self.conn.calls.append((" ".join(str(sql).split()), params))


class FakeConn:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list = []
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = True
