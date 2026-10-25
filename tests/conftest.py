import os
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("AIMX_WORKER_SANDBOX", "1")
os.environ.setdefault("REDIS_KEY_PREFIX", "aimarkets")
# Unit tests never touch a real database unless a test binds one.
os.environ["AIMX_DB_DISABLE"] = "1"
for _name in ("DATABASE_URL", "AIMX_DATABASE_URL"):
    os.environ.pop(_name, None)
