r"""
Прокси для Ollama: ключи с лимитами, управление ключами через API, HTTPS.

Роли
  ADMIN_KEY      полный доступ ко всему + управление ключами (/_admin/*)
  обычные ключи  создаются админом, хранятся в .env (как SHA-256 хэши),
                 доступны только инференс и чтение; у каждого свой лимит
                 на токены и/или запросы за период (hour | day | month | total)

Что где хранится
  .env       ADMIN_KEY, настройки, строки PROXYKEY_<имя>=<хэш>:<токены>:<запросы>:<период>
             (прокси сам правит этот файл при создании/изменении/удалении ключей)
  usage.db   счётчики расхода (SQLite) - они меняются на каждый запрос, поэтому не в .env

Установка:  pip install fastapi uvicorn httpx
Запуск:     python ollama_proxy.py     (ровно ОДИН процесс, без --workers)

Настройки (в окружении или в .env):
  ADMIN_KEY, OLLAMA_URL, HOST, PORT, SSL_CERTFILE, SSL_KEYFILE, SSL_KEYFILE_PASSWORD,
  DEFAULT_TOKEN_LIMIT (100000), DEFAULT_REQUEST_LIMIT (0), DEFAULT_PERIOD (day),
  MAX_USER_BODY_MB (64)
Только из окружения: ENV_FILE (путь к .env), USAGE_DB (путь к usage.db)

Админ-API (Authorization: Bearer <ADMIN_KEY>), 0 в лимитах = без ограничения:
  GET    /_admin/keys                  список ключей и расход
  POST   /_admin/keys                  {"name","token_limit","request_limit","period"}
  PATCH  /_admin/keys/{name}           изменить лимиты/период
  POST   /_admin/keys/{name}/rotate    выпустить новый ключ взамен старого
  POST   /_admin/keys/{name}/reset     обнулить расход в текущем периоде
  DELETE /_admin/keys/{name}           отозвать ключ
  POST   /_admin/reload                перечитать .env (если правили руками)
  GET    /_admin/panel                 веб-панель (сама страница открытая, ключ вводится в ней)

Для любого ключа:
  GET    /key/status                   лимиты, период и расход своего ключа
                                       (Authorization: Bearer <key>; -1 = без лимита, у админа всё -1)
"""

import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ollama-proxy")

# ------------------------------------------------------------------ .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.getenv("ENV_FILE", os.path.join(BASE_DIR, ".env"))
USAGE_DB = os.getenv("USAGE_DB", os.path.join(BASE_DIR, "usage.db"))
ENV_LINE_RE = re.compile(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def parse_env(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[k.strip()] = v
    return out


def read_env_file() -> dict:
    try:
        with open(ENV_FILE, encoding="utf-8") as f:
            return parse_env(f.read())
    except FileNotFoundError:
        return {}


def update_env_file(changes: dict) -> None:
    """Меняет/добавляет/удаляет (значение None) переменные, сохраняя остальные строки."""
    try:
        with open(ENV_FILE, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        lines = []

    pending = dict(changes)
    out = []
    for line in lines:
        m = ENV_LINE_RE.match(line)
        if m and m.group(1) in changes:
            key = m.group(1)
            if key in pending:
                val = pending.pop(key)
                if val is not None:
                    out.append(f"{key}={val}")
            continue  # дубликаты и удалённые - выкидываем
        out.append(line)
    for key, val in pending.items():
        if val is not None:
            out.append(f"{key}={val}")
    text = "\n".join(out) + "\n"

    tmp = None
    try:  # атомарная запись: временный файл + replace
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(ENV_FILE)), prefix=".env.tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, ENV_FILE)
    except OSError:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
        # запасной вариант (например, .env смонтирован в Docker как отдельный файл)
        with open(ENV_FILE, "w", encoding="utf-8") as f:
            f.write(text)


DOTENV = read_env_file()


def cfg(name: str, default: str = "") -> str:
    return os.environ.get(name) or DOTENV.get(name) or default


# ------------------------------------------------------------- настройки
OLLAMA_URL = cfg("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
ADMIN_KEY = cfg("ADMIN_KEY").strip()
HOST = cfg("HOST", "0.0.0.0")
PORT = int(cfg("PORT", "8080"))
SSL_CERTFILE = cfg("SSL_CERTFILE") or None
SSL_KEYFILE = cfg("SSL_KEYFILE") or None
SSL_KEYFILE_PASSWORD = cfg("SSL_KEYFILE_PASSWORD") or None

PERIODS = ("hour", "day", "month", "total")
DEFAULT_TOKEN_LIMIT = int(cfg("DEFAULT_TOKEN_LIMIT", "100000"))
DEFAULT_REQUEST_LIMIT = int(cfg("DEFAULT_REQUEST_LIMIT", "0"))
DEFAULT_PERIOD = cfg("DEFAULT_PERIOD", "day")
MAX_USER_BODY = int(cfg("MAX_USER_BODY_MB", "64")) * 1024 * 1024

if not ADMIN_KEY:
    raise SystemExit("Задайте ADMIN_KEY (в окружении или в .env)")
if len(ADMIN_KEY) < 16:
    log.warning("ADMIN_KEY короче 16 символов - используйте длинный случайный ключ")
if DEFAULT_PERIOD not in PERIODS:
    raise SystemExit(f"DEFAULT_PERIOD должен быть одним из: {', '.join(PERIODS)}")
if bool(SSL_CERTFILE) != bool(SSL_KEYFILE):
    raise SystemExit("Для HTTPS нужны оба параметра: SSL_CERTFILE и SSL_KEYFILE")
for _p in (SSL_CERTFILE, SSL_KEYFILE):
    if _p and not os.path.isfile(_p):
        raise SystemExit(f"Файл не найден: {_p}")
if os.getenv("PROXY_API_KEYS"):
    log.warning("PROXY_API_KEYS больше не используется: создавайте ключи через POST /_admin/keys")

# --------------------------------------------------------------- ключи
KEY_PREFIX = "PROXYKEY_"
NAME_RE = re.compile(r"[A-Za-z0-9_]{1,32}")


@dataclass
class KeyRec:
    hash: str
    token_limit: int
    request_limit: int
    period: str


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def env_value(rec: KeyRec) -> str:
    return f"{rec.hash}:{rec.token_limit}:{rec.request_limit}:{rec.period}"


def load_keys(env: dict) -> dict:
    keys = {}
    for var, val in env.items():
        if not var.startswith(KEY_PREFIX):
            continue
        name = var[len(KEY_PREFIX):]
        try:
            h, tl, rl, period = val.split(":")
            if not NAME_RE.fullmatch(name) or not re.fullmatch(r"[0-9a-f]{64}", h) or period not in PERIODS:
                raise ValueError
            keys[name] = KeyRec(h, int(tl), int(rl), period)
        except ValueError:
            log.warning("Пропускаю некорректную строку %s в .env", var)
    return keys


KEYS = load_keys(DOTENV)


def authenticate(key: str):
    """-> (role, name): ('admin', None) / ('user', имя) / (None, None). Без раннего выхода."""
    if not key:
        return None, None
    role = name = None
    if secrets.compare_digest(key.encode(), ADMIN_KEY.encode()):
        role = "admin"
    h = hash_key(key).encode()
    for n, rec in KEYS.items():
        if secrets.compare_digest(h, rec.hash.encode()) and role is None:
            role, name = "user", n
    return role, name


# ------------------------------------------------------------ счётчики
DB = sqlite3.connect(USAGE_DB, check_same_thread=False, isolation_level=None)
DB.execute("PRAGMA journal_mode=WAL")
DB.execute(
    "CREATE TABLE IF NOT EXISTS usage ("
    "name TEXT NOT NULL, period_start TEXT NOT NULL, "
    "requests INTEGER NOT NULL DEFAULT 0, tokens INTEGER NOT NULL DEFAULT 0, "
    "PRIMARY KEY (name, period_start))"
)


def period_start(period: str) -> str:
    now = datetime.now(timezone.utc)
    return {
        "hour": now.strftime("%Y-%m-%dT%H"),
        "day": now.strftime("%Y-%m-%d"),
        "month": now.strftime("%Y-%m"),
        "total": "total",
    }[period]


def period_reset_at(period: str, now: datetime):
    if period == "hour":
        return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    if period == "day":
        return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    if period == "month":
        y, m = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
        return now.replace(year=y, month=m, day=1, hour=0, minute=0, second=0, microsecond=0)
    return None


def get_usage(name: str, ps: str):
    row = DB.execute(
        "SELECT requests, tokens FROM usage WHERE name=? AND period_start=?", (name, ps)
    ).fetchone()
    return row if row else (0, 0)


def add_usage(name: str, ps: str, requests: int = 0, tokens: int = 0) -> None:
    DB.execute(
        "INSERT INTO usage(name, period_start, requests, tokens) VALUES (?,?,?,?) "
        "ON CONFLICT(name, period_start) DO UPDATE SET "
        "requests = requests + excluded.requests, tokens = tokens + excluded.tokens",
        (name, ps, requests, tokens),
    )


def delete_usage(name: str) -> None:
    DB.execute("DELETE FROM usage WHERE name=?", (name,))


class UsageMeter:
    """Лениво разбирает ответ Ollama и достаёт расход токенов, не задерживая стрим.

    Берёт счётчики самой Ollama (prompt_eval_count + eval_count или usage из /v1).
    Если их нет - оценка: чанки стрима ~ токены, иначе ~4 символа на токен.
    """

    MARKERS = (b'"prompt_eval_count"', b'"eval_count"', b'"usage"')
    MAX_BUF = 16 * 1024 * 1024

    def __init__(self):
        self.buf = bytearray()
        self.size = 0
        self.lines = 0
        self.total = None

    def feed(self, chunk: bytes) -> None:
        self.size += len(chunk)
        self.buf.extend(chunk)
        i = self.buf.rfind(b"\n")
        if i >= 0:
            complete = bytes(self.buf[:i])
            del self.buf[: i + 1]
            for line in complete.split(b"\n"):
                self._line(line)
        if len(self.buf) > self.MAX_BUF:
            self.buf.clear()

    def _line(self, line: bytes) -> None:
        line = line.strip()
        if line.startswith(b"data:"):
            line = line[5:].strip()
        if not line or line == b"[DONE]":
            return
        self.lines += 1
        if not any(m in line for m in self.MARKERS):
            return
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                return
            usage = obj.get("usage")
            if isinstance(usage, dict):
                total = usage.get("total_tokens")
                if not isinstance(total, int):
                    total = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
                self.total = total
            elif "prompt_eval_count" in obj or "eval_count" in obj:
                self.total = int(obj.get("prompt_eval_count") or 0) + int(obj.get("eval_count") or 0)
        except Exception:
            return

    def finish(self, prompt_estimate: int) -> int:
        if self.buf:
            self._line(bytes(self.buf))
            self.buf.clear()
        if self.total is not None:
            return self.total
        completion = self.lines if self.lines > 1 else self.size // 4
        return prompt_estimate + completion


# ---------------------------------------------------------- права доступа
USER_EXACT = {
    ("GET", "/"), ("HEAD", "/"),
    ("GET", "/api/version"), ("HEAD", "/api/version"),
    ("GET", "/api/tags"), ("HEAD", "/api/tags"),
    ("GET", "/api/ps"),
    ("POST", "/api/chat"),
    ("POST", "/api/generate"),
    ("POST", "/api/embed"),
    ("POST", "/api/embeddings"),
    ("POST", "/api/show"),
}
USER_PREFIXES = ("/v1/",)
# эти запросы считаются в лимитах (чтение списка моделей и т.п. - нет)
METERED = {
    ("POST", "/api/chat"), ("POST", "/api/generate"),
    ("POST", "/api/embed"), ("POST", "/api/embeddings"),
    ("POST", "/v1/chat/completions"), ("POST", "/v1/completions"),
    ("POST", "/v1/embeddings"), ("POST", "/v1/responses"),
}

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}
DROP_REQUEST = HOP_BY_HOP | {"host", "authorization", "x-api-key"}


def norm_path(path: str) -> str:
    return path.rstrip("/") or "/"


def user_allowed(method: str, path: str) -> bool:
    p = norm_path(path)
    return (method, p) in USER_EXACT or any(p.startswith(x) for x in USER_PREFIXES)


# ---------------------------------------------------------------------- app
class ApiError(Exception):
    def __init__(self, status: int, message: str, headers: Optional[dict] = None):
        self.status, self.message, self.headers = status, message, headers or {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(
        base_url=OLLAMA_URL,
        timeout=httpx.Timeout(None, connect=10.0),
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=20),
    )
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return JSONResponse({"error": exc.message}, status_code=exc.status, headers=exc.headers)


def extract_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def auth_request(request: Request):
    role, name = authenticate(extract_key(request))
    if role is None:
        ip = request.client.host if request.client else "?"
        log.warning("401 %r %r from %s", request.method, request.url.path, ip)
        raise ApiError(401, "invalid or missing API key", {"WWW-Authenticate": "Bearer"})
    return role, name


def require_admin(request: Request) -> None:
    role, _ = auth_request(request)
    if role != "admin":
        raise ApiError(403, "admin key required")


def check_limits(token_limit, request_limit, period) -> None:
    for v in (token_limit, request_limit):
        if v is not None and not (0 <= v <= 10**12):
            raise ApiError(400, "limits must be integers >= 0 (0 = unlimited)")
    if period is not None and period not in PERIODS:
        raise ApiError(400, f"period must be one of: {', '.join(PERIODS)}")


def persist(name: str, rec: Optional[KeyRec]) -> None:
    """Сначала пишем .env, и только потом меняем состояние в памяти."""
    try:
        update_env_file({KEY_PREFIX + name: env_value(rec) if rec else None})
    except OSError as e:
        log.error("не удалось записать %s: %s", ENV_FILE, e)
        raise ApiError(500, "cannot write .env file")


def new_key() -> str:
    return "sk-" + secrets.token_urlsafe(32)


def key_response(name: str, rec: KeyRec, key: Optional[str] = None, status: int = 200):
    body = {
        "name": name,
        "token_limit": rec.token_limit,
        "request_limit": rec.request_limit,
        "period": rec.period,
    }
    if key:
        body["key"] = key
        body["note"] = "Ключ показан только один раз - сохраните его."
    return JSONResponse(body, status_code=status, headers={"Cache-Control": "no-store"})


# ----------------------------------------------------------- админ-API
class KeyCreate(BaseModel):
    name: str
    token_limit: Optional[int] = None
    request_limit: Optional[int] = None
    period: Optional[str] = None


class KeyUpdate(BaseModel):
    token_limit: Optional[int] = None
    request_limit: Optional[int] = None
    period: Optional[str] = None


@app.get("/_admin/keys")
async def list_keys(request: Request):
    require_admin(request)
    now = datetime.now(timezone.utc)
    items = []
    for name, rec in sorted(KEYS.items()):
        reqs, toks = get_usage(name, period_start(rec.period))
        reset = period_reset_at(rec.period, now)
        items.append({
            "name": name,
            "token_limit": rec.token_limit,
            "request_limit": rec.request_limit,
            "period": rec.period,
            "used_tokens": toks,
            "used_requests": reqs,
            "resets_at": reset.isoformat() if reset else None,
        })
    return {"keys": items}


@app.post("/_admin/keys")
async def create_key(body: KeyCreate, request: Request):
    require_admin(request)
    if not NAME_RE.fullmatch(body.name):
        raise ApiError(400, "name: 1-32 символа, только A-Z a-z 0-9 _")
    if body.name in KEYS:
        raise ApiError(409, "key with this name already exists")
    check_limits(body.token_limit, body.request_limit, body.period)
    key = new_key()
    rec = KeyRec(
        hash_key(key),
        DEFAULT_TOKEN_LIMIT if body.token_limit is None else body.token_limit,
        DEFAULT_REQUEST_LIMIT if body.request_limit is None else body.request_limit,
        body.period or DEFAULT_PERIOD,
    )
    persist(body.name, rec)
    KEYS[body.name] = rec
    log.info("admin: created key %r", body.name)
    return key_response(body.name, rec, key, 201)


@app.patch("/_admin/keys/{name}")
async def update_key(name: str, body: KeyUpdate, request: Request):
    require_admin(request)
    rec = KEYS.get(name)
    if rec is None:
        raise ApiError(404, "no such key")
    check_limits(body.token_limit, body.request_limit, body.period)
    new = KeyRec(
        rec.hash,
        rec.token_limit if body.token_limit is None else body.token_limit,
        rec.request_limit if body.request_limit is None else body.request_limit,
        body.period or rec.period,
    )
    persist(name, new)
    KEYS[name] = new
    log.info("admin: updated key %r", name)
    return key_response(name, new)


@app.post("/_admin/keys/{name}/rotate")
async def rotate_key(name: str, request: Request):
    require_admin(request)
    rec = KEYS.get(name)
    if rec is None:
        raise ApiError(404, "no such key")
    key = new_key()
    new = KeyRec(hash_key(key), rec.token_limit, rec.request_limit, rec.period)
    persist(name, new)
    KEYS[name] = new
    log.info("admin: rotated key %r", name)
    return key_response(name, new, key)


@app.post("/_admin/keys/{name}/reset")
async def reset_usage(name: str, request: Request):
    require_admin(request)
    if name not in KEYS:
        raise ApiError(404, "no such key")
    delete_usage(name)
    log.info("admin: reset usage of %r", name)
    return {"ok": True}


@app.delete("/_admin/keys/{name}")
async def delete_key(name: str, request: Request):
    require_admin(request)
    if name not in KEYS:
        raise ApiError(404, "no such key")
    persist(name, None)
    del KEYS[name]
    delete_usage(name)
    log.info("admin: deleted key %r", name)
    return {"ok": True}


@app.post("/_admin/reload")
async def reload_keys(request: Request):
    require_admin(request)
    new = load_keys(read_env_file())
    KEYS.clear()
    KEYS.update(new)
    log.info("admin: reloaded %d keys from %s", len(KEYS), ENV_FILE)
    return {"ok": True, "keys": len(KEYS)}


# ------------------------------------------------- статус ключа и веб-панель
@app.get("/key/status")
async def key_status(request: Request):
    """Лимиты и расход ключа из Authorization: Bearer. -1 = без лимита (у админа всё -1)."""
    role, name = auth_request(request)
    nocache = {"Cache-Control": "no-store"}
    if role == "admin":
        return JSONResponse({
            "name": "admin", "role": "admin",
            "token_limit": -1, "request_limit": -1, "period": "none",
            "used_tokens": 0, "used_requests": 0,
            "remaining_tokens": -1, "remaining_requests": -1,
            "resets_at": None,
        }, headers=nocache)

    rec = KEYS[name]
    reqs, toks = get_usage(name, period_start(rec.period))
    reset = period_reset_at(rec.period, datetime.now(timezone.utc))

    def lim(v: int) -> int:
        return v if v > 0 else -1

    def rem(limit: int, used: int) -> int:
        return max(0, limit - used) if limit > 0 else -1

    return JSONResponse({
        "name": name, "role": "user",
        "token_limit": lim(rec.token_limit),
        "request_limit": lim(rec.request_limit),
        "period": rec.period,
        "used_tokens": toks,
        "used_requests": reqs,
        "remaining_tokens": rem(rec.token_limit, toks),
        "remaining_requests": rem(rec.request_limit, reqs),
        "resets_at": reset.isoformat() if reset else None,
    }, headers=nocache)


PANEL_HTML = r'''<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex">
<title>Ollama Proxy - админка</title>
<style nonce="{{NONCE}}">
:root{--bg:#f6f7f9;--card:#fff;--fg:#1a1d21;--muted:#6b7280;--line:#e3e6ea;--accent:#3b6df0;--danger:#d64545;--ok:#2f9e63;--warn:#e0a21b}
@media (prefers-color-scheme:dark){:root{--bg:#0f1114;--card:#181b20;--fg:#e8eaed;--muted:#9aa1ab;--line:#2a2f36;--accent:#6c93ff;--danger:#ef6b6b;--ok:#4cc38a;--warn:#f0b84a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:820px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:16px}
.bar h1{flex:1 1 auto}
button,input,select{font:inherit}
button{border:1px solid var(--line);background:var(--card);color:var(--fg);padding:8px 12px;border-radius:8px;cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.danger{color:var(--danger);border-color:var(--danger)}
button:disabled{opacity:.5;cursor:default}
input,select{width:100%;padding:9px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg)}
label{display:block;margin:10px 0 4px;color:var(--muted);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:12px}
.card h2{font-size:16px;margin:0 0 2px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.tag{font-size:12px;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:1px 8px;font-weight:400}
.meter{margin-top:10px}
.meter .row{display:flex;justify-content:space-between;font-size:13px;color:var(--muted)}
.track{height:6px;border-radius:3px;background:var(--line);overflow:hidden;margin-top:4px}
.fill{height:100%;background:var(--ok);width:0}
.fill.warn{background:var(--warn)}
.fill.full{background:var(--danger)}
.reset{margin-top:8px}
.actions{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}
.actions button{padding:6px 10px;font-size:13px}
.muted{color:var(--muted);font-size:13px}
.err{color:var(--danger);min-height:1.2em;font-size:14px;margin:8px 0 0}
dialog{border:1px solid var(--line);border-radius:14px;background:var(--card);color:var(--fg);padding:18px;width:min(440px,calc(100vw - 24px))}
dialog::backdrop{background:rgba(0,0,0,.55)}
dialog h3{margin:0 0 6px;font-size:17px}
.btns{display:flex;gap:8px;justify-content:flex-end;margin-top:14px}
code.key{display:block;word-break:break-all;padding:10px;background:var(--bg);border:1px solid var(--line);border-radius:8px;margin:10px 0;user-select:all}
.login{margin-top:12vh;max-width:380px}
[hidden]{display:none!important}
</style>
</head>
<body>
<main id="login" class="login">
  <h1>Ollama Proxy</h1>
  <p class="muted">Вход в панель администратора</p>
  <form id="loginForm" autocomplete="off">
    <label for="loginKey">Админский ключ</label>
    <input id="loginKey" type="password" autocomplete="off" required>
    <p class="err" id="loginErr"></p>
    <div class="btns"><button class="primary" type="submit">Войти</button></div>
  </form>
</main>

<main id="app" hidden>
  <div class="bar">
    <h1>Ключи</h1>
    <button type="button" id="refresh">Обновить</button>
    <button type="button" id="create" class="primary">+ Ключ</button>
    <button type="button" id="logout">Выйти</button>
  </div>
  <div id="list"></div>
  <p id="empty" class="muted" hidden>Ключей пока нет.</p>
  <p id="stamp" class="muted"></p>
</main>

<dialog id="dlg"></dialog>

<script nonce="{{NONCE}}">
(() => {
'use strict';
const $ = s => document.querySelector(s);
const dlg = $('#dlg');
const nf = new Intl.NumberFormat('ru-RU');
const PERIOD = {hour: 'час', day: 'сутки', month: 'месяц', total: 'без сброса'};
let adminKey = null;   // только в памяти страницы, никуда не сохраняется
let timer = null;

function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, '');
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
}

async function api(method, path, key, body) {
  const headers = {'Authorization': 'Bearer ' + key};
  if (body) headers['Content-Type'] = 'application/json';
  const r = await fetch(path, {method, headers, cache: 'no-store', body: body ? JSON.stringify(body) : undefined});
  let data = null;
  try { data = await r.json(); } catch (e) { /* пустой ответ */ }
  if (!r.ok) {
    const msg = data && data.error ? data.error : (data && data.detail ? 'Некорректные данные' : 'HTTP ' + r.status);
    const err = new Error(msg);
    err.status = r.status;
    throw err;
  }
  return data;
}

const isAuthErr = e => e.status === 401 || e.status === 403;
const btn = (text, onclick, cls) => h('button', {type: 'button', class: cls || '', onclick}, text);

// ---------- список ----------
function meter(label, used, limit) {
  const fill = h('div', {class: 'fill'});
  if (limit > 0) {
    const pct = Math.min(100, used / limit * 100);
    fill.style.width = pct + '%';
    if (pct >= 100) fill.classList.add('full');
    else if (pct >= 80) fill.classList.add('warn');
  }
  return h('div', {class: 'meter'},
    h('div', {class: 'row'}, h('span', {}, label),
      h('span', {}, nf.format(used) + ' / ' + (limit > 0 ? nf.format(limit) : '\u221e'))),
    h('div', {class: 'track'}, fill));
}

function card(k) {
  const reset = k.resets_at
    ? 'Сброс: ' + new Date(k.resets_at).toLocaleString('ru-RU')
    : 'Расход не сбрасывается';
  return h('div', {class: 'card'},
    h('h2', {}, k.name, h('span', {class: 'tag'}, PERIOD[k.period] || k.period)),
    meter('Токены', k.used_tokens, k.token_limit),
    meter('Запросы', k.used_requests, k.request_limit),
    h('div', {class: 'muted reset'}, reset),
    h('div', {class: 'actions'},
      btn('Изменить', () => editKey(k)),
      btn('Новый ключ', () => rotateKey(k)),
      btn('Сбросить расход', () => resetKey(k)),
      btn('Удалить', () => deleteKey(k), 'danger')));
}

function render(keys) {
  $('#list').replaceChildren(...keys.map(card));
  $('#empty').hidden = keys.length > 0;
  $('#stamp').textContent = 'Обновлено: ' + new Date().toLocaleTimeString('ru-RU');
}

async function refresh() {
  if (!adminKey) return;
  try {
    render((await api('GET', '/_admin/keys', adminKey)).keys);
  } catch (e) {
    if (isAuthErr(e)) logout('Сессия недействительна, войдите снова');
    else $('#stamp').textContent = 'Ошибка обновления: ' + e.message;
  }
}

// ---------- диалоги ----------
function closeDlg() { if (dlg.open) dlg.close(); dlg.replaceChildren(); }
dlg.addEventListener('close', () => { if (!dlg.open) dlg.replaceChildren(); });

function field(id, label, attrs) {
  return [h('label', {for: id}, label), h('input', {id, type: 'text', autocomplete: 'off', ...attrs})];
}

function periodSelect(id, value, withDefault) {
  const opts = Object.entries(PERIOD).map(([v, t]) => h('option', {value: v}, t));
  if (withDefault) opts.unshift(h('option', {value: ''}, 'по умолчанию'));
  const s = h('select', {id}, opts);
  s.value = value;
  return [h('label', {for: id}, 'Период'), s];
}

const num = v => {
  v = v.trim();
  if (v === '') return undefined;
  const n = Number(v);
  if (!Number.isInteger(n) || n < 0) throw new Error('Лимиты - целые числа от 0 (0 = без лимита)');
  return n;
};

// Каждое изменение требует повторного ввода админского ключа.
function askAndRun({title, text, fields = [], submit = 'Подтвердить', danger = false, run}) {
  const err = h('p', {class: 'err'});
  const keyInput = h('input', {id: 'dlgKey', type: 'password', required: true, autocomplete: 'off'});
  const submitBtn = h('button', {type: 'submit', class: danger ? 'danger' : 'primary'}, submit);
  const form = h('form', {autocomplete: 'off'},
    h('h3', {}, title),
    text ? h('p', {class: 'muted'}, text) : null,
    fields,
    h('label', {for: 'dlgKey'}, 'Админский ключ (подтверждение)'),
    keyInput, err,
    h('div', {class: 'btns'}, btn('Отмена', closeDlg), submitBtn));

  form.addEventListener('submit', async ev => {
    ev.preventDefault();
    err.textContent = '';
    submitBtn.disabled = true;
    let result;
    try {
      result = await run(keyInput.value.trim(), form.elements);
    } catch (e) {
      err.textContent = isAuthErr(e) ? 'Неверный админский ключ' : e.message;
      submitBtn.disabled = false;
      return;
    }
    closeDlg();
    refresh();
    if (result && result.key) showKey(result);
  });

  dlg.replaceChildren(form);
  dlg.showModal();
}

function showKey(res) {
  const code = h('code', {class: 'key'}, res.key);
  const msg = h('p', {class: 'muted'});
  const copy = btn('Копировать', async () => {
    try {
      await navigator.clipboard.writeText(res.key);
      msg.textContent = 'Скопировано';
    } catch (e) {
      const range = document.createRange();
      range.selectNodeContents(code);
      const sel = getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      msg.textContent = 'Ключ выделен - скопируйте вручную';
    }
  }, 'primary');
  dlg.replaceChildren(
    h('h3', {}, 'Ключ \u00ab' + res.name + '\u00bb'),
    h('p', {class: 'muted'}, 'Ключ показан только один раз - сохраните его.'),
    code, msg,
    h('div', {class: 'btns'}, copy, btn('Готово', closeDlg)));
  dlg.showModal();
}

// ---------- действия ----------
function createKey() {
  askAndRun({
    title: 'Новый ключ',
    text: 'Пустое поле - значение по умолчанию, 0 - без лимита.',
    fields: [
      ...field('fName', 'Имя (A-Z, a-z, 0-9, _)', {required: true, pattern: '[A-Za-z0-9_]{1,32}', maxlength: 32}),
      ...field('fTok', 'Лимит токенов', {inputmode: 'numeric'}),
      ...field('fReq', 'Лимит запросов', {inputmode: 'numeric'}),
      ...periodSelect('fPer', '', true)],
    submit: 'Создать',
    run: (key, f) => api('POST', '/_admin/keys', key, {
      name: f.fName.value.trim(),
      token_limit: num(f.fTok.value),
      request_limit: num(f.fReq.value),
      period: f.fPer.value || undefined,
    }),
  });
}

function editKey(k) {
  askAndRun({
    title: 'Изменить \u00ab' + k.name + '\u00bb',
    text: '0 - без лимита.',
    fields: [
      ...field('fTok', 'Лимит токенов', {inputmode: 'numeric', value: k.token_limit}),
      ...field('fReq', 'Лимит запросов', {inputmode: 'numeric', value: k.request_limit}),
      ...periodSelect('fPer', k.period, false)],
    submit: 'Сохранить',
    run: (key, f) => api('PATCH', '/_admin/keys/' + encodeURIComponent(k.name), key, {
      token_limit: num(f.fTok.value),
      request_limit: num(f.fReq.value),
      period: f.fPer.value,
    }),
  });
}

function rotateKey(k) {
  askAndRun({
    title: 'Новый ключ для \u00ab' + k.name + '\u00bb',
    text: 'Старый ключ перестанет работать сразу. Лимиты сохранятся.',
    submit: 'Выпустить',
    run: key => api('POST', '/_admin/keys/' + encodeURIComponent(k.name) + '/rotate', key),
  });
}

function resetKey(k) {
  askAndRun({
    title: 'Сбросить расход \u00ab' + k.name + '\u00bb?',
    text: 'Счётчики токенов и запросов в текущем периоде обнулятся.',
    submit: 'Сбросить',
    run: key => api('POST', '/_admin/keys/' + encodeURIComponent(k.name) + '/reset', key),
  });
}

function deleteKey(k) {
  askAndRun({
    title: 'Удалить \u00ab' + k.name + '\u00bb?',
    text: 'Ключ будет отозван, расход удалён. Это нельзя отменить.',
    submit: 'Удалить', danger: true,
    run: key => api('DELETE', '/_admin/keys/' + encodeURIComponent(k.name), key),
  });
}

// ---------- вход / выход ----------
function logout(message) {
  adminKey = null;
  clearInterval(timer);
  closeDlg();
  $('#list').replaceChildren();
  $('#app').hidden = true;
  $('#login').hidden = false;
  $('#loginErr').textContent = message || '';
}

$('#loginForm').addEventListener('submit', async ev => {
  ev.preventDefault();
  const input = $('#loginKey');
  const err = $('#loginErr');
  err.textContent = '';
  try {
    const data = await api('GET', '/_admin/keys', input.value.trim());
    adminKey = input.value.trim();
    input.value = '';
    $('#login').hidden = true;
    $('#app').hidden = false;
    render(data.keys);
    clearInterval(timer);
    timer = setInterval(refresh, 30000);
  } catch (e) {
    err.textContent = isAuthErr(e) ? 'Неверный админский ключ' : 'Ошибка: ' + e.message;
  }
});

$('#refresh').addEventListener('click', refresh);
$('#create').addEventListener('click', createKey);
$('#logout').addEventListener('click', () => logout(''));
})();
</script>
</body>
</html>
'''


@app.get("/_admin/panel")
async def admin_panel():
    """Страница статическая и секретов не содержит; доступ к данным - только по админскому ключу."""
    nonce = secrets.token_urlsafe(16)
    csp = (
        f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )
    return HTMLResponse(
        PANEL_HTML.replace("{{NONCE}}", nonce),
        headers={
            "Content-Security-Policy": csp,
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Cache-Control": "no-store",
        },
    )


# --------------------------------------------------------------- прокси
def check_quota(name: str, rec: KeyRec, ps: str) -> None:
    reqs, toks = get_usage(name, ps)
    if rec.token_limit and toks >= rec.token_limit:
        what, used, limit = "token", toks, rec.token_limit
    elif rec.request_limit and reqs >= rec.request_limit:
        what, used, limit = "request", reqs, rec.request_limit
    else:
        return
    now = datetime.now(timezone.utc)
    reset = period_reset_at(rec.period, now)
    headers = {}
    if reset:
        headers["Retry-After"] = str(max(1, int((reset - now).total_seconds())))
    scope = "in total" if rec.period == "total" else f"per {rec.period}"
    raise ApiError(429, f"{what} limit reached ({used}/{limit} {scope})", headers)


async def read_capped(request: Request, cap: int) -> bytes:
    buf = bytearray()
    async for chunk in request.stream():
        buf.extend(chunk)
        if len(buf) > cap:
            raise ApiError(413, "request body too large")
    return bytes(buf)


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
async def proxy(path: str, request: Request):
    client_ip = request.client.host if request.client else "?"
    method = request.method

    # 1) защита от обхода whitelist через ../ и //
    # (% ? # в уже раскодированном пути значат двойное кодирование - у Ollama таких путей нет)
    if (
        "//" in path or "\\" in path or "%" in path or "?" in path or "#" in path
        or any(ord(c) < 32 for c in path)
        or any(s in (".", "..") for s in path.split("/"))
    ):
        raise ApiError(400, "bad path")
    if path == "_admin" or path.startswith("_admin/"):
        raise ApiError(404, "not found")

    # 2) аутентификация и права
    role, name = auth_request(request)
    full = "/" + path
    if role == "user" and not user_allowed(method, full):
        log.warning("403 %r %r from %s", method, request.url.path, client_ip)
        raise ApiError(403, "admin key required for this endpoint")

    # 3) лимиты (только для обычных ключей и только на инференс)
    rec = KEYS.get(name) if role == "user" else None
    metered = rec is not None and (method, norm_path(full)) in METERED
    ps = None
    if metered:
        ps = period_start(rec.period)
        check_quota(name, rec, ps)

    req_body = None
    if role == "user":  # у обычных ключей тела маленькие - читаем целиком, с потолком
        req_body = await read_capped(request, MAX_USER_BODY)

    if metered:
        add_usage(name, ps, requests=1)

    def refund() -> None:  # неудавшиеся запросы не списываются с лимита
        if metered:
            add_usage(name, ps, requests=-1)

    # 4) проксирование
    client: httpx.AsyncClient = request.app.state.client
    drop = DROP_REQUEST | ({"content-length"} if req_body is not None else set())
    headers = {k: v for k, v in request.headers.items() if k.lower() not in drop}
    if role == "user":
        headers["accept-encoding"] = "identity"  # чтобы можно было читать usage в ответе

    has_body = "content-length" in request.headers or "transfer-encoding" in request.headers
    if req_body is not None:
        content = req_body or None
    else:
        content = request.stream() if has_body else None

    upstream_req = client.build_request(
        method, full, params=request.query_params, headers=headers, content=content
    )

    try:
        upstream = await client.send(upstream_req, stream=True)
    except httpx.ConnectError:
        refund()
        raise ApiError(502, "Ollama is unreachable")
    except httpx.HTTPError:
        refund()
        log.exception("upstream error")
        raise ApiError(502, "upstream error")

    count_tokens = metered and upstream.status_code < 400
    if metered and not count_tokens:
        refund()

    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP}
    meter = UsageMeter() if count_tokens else None
    prompt_estimate = len(req_body or b"") // 4

    async def body():
        try:
            async for chunk in upstream.aiter_raw():
                if meter:
                    meter.feed(chunk)
                yield chunk
        finally:
            if meter:  # пишем расход и при обрыве клиентом
                try:
                    add_usage(name, ps, tokens=meter.finish(prompt_estimate))
                except Exception:
                    log.exception("failed to record usage")
            await upstream.aclose()

    return StreamingResponse(body(), status_code=upstream.status_code, headers=resp_headers)


# ---------------------------------------------------------------- запуск
if __name__ == "__main__":
    import uvicorn

    scheme = "https" if SSL_CERTFILE else "http"
    if not SSL_CERTFILE:
        log.warning("TLS выключен: ключи передаются открытым текстом (задайте SSL_CERTFILE/SSL_KEYFILE)")
    log.info("Слушаю %s://%s:%s -> %s | ключей: %d | .env: %s",
             scheme, HOST, PORT, OLLAMA_URL, len(KEYS), ENV_FILE)
    log.info("Админ-панель: %s://%s:%s/_admin/panel", scheme, "localhost" if HOST == "0.0.0.0" else HOST, PORT)

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        ssl_certfile=SSL_CERTFILE,
        ssl_keyfile=SSL_KEYFILE,
        ssl_keyfile_password=SSL_KEYFILE_PASSWORD,
    )
