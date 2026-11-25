"""AI first pass with Gemini or Claude: read the plan like a person would.

The model reads each part of the sheet and reports every labelled room: its name and
number, a point inside it, a rough box, and whether it looks like a takeoff area
(gym, fitness, training floor ...). The exact outlines are still traced from the
drawing by rooms.py, using these points the same way it uses printed room labels.

Only plan images are sent (no file names or project details). Use a paid Gemini
API key: Google's free tier may use content to improve its products. Claude models
(model ids starting "claude-") use the Claude API key in config/anthropic_key.txt
or ANTHROPIC_API_KEY.
"""
import base64
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf

try:                                   # python.org's macOS Python ships without CA certificates
    import certifi
    import ssl
    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL = None

ROOT = Path(__file__).resolve().parent.parent
KEY_FILE = ROOT / "config" / "gemini_key.txt"          # local only, never committed
CLAUDE_KEY_FILE = ROOT / "config" / "anthropic_key.txt"  # local only, never committed
SETTINGS_FILE = ROOT / "config" / "ai_settings.json"   # chosen model + requests per minute
MODEL = "gemini-3.5-flash-lite"                         # default model

# USD per 1M tokens (input, output), paid Standard tier, from Google's pricing page (Nov 2026).
# Some prices change on a date: (from_date, input, output) entries, latest that applies wins.
PRICES = {
    "gemini-3.5-flash-lite": [("2026-01-01", 0.30, 2.50)],
    "gemini-3.1-flash-lite": [("2026-01-01", 0.25, 1.50)],
    "gemini-3.6-flash": [("2026-01-01", 0.75, 3.75), ("2027-01-01", 1.50, 7.50)],
    "gemini-3.8-flash": [("2026-01-01", 0.75, 3.75), ("2027-01-01", 1.50, 7.50)],
    "gemini-3-flash-preview": [("2026-01-01", 0.50, 3.00)],
    "gemini-3.1-pro-preview": [("2026-01-01", 2.00, 12.00)],
    "gemini-4-carbon": [("2026-01-01", 2.00, 10.00), ("2026-12-01", 4.00, 20.00)],
    "gemini-4-argon": [("2026-01-01", 2.00, 10.00), ("2026-12-01", 4.00, 20.00)],
    # Claude API standard prices (Nov 2026). Haiku's is for prompts up to 100K tokens (a section is ~6K).
    "claude-haiku-5-5": [("2026-01-01", 0.10, 0.50)],
    "claude-sonnet-5-5": [("2026-01-01", 2.00, 10.00)],
    "claude-opus-5-5": [("2026-01-01", 4.00, 20.00)],
}
CLAUDE_MODELS = {"claude-haiku-5-5": "Claude Haiku 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5",
                 "claude-opus-5-5": "Claude Opus 5.5"}


TRACE_MODEL = "claude-opus-5-5"     # traces room borders (ai_outline.py); Sonnet is cheaper but missed open edges


def provider(model):
    return "claude" if (model or "").startswith("claude") else "gemini"


def provider_name(model):
    return "Claude" if provider(model) == "claude" else "Gemini"


def price(model, today=None):
    """(input, output) USD per 1M tokens for a model today, or None if not known."""
    import datetime
    today = today or datetime.date.today().isoformat()
    rows = [r for r in PRICES.get(model, []) if r[0] <= today]
    return (rows[-1][1], rows[-1][2]) if rows else None


def get_settings():
    s = {"model": MODEL, "rpm": int(os.environ.get("GEMINI_RPM", "15")), "trace_model": TRACE_MODEL}
    if SETTINGS_FILE.exists():
        try:
            s.update({k: v for k, v in json.loads(SETTINGS_FILE.read_text()).items() if k in ("model", "rpm", "trace_model")})
        except ValueError:
            pass
    if os.environ.get("GEMINI_RPM"):
        s["rpm"] = int(os.environ["GEMINI_RPM"])
    return s


def save_settings(model=None, rpm=None, trace_model=None):
    s = get_settings()
    if trace_model:
        if trace_model not in CLAUDE_MODELS:
            raise AiError("Borders can only be traced with a Claude model.")
        s["trace_model"] = trace_model
    if model:
        if not re.fullmatch(r"[a-z0-9][a-z0-9.\-]{2,80}", model):
            raise AiError("That isn't a model name.")
        s["model"] = model
    if rpm is not None:
        rpm = int(rpm)
        if not 1 <= rpm <= 10000:
            raise AiError("Requests per minute should be between 1 and 10000.")
        s["rpm"] = rpm
    SETTINGS_FILE.parent.mkdir(exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(s, indent=2))
    return s


def list_models():
    """Models the saved keys can use: Gemini models the Gemini key lists, plus the Claude
    models when a Claude key is saved. [{id, name, price}]"""
    out, err = [], None
    if get_key():
        try:
            out = _gemini_models(get_key())
        except AiError as e:
            err = e
    if get_key("claude"):
        out = [{"id": m, "name": n, "price": price(m)} for m, n in CLAUDE_MODELS.items()] + out
    if not out:
        raise err or AiError("No API key yet.")
    return out


def _gemini_models(key):
    """Gemini models this key can call for image + text requests: [{id, name, price}]."""
    url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000"
    req = urllib.request.Request(url, headers={"x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=20, context=_SSL) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise AiError(f"Google error {e.code} listing models.")
    except urllib.error.URLError as e:
        raise AiError(f"Couldn't reach Google ({e.reason}).")
    out = []
    for m in data.get("models", []):
        mid = m.get("name", "").split("/")[-1]
        if not mid.startswith("gemini") or "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        if any(w in mid for w in ("embedding", "tts", "audio", "live", "image", "robotics", "omni", "banana",
                                  "transcribe", "computer-use", "customtools")):
            continue
        out.append({"id": mid, "name": m.get("displayName", mid), "price": price(mid)})
    out.sort(key=lambda m: (m["price"] is None, -(m["price"] or (0, 0))[0], m["id"]))
    return out
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Requests per minute allowed on the Gemini key. 15 matches the free tier and is safe anywhere;
# on a paid tier set GEMINI_RPM (e.g. 300) to run a sheet faster.
MAX_RPM = int(os.environ.get("GEMINI_RPM", "15"))

TILE_PX = 2048            # each request's image size
PX_PER_PT = 2.0           # ~144 dpi: room labels stay legible
OVERLAP = 0.2

PROMPT = """This image is one section of an architectural floor plan (a construction drawing).

List every room or space whose name and/or room number is printed in this image, for example
"FITNESS 422", "YOGA 420", "LOBBY 100", "CORR C402". Skip anything in a title block, legend,
schedule, notes, key plan or enlarged detail, and skip labels cut off at the image edge.

For each room give:
- name: the room name as printed (e.g. "TRAINING FLOOR"), without the number
- number: the room number as printed, or null if none
- point: [y, x] of a spot on open floor inside the room, away from text, walls and furniture,
  normalized to 0-1000 over this image
- box_2d: [ymin, xmin, ymax, xmax] of the room's floor area within this image, normalized 0-1000
- outline: the room's floor area as a polygon, a list of [y, x] corners in order (normalized 0-1000),
  tracing the inside face of the walls that enclose it. Follow the walls around every corner and
  notch (an L-shaped room has 6 corners). Go straight across door openings. Ignore furniture,
  equipment, hatching, dimension lines and text inside the room. Where the room runs past the
  image edge, follow the image edge.
- takeoff_candidate: true if the room is a gym, fitness or weight room, training floor,
  yoga/spin/group-exercise studio, or is marked with rubber or athletic flooring; otherwise false
- finish_tag: the floor finish tag printed in the room (e.g. "RF-1", "CPT-3"), or null

Also give scale_note: the drawing scale text if one is printed here (e.g. 1/8" = 1'-0"), else null."""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "rooms": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {
                "name": {"type": "STRING"},
                "number": {"type": "STRING", "nullable": True},
                "point": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                "outline": {"type": "ARRAY", "items": {"type": "ARRAY", "items": {"type": "INTEGER"}}},
                "takeoff_candidate": {"type": "BOOLEAN"},
                "finish_tag": {"type": "STRING", "nullable": True},
            },
            "required": ["name", "point", "box_2d", "outline", "takeoff_candidate"],
        }},
        "scale_note": {"type": "STRING", "nullable": True},
    },
    "required": ["rooms"],
}


class AiError(Exception):
    pass


# ---------------------------------------------------------------- key
def _env_key(prov):
    names = ("ANTHROPIC_API_KEY",) if prov == "claude" else ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    return next((os.environ[n].strip() for n in names if os.environ.get(n)), None)


def get_key(prov="gemini"):
    """The saved API key for "gemini" (default) or "claude", or None."""
    k = _env_key(prov)
    if k:
        return k
    f = CLAUDE_KEY_FILE if prov == "claude" else KEY_FILE
    if f.exists():
        return f.read_text(encoding="utf-8").strip() or None
    return None


def clean_key(key):
    """Tidy a pasted key: surrounding spaces/quotes, or a pasted 'GEMINI_API_KEY=' / 'key:' prefix."""
    k = (key or "").strip().strip("\"'`").strip()
    k = re.sub(r"^(export\s+)?(GEMINI_API_KEY|GOOGLE_API_KEY|ANTHROPIC_API_KEY|x-goog-api-key|x-api-key|api[_ ]?key|key)\s*[:=]\s*", "", k, flags=re.I)
    return k.strip().strip("\"'`").strip()


def check_key(key, model=None):
    """Ask Google whether the key works, and whether the model is available to it.
    Returns (ok, message, model_available); ok is None when Google couldn't be reached."""
    url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000"
    req = urllib.request.Request(url, headers={"x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=20, context=_SSL) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")
        try:
            msg = json.loads(msg)["error"]["message"]
        except Exception:
            pass
        return False, f"Google rejected this key: {msg[:200]}", False
    except urllib.error.URLError as e:
        return None, f"Couldn't reach Google to check the key ({e.reason}). Saved it anyway.", None
    model = model or get_settings()["model"]
    names = {m.get("name", "").split("/")[-1] for m in data.get("models", [])}
    if model in names:
        return True, "Key works.", True
    similar = sorted(n for n in names if "flash-lite" in n)
    return True, (f"Key works, but {model} isn't listed for this key"
                  + (f" (available: {', '.join(similar[:4])})" if similar else "") + "."), False


def check_claude_key(key):
    """Ask Anthropic whether the key works. Returns (ok, message); ok is None if unreachable."""
    import anthropic
    try:
        anthropic.Anthropic(api_key=key, max_retries=1, timeout=20).models.list(limit=1)
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
        return False, "Anthropic rejected this key. Copy the whole key again from platform.claude.com (Settings → API keys)."
    except anthropic.APIConnectionError:
        return None, "Couldn't reach Anthropic to check the key. Saved it anyway."
    except anthropic.APIStatusError as e:
        return None, f"Anthropic couldn't check the key right now (error {e.status_code}). Saved it anyway."
    return True, "Key works."


def save_key(key):
    """Save a pasted key: a Claude key (starts "sk-ant-") or a Gemini key.
    Returns (message, model_available, provider)."""
    key = clean_key(key)
    if not key:
        raise AiError("Paste the key first.")
    prov = "claude" if key.startswith("sk-ant-") else "gemini"
    if re.search(r"\s", key) or len(key) < 20:
        raise AiError("That looks incomplete or has spaces in it. Copy the whole key again.")
    if prov == "claude":
        if key.startswith("sk-ant-admin"):
            raise AiError("That's an Admin key. Create a normal API key instead (Settings → API keys).")
        ok, msg = check_claude_key(key)
        model_ok = True
    else:
        ok, msg, model_ok = check_key(key)
    if ok is False:
        raise AiError(msg)
    f = CLAUDE_KEY_FILE if prov == "claude" else KEY_FILE
    f.parent.mkdir(exist_ok=True)
    f.write_text(key, encoding="utf-8")
    os.chmod(f, 0o600)
    return msg, model_ok, prov


def key_status():
    st = get_settings()
    out = {"model": st["model"], "rpm": st["rpm"], "price": price(st["model"]), "trace_model": st["trace_model"],
           "provider": provider(st["model"]), "provider_name": provider_name(st["model"])}
    for prov in ("gemini", "claude"):
        k = get_key(prov)
        out[prov] = {"configured": bool(k), "hint": f"…{k[-4:]}" if k else None,
                     "source": ("environment" if _env_key(prov) else "file") if k else None}
    cur = out[out["provider"]]
    out.update(configured=cur["configured"], hint=cur["hint"], source=cur["source"])
    return out


# ---------------------------------------------------------------- tiles
def tiles(page, px_per_pt=PX_PER_PT, tile_px=TILE_PX, overlap=OVERLAP):
    """Overlapping sections of the page, in PDF points: (x0, y0, x1, y1)."""
    w, h = page.rect.width, page.rect.height
    size = tile_px / px_per_pt
    step = size * (1 - overlap)
    nx = max(1, math.ceil((w - size) / step) + 1) if w > size else 1
    ny = max(1, math.ceil((h - size) / step) + 1) if h > size else 1
    out = []
    for j in range(ny):
        for i in range(nx):
            x0 = min(i * step, max(0, w - size))
            y0 = min(j * step, max(0, h - size))
            out.append((x0, y0, min(w, x0 + size), min(h, y0 + size)))
    return out


def _tile_png(page, rect, px_per_pt):
    pix = page.get_pixmap(matrix=pymupdf.Matrix(px_per_pt, px_per_pt), clip=pymupdf.Rect(*rect), alpha=False)
    # skip (nearly) blank sections: no point paying to read empty paper
    samples = pix.samples
    step = max(1, len(samples) // 20000)
    dark = sum(1 for b in samples[::step] if b < 200)
    if dark < 0.002 * len(samples[::step]):
        return None, pix.width, pix.height
    return pix.tobytes("png"), pix.width, pix.height


# ---------------------------------------------------------------- Gemini
class _Pacer:
    """Keeps requests under MAX_RPM across threads, and holds everyone back after a 429."""
    def __init__(self, rpm):
        import threading
        self.rpm, self.lock, self.sent, self.hold_until, self.count = max(1, rpm), threading.Lock(), [], 0.0, 0

    def wait(self):
        while True:
            with self.lock:
                now = time.time()
                self.sent = [t for t in self.sent if now - t < 60]
                delay = max(self.hold_until - now, 0)
                if not delay and len(self.sent) >= self.rpm:
                    delay = 60 - (now - self.sent[0]) + 0.2
                if not delay:
                    self.sent.append(now)
                    self.count += 1
                    return
            time.sleep(min(delay, 5))

    def back_off(self, seconds):
        with self.lock:
            self.hold_until = max(self.hold_until, time.time() + seconds)


def _retry_delay(err_text, default):
    """Seconds Google asks us to wait (RetryInfo.retryDelay like "27s"), else default."""
    try:
        for d in json.loads(err_text)["error"].get("details", []):
            if "retryDelay" in d:
                return min(90.0, float(str(d["retryDelay"]).rstrip("s")) + 1)
    except Exception:
        pass
    return default


_NO_THINKING_CONFIG = set()      # models that rejected the thinking setting (learned per run)


def _call(key, png, model=MODEL, retries=4, pacer=None, prompt=None, schema=None, thinking="low"):
    def request():
        gen = {"responseMimeType": "application/json", "responseSchema": schema or SCHEMA, "temperature": 0}
        if model not in _NO_THINKING_CONFIG:
            # reading labels and pointing at rooms doesn't need long reasoning: keep it short (faster, cheaper)
            gen["thinkingConfig"] = {"thinkingLevel": thinking}
        body = {"contents": [{"parts": [
                    {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(png).decode()}},
                    {"text": prompt or PROMPT}]}],
                "generationConfig": gen}
        return urllib.request.Request(ENDPOINT.format(model=model), data=json.dumps(body).encode(),
                                      headers={"Content-Type": "application/json", "x-goog-api-key": key},
                                      method="POST")
    req = request()
    for attempt in range(retries + 1):
        if pacer:
            pacer.wait()
        try:
            with urllib.request.urlopen(req, timeout=300, context=_SSL) as r:
                data = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            msg = raw
            try:
                msg = json.loads(raw)["error"]["message"]
            except Exception:
                pass
            if e.code == 429 and attempt < retries:
                wait = _retry_delay(raw, 20.0 * (attempt + 1))   # rate limit: wait as long as Google asks
                if pacer:
                    pacer.back_off(wait)
                else:
                    time.sleep(wait)
                continue
            if e.code == 400 and "think" in msg.lower() and model not in _NO_THINKING_CONFIG:
                _NO_THINKING_CONFIG.add(model)      # this model doesn't take the setting: send without it
                req = request()
                continue
            if e.code in (500, 502, 503, 504) and attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            if e.code == 429:
                raise AiError("Gemini's rate limit was reached and didn't clear. Wait a minute and try again"
                              + (" (the daily request limit may be used up)." if "day" in msg.lower() else "."))
            if e.code == 402:
                raise AiError("Your Gemini account has no prepaid credit left. Add credit in Google AI Studio "
                              "(aistudio.google.com → your project → Billing), then run the AI first pass again. "
                              "Nothing was charged for this attempt.")
            if e.code in (400, 401, 403) and "key" in msg.lower():
                raise AiError("Gemini rejected the API key. Check it in AI settings.")
            raise AiError(f"Gemini error {e.code}: {msg[:300]}")
        except urllib.error.URLError as e:
            if attempt < retries:
                time.sleep(2)
                continue
            raise AiError(f"Couldn't reach Gemini ({e.reason}). Check the internet connection.")
    usage = data.get("usageMetadata", {})
    try:
        text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        parsed = json.loads(text)
    except (KeyError, IndexError, ValueError):
        reason = (data.get("candidates") or [{}])[0].get("finishReason") or data.get("promptFeedback")
        raise AiError(f"Gemini returned no usable answer ({reason}).")
    return parsed, usage


# ---------------------------------------------------------------- Claude
def _claude_schema(g):
    """The Gemini response schema in JSON Schema form for Claude's structured outputs."""
    t = g["type"].lower()
    if t == "object":
        out = {"type": "object", "properties": {k: _claude_schema(v) for k, v in g["properties"].items()},
               "required": list(g.get("required", [])), "additionalProperties": False}
    elif t == "array":
        out = {"type": "array", "items": _claude_schema(g["items"])}
    else:
        out = {"type": t}
    return {"anyOf": [out, {"type": "null"}]} if g.get("nullable") else out


def _claude_client(key):
    import anthropic
    # the SDK retries rate limits (429) and server errors itself, with backoff
    return anthropic.Anthropic(api_key=key, max_retries=4, timeout=300)


def _call_claude(client, png, model, pacer=None, prompt=None, schema=None, effort="low"):
    """One Claude request: (parsed JSON, usage in the same keys as Gemini's)."""
    import anthropic
    if pacer:
        pacer.wait()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=16000,
            # reading labels and pointing at rooms doesn't need long reasoning: keep it short (faster, cheaper)
            output_config={"effort": effort,
                           "format": {"type": "json_schema", "schema": _claude_schema(schema or SCHEMA)}},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.standard_b64encode(png).decode()}},
                {"type": "text", "text": prompt or PROMPT}]}],
        )
    except anthropic.AuthenticationError:
        raise AiError("Anthropic rejected the Claude API key. Check it in AI settings.")
    except anthropic.PermissionDeniedError as e:
        raise AiError(f"This Claude key isn't allowed to use {model}: {e.message[:200]}")
    except anthropic.RateLimitError:
        raise AiError("Claude's rate limit was reached and didn't clear. Wait a minute and try again, "
                      "or lower Requests per minute in AI settings.")
    except anthropic.BadRequestError as e:
        if "credit" in str(e.message).lower():
            raise AiError("The Claude account has no credit left. Add credit at platform.claude.com "
                          "(Settings → Billing), then run the AI first pass again.")
        raise AiError(f"Claude error 400: {e.message[:300]}")
    except anthropic.APIConnectionError:
        raise AiError("Couldn't reach Claude. Check the internet connection.")
    except anthropic.APIStatusError as e:
        raise AiError(f"Claude error {e.status_code}: {str(e.message)[:300]}")
    usage = {"promptTokenCount": resp.usage.input_tokens, "candidatesTokenCount": resp.usage.output_tokens}
    if resp.stop_reason == "refusal":
        return {"rooms": []}, usage                    # declined this section: skip it, keep the rest
    if resp.stop_reason == "max_tokens":
        raise AiError("Claude's answer for one section was cut off (too long). Try again, or use a Gemini model.")
    text = next((b.text for b in resp.content if b.type == "text"), None)
    try:
        return json.loads(text), usage
    except (TypeError, ValueError):
        raise AiError(f"Claude returned no usable answer ({resp.stop_reason}).")


def read_rooms(page, key=None, model=None, workers=6, call=None):
    """Ask the AI model (Gemini or Claude) about every section of the page.

    Returns (rooms, info): rooms in PDF points as
    {name, number, pt, box, candidate, finish_tag}; info has tile count, tokens and cost.
    `call` can replace the Gemini request (for tests)."""
    st = get_settings()
    model = model or st["model"]
    prov = provider(model)
    key = key or get_key(prov)
    if not key and call is None:
        raise AiError(f"No {provider_name(model)} API key yet. Add one in AI settings.")
    rpm = st["rpm"]
    pacer = _Pacer(rpm)
    if call is None:
        if prov == "claude":
            client = _claude_client(key)
            call = lambda png: _call_claude(client, png, model, pacer=pacer)
        else:
            call = lambda png: _call(key, png, model, pacer=pacer)
    rects = tiles(page)
    jobs = []
    for rect in rects:
        png, w, h = _tile_png(page, rect, PX_PER_PT)
        if png:
            jobs.append((rect, png))

    def run(job):
        rect, png = job
        return rect, call(png)

    # send every section at once, up to the per-minute limit (the pacer holds back the rest)
    workers = max(1, min(len(jobs) or 1, rpm, 24))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(run, jobs))
    seconds = round(time.time() - t0, 1)

    found, tok_in, tok_out, scales = [], 0, 0, []
    for rect, (parsed, usage) in results:
        tok_in += usage.get("promptTokenCount", 0)
        tok_out += usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
        if parsed.get("scale_note"):
            scales.append(parsed["scale_note"])
        x0, y0, x1, y1 = rect
        tw, th = x1 - x0, y1 - y0
        for r in parsed.get("rooms", []):
            try:
                py, px = r["point"][:2]
                b = r.get("box_2d") or [py, px, py, px]
            except (TypeError, ValueError):
                continue
            if not (0 <= px <= 1000 and 0 <= py <= 1000):
                continue
            pt = (x0 + px / 1000 * tw, y0 + py / 1000 * th)
            box = (x0 + b[1] / 1000 * tw, y0 + b[0] / 1000 * th, x0 + b[3] / 1000 * tw, y0 + b[2] / 1000 * th)
            margin = min(px, py, 1000 - px, 1000 - py)        # how far from the section's edge
            poly = []
            for c in r.get("outline") or []:
                try:
                    cy, cx = float(c[0]), float(c[1])
                except (TypeError, ValueError, IndexError):
                    continue
                cy, cx = min(1000, max(0, cy)), min(1000, max(0, cx))
                poly.append((x0 + cx / 1000 * tw, y0 + cy / 1000 * th))
            found.append({"name": (r.get("name") or "").strip().upper(),
                          "number": (r.get("number") or "").strip().upper() or None,
                          "pt": pt, "box": box, "candidate": bool(r.get("takeoff_candidate")),
                          "finish_tag": r.get("finish_tag"), "_margin": margin,
                          "polys": [poly] if len(poly) >= 3 else []})
    rooms = _dedupe(found)
    _to_pdf_frame(page, rooms)
    pr = price(model)
    cost = (tok_in / 1e6 * pr[0] + tok_out / 1e6 * pr[1]) if pr else None
    info = {"model": model, "sections": len(rects), "sent": len(jobs), "requests": pacer.count or len(jobs),
            "rpm": rpm, "tokens_in": tok_in,
            "tokens_out": tok_out, "cost_usd": round(cost, 4) if cost is not None else None, "scale_notes": scales,
            "seconds": seconds}
    print(f"  AI first pass: {model}, {len(jobs)} sections, {seconds}s, {tok_in} tokens in / {tok_out} out"
          + (f", ~${cost:.3f}" if cost is not None else ""), flush=True)
    return rooms, info


def _to_pdf_frame(page, rooms):
    """Sections are cut from the page as it's shown (rotation applied), but the wall tracing
    and the text layer use the PDF's unrotated coordinates: convert everything over."""
    if not page.rotation:
        return
    m = page.derotation_matrix

    def pt(x, y):
        q = pymupdf.Point(x, y) * m
        return (q.x, q.y)
    for r in rooms:
        r["pt"] = pt(*r["pt"])
        if r.get("box"):
            x0, y0, x1, y1 = r["box"]
            a, b = pt(x0, y0), pt(x1, y1)
            r["box"] = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
        r["polys"] = [[pt(x, y) for x, y in poly] for poly in r.get("polys") or []]


def _dedupe(found, near_pt=120):
    """The same room seen in two overlapping sections: keep the sighting furthest from an edge."""
    found = sorted(found, key=lambda r: -r["_margin"])
    out = []
    for r in found:
        if not r["name"] and not r["number"]:
            continue
        dup = None
        for q in out:
            if r["number"] and q["number"]:
                same = r["number"] == q["number"] and math.dist(r["pt"], q["pt"]) < 4 * near_pt
            else:
                same = r["name"] == q["name"] and math.dist(r["pt"], q["pt"]) < near_pt
            if same:
                dup = q
                break
        if dup:
            dup["polys"] += r["polys"]                 # the room's pieces from every section it's in
            dup["candidate"] = dup["candidate"] or r["candidate"]
            dup["finish_tag"] = dup["finish_tag"] or r["finish_tag"]
        else:
            out.append(r)
    for r in out:
        r.pop("_margin", None)
    return out
