"""Room outlines from an AI model, snapped onto the plan's walls.

The model (Gemini with the app's key, or Claude) sees a picture of the plan around a click
and marks the rough outline of the room there: it reads the plan like a person, so it isn't
fooled by tag boxes, furniture or finish lines the way line rules are. The app then moves
every edge onto the wall face drawn in the PDF, so the corners land on the walls exactly;
the model only has to be roughly right.

Gemini uses the key and model from AI settings (config/gemini_key.txt). Claude uses
ANTHROPIC_API_KEY or config/anthropic_key.txt (local only). Plan pictures are sent to the
provider's API.
"""
import base64
import json
import math
import os
import threading
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from .detect import axis_lines, point_in_polygon, polygon_area

ROOT = Path(__file__).resolve().parent.parent
KEY_FILE = ROOT / "config" / "anthropic_key.txt"
MODEL = "claude-opus-5-5"
MAX_PX = 2000                       # long side of the picture sent (Claude reads up to 2576)

PROMPT = """This picture is part of an architectural floor plan, for a flooring takeoff. A red dot inside a red ring marks a point inside one room.

Trace the floor of the room that contains the red dot:
- Walls: follow the inside face of the walls that enclose it (the face on the room's side). Step in and out round pilasters, piers and columns that are part of the walls. Walls are usually drawn as two parallel lines, often filled or hatched.
- Doorways and cased openings: close the outline straight across, in line with the wall.
- Include alcoves, recesses and bays that belong to the room. Ignore free-standing columns inside the room: trace as if they weren't there.
- These are not walls, so never stop at them: furniture and equipment (even when it stands against a wall: go behind it, along the wall), tags and labels (even when they sit on a wall), door swings, dimension lines, grid lines, hatching and floor-pattern lines inside the room, and text.
- Open plan (no wall between this room and the next space): follow the line the plan draws for the room's edge. That is usually a thin line where the floor finish or pattern changes, and it can run at an angle. If no line is drawn, stop where the floor pattern changes, or where the area its label describes ends.

Edges can run at any angle: follow the drawing. Give the corners in order around the room, in this picture's pixel coordinates (x to the right, y down; the numbers along the picture's edges are pixel positions). Set touches_edge to true if the room continues past the edge of the picture.

Also list the room's openings: every place on its border where people walk in or out of the room, such as a doorway (with or without a door drawn), a cased opening, or an open edge onto another space with no wall. For each, give its two ends (the jambs, or where the open edge starts and stops) in the same pixel coordinates. Don't list windows, glazing or storefront panels: only places people can walk through. Give an empty list if there are none."""

SCHEMA = {
    "type": "object",
    "properties": {
        "vertices": {"type": "array", "items": {
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["x", "y"], "additionalProperties": False}},
        "touches_edge": {"type": "boolean"},
        "openings": {"type": "array", "items": {
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/pt"}, "b": {"$ref": "#/$defs/pt"}},
            "required": ["a", "b"], "additionalProperties": False}},
    },
    "$defs": {"pt": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                     "required": ["x", "y"], "additionalProperties": False}},
    "required": ["vertices", "touches_edge", "openings"],
    "additionalProperties": False,
}


def get_key():
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k.strip()
    if KEY_FILE.exists():
        k = KEY_FILE.read_text().strip()
        return k or None
    return None


def _client():
    import anthropic
    key = get_key()
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


def _picture(page, rect, pt, ticks=True, hint=None):
    """PNG of `rect` (PDF points) with the click marked and pixel numbers along the edges, and
    the user's rough outline (`hint`, PDF points) in orange. Returns (png bytes, px per pt)."""
    z = MAX_PX / max(rect.width, rect.height)
    rot = page.rotation
    if rot:                                         # outlines are in unrotated page points
        page.set_rotation(0)
    try:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), clip=rect, alpha=False)
    finally:
        if rot:
            page.set_rotation(rot)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    for v in (range(100, max(W, H), 100) if ticks else ()):   # pixel ticks along the edges
        if v < W:
            cv2.line(img, (v, 0), (v, 10), (200, 120, 0), 1)
            cv2.putText(img, str(v), (v + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 120, 0), 1)
        if v < H:
            cv2.line(img, (0, v), (10, v), (200, 120, 0), 1)
            cv2.putText(img, str(v), (12, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 120, 0), 1)
    if hint:
        P = np.round((np.array(hint, float) - (rect.x0, rect.y0)) * z).astype(np.int32)
        over = img.copy()
        cv2.polylines(over, [P], True, (0, 140, 255), 5)
        img = cv2.addWeighted(over, 0.55, img, 0.45, 0)
    cx, cy = int((pt[0] - rect.x0) * z), int((pt[1] - rect.y0) * z)
    cv2.circle(img, (cx, cy), 14, (0, 0, 230), 2)
    cv2.circle(img, (cx, cy), 5, (0, 0, 230), -1)
    ok, png = cv2.imencode(".png", img)
    return png.tobytes(), z


GEMINI_PROMPT = PROMPT.split("\n\nAlso list the room's openings")[0].replace(
    "in this picture's pixel coordinates (x to the right, y down; the numbers along the picture's edges are pixel positions)",
    "as [y, x] normalized to 0-1000 over this picture")
GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "outline": {"type": "ARRAY", "items": {"type": "ARRAY", "items": {"type": "INTEGER"}}},
        "touches_edge": {"type": "BOOLEAN"},
    },
    "required": ["outline", "touches_edge"],
}


def _ask_gemini(png, W, H):
    """Gemini's answer as {vertices: [{x, y} in pixels], touches_edge}, plus usage."""
    from . import ai
    key = ai.get_key()
    if not key:
        raise RuntimeError("No Gemini key: add one in AI settings.")
    model = os.environ.get("AI_OUTLINE_MODEL") or ai.get_settings()["model"]
    data, use = ai._call(key, png, model=model, prompt=GEMINI_PROMPT,
                         schema=GEMINI_SCHEMA, thinking="high")
    verts = [{"x": p[1] / 1000 * W, "y": p[0] / 1000 * H} for p in data.get("outline") or [] if len(p) == 2]
    return ({"vertices": verts, "touches_edge": bool(data.get("touches_edge"))},
            use.get("promptTokenCount", 0), use.get("candidatesTokenCount", 0) + use.get("thoughtsTokenCount", 0))


def _label_hint(label):
    return (f"\n\nThe room is labelled \"{label}\" on the plan. If the red dot isn't inside that room, "
            "trace the room that carries that label instead.") if label else ""


HINT_TEXT = ("\n\nA thick orange line shows the user's rough outline of this room. It is only a guide: it may cut "
             "corners, sit inside or outside the walls, run diagonally where the walls step, or skip recesses and "
             "pilasters. Use it to know which room is meant and roughly how far it reaches, but trace the actual "
             "wall faces and edge lines of the drawing as described above, never the orange line itself.")


def _ask_claude(client, png, label=None, model=None, hint=False):
    model = model or MODEL
    extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"} if model == "claude-opus-5-5" else {}
    resp = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        **extra,
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.standard_b64encode(png).decode()}},
            {"type": "text", "text": PROMPT + _label_hint(label) + (HINT_TEXT if hint else "")}]}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), None)
    data = json.loads(text) if text and resp.stop_reason != "refusal" else None
    return data, resp.usage.input_tokens, resp.usage.output_tokens


def rough_outline(page, pt, pt_ft, half_ft=45.0, provider="gemini", client=None, log=None, view=None, label=None,
                  model=None, doors=None, hint=None):
    """The model's rough outline (PDF points) of the room at `pt`; grows the view once if the
    room runs off the picture. `view` (x0, y0, x1, y1 in PDF points) frames the picture around
    the whole room when its rough extent is known; otherwise it's a square of 2 × half_ft.
    Returns (points, usage) or (None, usage). If `doors` is a list, the openings the model
    marked (Claude only) are put in it as [a, b] in PDF points."""
    if provider == "claude":
        client = client or _client()
    usage = {"input": 0, "output": 0, "calls": 0}
    for attempt in range(2):
        if view is not None:
            k = 1.6 if attempt else 1.0
            cx, cy, hw, hh = (view[0] + view[2]) / 2, (view[1] + view[3]) / 2, (view[2] - view[0]) / 2 * k, (view[3] - view[1]) / 2 * k
            rect = pymupdf.Rect(cx - hw, cy - hh, cx + hw, cy + hh) & page.rect
        else:
            h = half_ft * pt_ft * (2 if attempt else 1)
            rect = pymupdf.Rect(pt[0] - h, pt[1] - h, pt[0] + h, pt[1] + h) & page.rect
        with _SNAP_LOCK:                            # PyMuPDF pages aren't safe to render from two threads at once
            png, z = _picture(page, rect, pt, ticks=provider == "claude", hint=hint)
        if provider == "claude":
            data, i_tok, o_tok = _ask_claude(client, png, label, model, hint=bool(hint))
        else:
            W, H = int(round(rect.width * z)), int(round(rect.height * z))
            data, i_tok, o_tok = _ask_gemini(png, W, H)
        usage["calls"] += 1
        usage["input"] += i_tok
        usage["output"] += o_tok
        if not data:
            return None, usage
        if log is not None:
            log.append({"rect": [rect.x0, rect.y0, rect.x1, rect.y1], "z": z, "data": data})
        pts = [[rect.x0 + v["x"] / z, rect.y0 + v["y"] / z] for v in data["vertices"]]
        if len(pts) < 3:
            return None, usage
        if doors is not None:
            doors[:] = [[[rect.x0 + o["a"]["x"] / z, rect.y0 + o["a"]["y"] / z],
                         [rect.x0 + o["b"]["x"] / z, rect.y0 + o["b"]["y"] / z]] for o in data.get("openings") or []]
        if not data["touches_edge"] or attempt:
            return pts, usage
    return pts, usage


# ---------------------------------------------------------------- snapping
_SOLID = {}
_SNAP_LOCK = threading.Lock()


def tag_filter(page, pt_ft, max_ft=6.0, min_ft=0.25):
    """A test for line work that belongs to a tag rather than a wall: window and door tags,
    finish tags, detail bubbles, drawn round a bit of text and often sitting right on a wall.
    A thin stroke counts as a tag's if it's short (under max_ft), hugs a word (lies entirely
    within about one text height of it), and that word is enclosed: such strokes sit on all four
    sides of it (a bubble, an oval, a hexagon). Specks (floor pattern dots) are ignored. A line
    beside a label (a step edge, a finish line) stays, and so do walls, which run on past it."""
    from .detect import _drawings
    lim, tiny = max_ft * pt_ft, min_ft * pt_ft
    cell = 50.0
    words, grid = [], {}
    for w in page.get_text("words"):
        m = max(0.4 * pt_ft, 1.0 * min(w[2] - w[0], w[3] - w[1]))     # about one text height
        x0, y0, x1, y1 = w[0] - m, w[1] - m, w[2] + m, w[3] + m
        if (x1 - x0) > 20 * pt_ft or (y1 - y0) > 20 * pt_ft:
            continue
        k = len(words)
        words.append(((w[0] + w[2]) / 2, (w[1] + w[3]) / 2, (x0, y0, x1, y1)))
        for gx in range(int(x0 // cell), int(x1 // cell) + 1):
            for gy in range(int(y0 // cell), int(y1 // cell) + 1):
                grid.setdefault((gx, gy), []).append(k)

    def points(it):
        kind = it[0]
        if kind == "l":
            return [it[1], it[2]]
        if kind == "c":
            return [it[1], it[2], it[3], it[4]]
        if kind == "re":
            r = it[1]
            return [r.tl, r.br, r.tr, r.bl]
        if kind == "qu":
            q = it[1]
            return [q.ul, q.ur, q.ll, q.lr]
        return None

    cand, sides = {}, {}
    for d in _drawings(page):
        if d.get("fill") is not None and d.get("type") == "f":
            continue
        for it in d["items"]:
            pts = points(it)
            if not pts:
                continue
            xs, ys = [q.x for q in pts], [q.y for q in pts]
            span = max(max(xs) - min(xs), max(ys) - min(ys))
            if span > lim:
                continue
            p0 = pts[0]
            for k in grid.get((int(p0.x // cell), int(p0.y // cell)), ()):
                cx, cy, b = words[k]
                if all(b[0] <= q.x <= b[2] and b[1] <= q.y <= b[3] for q in pts):
                    cand.setdefault(id(it), []).append(k)
                    if span < tiny:                   # specks (pattern dots, bits of an arc) don't
                        continue                      # decide whether the word is surrounded
                    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
                    sd = sides.setdefault(k, set())
                    sd.add("l" if mx < cx else "r")
                    sd.add("t" if my < cy else "b")
    enclosed = {k for k, sd in sides.items() if len(sd) == 4}
    tag_items = {i for i, ks in cand.items() if any(k in enclosed for k in ks)}
    return lambda it: id(it) in tag_items

def _wall_masks(page, s, pt_ft):
    """(solid, fine): every grey line on the page (hatching included), and the same with gaps
    up to ~6" closed, so a hatched or narrow double-line wall reads solid. A wall face has
    `solid` just outside it and nothing at all (`fine`) just inside it."""
    from .detect import _wall_mask_vector
    key = (page.parent.name, page.number, round(s, 4), "tags")
    if key not in _SOLID:
        _SOLID.clear()
        fine = _wall_mask_vector(page, s, 0.0, 0.85, 0.5, False, 0, curves=False, skip=tag_filter(page, pt_ft))
        if fine is None:
            _SOLID[key] = (None, None)
        else:
            k = max(3, int(round(6 / 12 * pt_ft * s)) | 1)
            _SOLID[key] = (cv2.morphologyEx(fine, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)), fine)
    return _SOLID[key]


def _face_score(m, raw, s, orient, c, lo, hi, inside_sign, n=24):
    """How cleanly the line at c (an h line y=c or a v line x=c, from lo to hi) reads as a wall
    face with the room on `inside_sign`'s side: the share of samples along it with wall just
    outside (m) and nothing just inside (raw), the weaker of the two. 1.0 = a clean face."""
    H, W = m.shape
    wall = free = 0
    for i in range(n):
        t = lo + (hi - lo) * (i + 0.5) / n
        for off, want in ((-inside_sign * 3, 1), (inside_sign * 3, 0)):
            if orient == "h":
                x, y = int(t * s), int(c * s + off)
            else:
                x, y = int(c * s + off), int(t * s)
            if 0 <= x < W and 0 <= y < H and (m if want else raw)[y, x] == want:
                if want:
                    wall += 1
                else:
                    free += 1
    return min(wall, free) / n


def snap_outline(page, pts, pt_ft, tol_ft=1.75):
    """Move each roughly horizontal/vertical edge of `pts` onto the nearest wall face drawn in
    the PDF (within tol_ft), then rebuild the corners where the edges meet."""
    W_, H_ = page.rect.width, page.rect.height
    s = min(3.0, max(1.0, 30.0 / pt_ft), (40e6 / (W_ * H_)) ** 0.5)
    m, raw = _wall_masks(page, s, pt_ft)
    hs, vs = axis_lines(page, min_len_pt=2)
    tol = tol_ft * pt_ft
    edges = []
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        ang = math.degrees(math.atan2(abs(dy), abs(dx)))
        if ang < 12 or ang > 78:
            orient = "h" if ang < 12 else "v"
            c = (a[1] + b[1]) / 2 if orient == "h" else (a[0] + b[0]) / 2
            lo, hi = sorted((a[0], b[0]) if orient == "h" else (a[1], b[1]))
            # which side is inside: test a point just off the edge's middle
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            probe = 0.5 * pt_ft
            if orient == "h":
                inside = 1 if point_in_polygon(mx, my + probe, pts) else -1
            else:
                inside = 1 if point_in_polygon(mx + probe, my, pts) else -1
            segs = hs if orient == "h" else vs
            best = None
            import bisect
            keys = [q[0] for q in segs]
            for q in segs[bisect.bisect_left(keys, c - tol):bisect.bisect_right(keys, c + tol)]:
                ov = min(hi, q[2]) - max(lo, q[1])
                if ov < 0.25 * (hi - lo):
                    continue
                score = _face_score(m, raw, s, orient, q[0], max(lo, q[1]), min(hi, q[2]), inside) if m is not None else 0
                # the cleanest wall face wins (ties: the one furthest out from the room, as a line
                # just inside a wall, like wainscoting, can read as a face too); with no face at
                # all, the longest overlap, then the nearest
                if score >= 0.6:
                    key = (0, -round(score, 1), q[0] * inside)
                else:
                    key = (1, -round(min(ov, hi - lo) / (0.5 * pt_ft)), abs(q[0] - c))
                if best is None or key < best[0]:
                    best = (key, q[0])
            edges.append([orient, best[1] if best else c, a, b])
        else:
            edges.append(["g", None, a, b])
    merged = []
    for e in edges:
        if merged and e[0] in "hv" and merged[-1][0] == e[0] and abs(merged[-1][1] - e[1]) < 0.5:
            merged[-1][3] = e[3]
        else:
            merged.append(e)
    if len(merged) > 2 and merged[0][0] in "hv" and merged[0][0] == merged[-1][0] and abs(merged[0][1] - merged[-1][1]) < 0.5:
        last = merged.pop()
        merged[0][2] = last[2]
    if len(merged) < 3:
        return pts

    def coef(e):
        if e[0] == "h":
            return 0.0, 1.0, e[1]
        if e[0] == "v":
            return 1.0, 0.0, e[1]
        (xa, ya), (xb, yb) = e[2], e[3]
        A, B = yb - ya, xa - xb
        return A, B, A * xa + B * ya
    out = []
    for i in range(len(merged)):
        p, q = merged[i - 1], merged[i]
        a1, b1, c1 = coef(p)
        a2, b2, c2 = coef(q)
        det = a1 * b2 - a2 * b1
        if abs(det) > 1e-6 * (abs(a1) + abs(b1)) * (abs(a2) + abs(b2)):
            out.append([round((c1 * b2 - c2 * b1) / det, 2), round((a1 * c2 - a2 * c1) / det, 2)])
        else:                                       # two parallel edges in a row: a small jog
            v = q[2]
            if p[0] == "h":
                out += [[round(v[0], 2), round(p[1], 2)], [round(v[0], 2), round(q[1], 2)]]
            elif p[0] == "v":
                out += [[round(p[1], 2), round(v[1], 2)], [round(q[1], 2), round(v[1], 2)]]
            else:
                out.append([round(v[0], 2), round(v[1], 2)])
    if polygon_area(out) < 0.6 * polygon_area(pts) or polygon_area(out) > 1.6 * polygon_area(pts):
        return pts
    return out


_XBOX = {}


def _x_boxes(page, min_pt=6.0, max_pt=400.0):
    """Boxes with both diagonals drawn (an X): the usual sign for a shaft or an opening in the
    floor. Returns [(x0, y0, x1, y1)] in PDF points."""
    key = (page.parent.name, page.number)
    if key in _XBOX:
        return _XBOX[key]
    from .detect import _drawings
    diag = []
    for d in _drawings(page):
        for it in d["items"]:
            if it[0] == "l":
                a, b = it[1], it[2]
                dx, dy = abs(a.x - b.x), abs(a.y - b.y)
                if dx > min_pt and dy > min_pt and dx < max_pt and dy < max_pt:
                    x0_, x1_ = sorted((a.x, b.x))
                    y0_, y1_ = sorted((a.y, b.y))
                    down = (b.y - a.y) * (b.x - a.x) > 0      # "\" vs "/"
                    diag.append((round(x0_, 1), round(y0_, 1), round(x1_, 1), round(y1_, 1), down))
    boxes, seen = [], {}
    for x0_, y0_, x1_, y1_, down in diag:
        k = (round(x0_), round(y0_), round(x1_), round(y1_))
        if k in seen and seen[k] != down:
            boxes.append((x0_, y0_, x1_, y1_))
        seen.setdefault(k, down)
    _XBOX.clear()
    _XBOX[key] = boxes
    return boxes


_SEGS = {}


def _page_segments(page):
    """Every straight drawn line on the page (any colour or weight), as (x0, y0, x1, y1)."""
    key = (page.parent.name, page.number)
    if key not in _SEGS:
        from .detect import _drawings
        out = []
        for d in _drawings(page):
            if d.get("type") == "f":
                continue
            for it in d["items"]:
                if it[0] == "l":
                    out.append((it[1].x, it[1].y, it[2].x, it[2].y))
                elif it[0] == "re":
                    r = it[1]
                    out += [(r.x0, r.y0, r.x1, r.y0), (r.x1, r.y0, r.x1, r.y1), (r.x1, r.y1, r.x0, r.y1), (r.x0, r.y1, r.x0, r.y0)]
        _SEGS.clear()
        _SEGS[key] = np.array(out, float) if out else np.zeros((0, 4))
    return _SEGS[key]


def _zone_lines(page, cuts, pt_ft, max_off_ft=2.5, min_ft=1.5):
    """Drawn lines alongside each open edge (where nothing stopped the fill): in an open plan
    a zone's edge is a thin finish line, not a wall. For each cut, the line running the same way
    within max_off_ft that covers most of it. Returns [(a, b)] in PDF points."""
    S = _page_segments(page)
    if not len(S):
        return []
    v = S[:, 2:] - S[:, :2]
    L = np.hypot(v[:, 0], v[:, 1])
    ok = L >= min_ft * pt_ft
    out = []
    for a, b in cuts:
        a, b = np.array(a, float), np.array(b, float)
        CL = np.linalg.norm(b - a)
        if CL < 2 * pt_ft:
            continue
        u = (b - a) / CL
        nrm = np.array([u[1], -u[0]])
        par = ok & (np.abs(v @ u) / np.maximum(L, 1e-9) > 0.9976)          # within 4 degrees
        if not par.any():
            continue
        P, Q = S[par, :2] - a, S[par, 2:] - a
        off = ((P @ nrm) + (Q @ nrm)) / 2
        t0, t1 = np.minimum(P @ u, Q @ u), np.maximum(P @ u, Q @ u)
        cover = (np.minimum(t1, CL) - np.maximum(t0, 0)) / CL
        cand = (np.abs(off) <= max_off_ft * pt_ft) & (cover >= 0.6)
        if not cand.any():
            continue
        i = np.flatnonzero(cand)[np.argmin(np.abs(off[cand]))]
        o = off[i]
        e = 1.0 * pt_ft                               # run it a foot past each end so it closes
        out.append((tuple(a + nrm * o - u * e), tuple(b + nrm * o + u * e)))
    return out


def fill_outline(page, rough, seed, pt_ft, reach_ft=6.0, deep_ft=5.0, openings=None, notches=None):
    """_fill_once, then again with the drawn finish lines found alongside any open edges, so a
    zone in an open plan stops at its own drawn edge rather than at the model's rough line."""
    cuts = []
    pts = _fill_once(page, rough, seed, pt_ft, reach_ft, deep_ft, openings=cuts)
    if pts is None or not cuts:
        if pts is not None and openings is not None:
            openings.extend(cuts)
        if pts is not None and notches is not None:
            _fill_once(page, rough, seed, pt_ft, reach_ft, deep_ft, notches=notches)
        return pts
    lines = _zone_lines(page, cuts, pt_ft)
    if not lines:
        if openings is not None:
            openings.extend(cuts)
        if notches is not None:
            _fill_once(page, rough, seed, pt_ft, reach_ft, deep_ft, notches=notches)
        return pts
    ops2 = []
    pts2 = _fill_once(page, rough, seed, pt_ft, reach_ft, deep_ft, openings=ops2, notches=notches,
                      extra_lines=lines)
    if pts2 is None or not 0.85 * polygon_area(pts) <= polygon_area(pts2) <= 1.15 * polygon_area(pts):
        if openings is not None:
            openings.extend(cuts)
        return pts
    if openings is not None:
        # the stretches along those finish lines are open edges too (no wall there)
        openings.extend(ops2)
        openings.extend(_along_lines(pts2, lines, pt_ft))
    return pts2


def _along_lines(pts, lines, pt_ft, min_ft=2.5):
    """Stretches of the outline lying on any of the given lines: [[a, b], ...]."""
    out = []
    tol = 0.5 * pt_ft
    for i in range(len(pts)):
        p, q = pts[i], pts[(i + 1) % len(pts)]
        if math.dist(p, q) < min_ft * pt_ft:
            continue
        for a, b in lines:
            if _seg_dist(p, a, b)[0] < tol and _seg_dist(q, a, b)[0] < tol:
                out.append([list(p), list(q)])
                break
    return out


def _fill_once(page, rough, seed, pt_ft, reach_ft=6.0, deep_ft=5.0, openings=None, notches=None, extra_lines=None):
    """The room's exact floor, filled out from the AI's rough outline using the drawn walls.

    The floor is every point not covered by wall line work that can be reached from inside the
    rough outline, so the border follows each wall face: angled walls, piers, pilasters and bays
    included. The fill may spread up to reach_ft past the rough outline (a bay or a gap the model
    cut across); where it would keep going, it's running out through an opening, so that part is
    cut back to the rough outline (the straight line across the doorway). Furniture and tags
    inside are filled over. Returns the outline in PDF points, or None if it doesn't work out.
    If `openings` is a list, the openings are added to it as [a, b] in PDF points: stretches of
    the border where the fill was cut off because the floor carried on past it (doorways,
    openings onto other spaces). Windows and walls stop the fill, so they never count."""
    W_, H_ = page.rect.width, page.rect.height
    s = min(3.0, max(1.0, 30.0 / pt_ft), (40e6 / (W_ * H_)) ** 0.5)
    solid, _fine = _wall_masks(page, s, pt_ft)
    if solid is None or len(rough) < 3:
        return None
    reach = int(round(reach_ft * pt_ft * s))
    P = np.array(rough, float) * s
    x0, y0 = np.floor(P.min(0)).astype(int) - reach - 4
    x1, y1 = np.ceil(P.max(0)).astype(int) + reach + 4
    Hm, Wm = solid.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(Wm, x1), min(Hm, y1)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    wall = (solid[y0:y1, x0:x1] > 0).astype(np.uint8)
    # a solid block thicker than ~3 ft is a shaded zone or a hidden fill, not a wall: ignore it
    kb = max(3, int(round(3.0 * pt_ft * s)) | 1)
    core = cv2.erode(wall, np.ones((kb, kb), np.uint8))
    if core.any():
        # keep a thin rim at its edge: where a wall runs along casework drawn hatched behind it
        # (IA310's PT01 wall), the two close up into one block, and clearing it all let the floor
        # run through the wall
        gone = cv2.dilate(core, np.ones((kb, kb), np.uint8))
        gone = cv2.erode(gone, np.ones((5, 5), np.uint8))
        wall &= ~(gone > 0)
    # (after the step above, so a shaft isn't taken for a hidden fill)
    for bx0, by0, bx1, by1 in _x_boxes(page):         # shafts and floor openings: no floor in them
        if min(bx1 - bx0, by1 - by0) < 2.0 * pt_ft:
            continue                                  # a small X is a symbol (change in floor finish), not a shaft
        X0, Y0, X1, Y1 = int(bx0 * s) - x0, int(by0 * s) - y0, int(bx1 * s) - x0, int(by1 * s) - y0
        if X1 > 0 and Y1 > 0 and X0 < wall.shape[1] and Y0 < wall.shape[0]:
            wall[max(0, Y0):max(0, Y1), max(0, X0):max(0, X1)] = 1
    for la, lb in extra_lines or ():                  # finish lines that edge an open-plan zone
        cv2.line(wall, (int(la[0] * s) - x0, int(la[1] * s) - y0), (int(lb[0] * s) - x0, int(lb[1] * s) - y0), 1, 2)
    wall = wall > 0
    R = np.zeros(wall.shape, np.uint8)
    cv2.fillPoly(R, [np.round(P - (x0, y0)).astype(np.int32)], 1)
    k = 2 * reach + 1
    zone = cv2.dilate(R, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    free = (~wall & zone).astype(np.uint8)
    n, lab = cv2.connectedComponents(free, connectivity=4)
    if n < 2:
        return None
    # the floor pieces: mostly inside the rough outline (a dimension or finish line can split
    # the room into several), plus the piece holding the seed
    inside = np.bincount(lab[R > 0].ravel(), minlength=n)
    total = np.bincount(lab.ravel(), minlength=n)
    keep = (inside > 0.5 * total)
    keep[0] = False
    sx, sy = int(seed[0] * s) - x0, int(seed[1] * s) - y0
    if 0 <= sy < lab.shape[0] and 0 <= sx < lab.shape[1] and lab[sy, sx]:
        keep[lab[sy, sx]] = True
    floor = keep[lab]
    reached = floor.copy()                            # floor reachable before anything is cut off
    # outside the rough outline: keep enclosed pockets, drop what runs to the edge of the zone
    out = (floor & (R == 0)).astype(np.uint8)
    edge = zone & ~(cv2.erode(zone.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    m2, lab2 = cv2.connectedComponents(out, connectivity=4)
    leaks = set(np.unique(lab2[edge & (out > 0)]).tolist()) - {0}
    # a pocket counts as the room's (a bay, a strip the model cut off) only if it opens wide onto
    # the room (no more than ~3 ft deep on average along its contact) and reaches no more than
    # deep_ft past the model's outline. A closet or vestibule behind a door stays out.
    ring = (cv2.dilate(R, np.ones((3, 3), np.uint8)) > 0) & (R == 0)
    px_ft = pt_ft * s
    contact = np.bincount(lab2[ring & (out > 0)].ravel(), minlength=m2) / px_ft
    area = np.bincount(lab2.ravel(), minlength=m2) / px_ft ** 2
    depth = cv2.distanceTransform((R == 0).astype(np.uint8), cv2.DIST_L2, 3)
    deepest = np.zeros(m2)
    np.maximum.at(deepest, lab2.ravel(), depth.ravel())
    for i in range(1, m2):
        through_door = contact[i] < 8.0 and deepest[i] > 1.0 * px_ft      # out through a doorway (single or double)
        if through_door or area[i] > 3.0 * contact[i] + 1.0 or deepest[i] > deep_ft * px_ft:
            leaks.add(i)
    floor &= ~np.isin(lab2, list(leaks))
    # the deeper part of what's left (over 1.5 ft past the model's outline) must also open wide
    # onto the rest of the floor: a door swing behind a thin strip along the wall doesn't
    near = (R > 0) | (floor & (depth <= 1.5 * px_ft))
    deep = (floor & ~near).astype(np.uint8)
    if deep.any():
        m3, lab3d = cv2.connectedComponents(deep, connectivity=4)
        rim = (cv2.dilate(near.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~near
        c3 = np.bincount(lab3d[rim & (deep > 0)].ravel(), minlength=m3) / px_ft
        a3 = np.bincount(lab3d.ravel(), minlength=m3) / px_ft ** 2
        drop = [i for i in range(1, m3) if c3[i] < 8.0 or a3[i] > 3.0 * c3[i] + 1.0]
        floor &= ~np.isin(lab3d, drop)
    # join floor pieces split by lines inside the room (finish or dimension lines), fill the thin
    # gaps along wall lines, and close notches round thin wall stubs and fins; anything narrower
    # than ~12" closes up (pilasters and piers are wider, so the border still steps round them)
    kc = max(3, int(round(12 / 12 * px_ft)) | 1)
    # a square brush: a round one can't reach into square corners and cuts them off at 45°
    floor = cv2.morphologyEx(floor.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kc, kc), np.uint8))
    n3, lab3, stats, _c = cv2.connectedComponentsWithStats(floor, connectivity=4)
    if n3 < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    floor = (lab3 == big).astype(np.uint8)
    if notches is not None:
        # where the border cuts inside the model's outline round something: a pilaster or
        # column, or equipment against the wall. Thin strips along the wall (the model's line
        # a little off the face) are left out; the rest go back as polygons to be checked.
        whole = np.zeros_like(floor)                  # the floor with free-standing things filled over
        cs0, _h0 = cv2.findContours(floor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(whole, cs0, -1, 1, -1)
        shut = ((R > 0) & ~(whole > 0)).astype(np.uint8)
        ko = max(3, int(round(0.5 * px_ft)) | 1)
        shut = cv2.morphologyEx(shut, cv2.MORPH_OPEN, np.ones((ko, ko), np.uint8))
        n5, lab5, st5, _c5 = cv2.connectedComponentsWithStats(shut, connectivity=4)
        for i in range(1, n5):
            if st5[i, cv2.CC_STAT_AREA] / px_ft ** 2 < 0.75:
                continue
            cs5, _h5 = cv2.findContours((lab5 == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs5:
                c5 = cv2.approxPolyDP(max(cs5, key=cv2.contourArea), 2.0, True)[:, 0, :]
                if len(c5) >= 3:
                    notches.append([[round((x + x0 + 0.5) / s, 2), round((y + y0 + 0.5) / s, 2)] for x, y in c5])
    cs, _h = cv2.findContours(floor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # outer edge: furniture filled over
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, max(1.0, 1.0 / 12 * pt_ft * s), True)[:, 0, :]
    pts = [[round((x + x0 + 0.5) / s, 2), round((y + y0 + 0.5) / s, 2)] for x, y in c]
    pts = sharpen_corners(pts, pt_ft)
    if len(pts) < 3:
        return None
    a_out, a_in = polygon_area(pts), polygon_area(rough)
    if not 0.75 * a_in <= a_out <= 1.35 * a_in:
        return None
    cut = reached & ~(floor > 0)
    first = _cut_openings(pts, cut, s, x0, y0, pt_ft)
    if first:                                         # carry the wall's line straight across each doorway
        sq = square_openings(pts, first, pt_ft)
        if 0.97 * polygon_area(pts) <= polygon_area(sq) <= 1.03 * polygon_area(pts):
            pts = sq
    if openings is not None:
        openings.extend(_cut_openings(pts, cut, s, x0, y0, pt_ft))
    return pts


def sharpen_corners(pts, pt_ft, max_in=3.0, min_in=6.0):
    """Tracing a pixel outline leaves a small bevel at some corners. An edge shorter than max_in
    inches between two edges longer than min_in, turning well away from both, becomes the point
    where those two edges' lines meet."""
    pts = [list(p) for p in pts]
    short, long_ = max_in / 12 * pt_ft, min_in / 12 * pt_ft
    i = 0
    while i < len(pts) and len(pts) > 4:
        n = len(pts)
        a, b = pts[i], pts[(i + 1) % n]
        if math.dist(a, b) < short:
            p, q = pts[(i - 1) % n], pts[(i + 2) % n]
            if math.dist(p, a) > long_ and math.dist(b, q) > long_ and not _parallel((p, a), (b, q), 15):
                X = _line_x((p, a), (b, q))
                if X and math.dist(X, a) < 2 * short and math.dist(X, b) < 2 * short:
                    pts[i] = X
                    del pts[(i + 1) % n]
                    continue
        i += 1
    return pts


def _ang(p, q):
    return math.atan2(q[1] - p[1], q[0] - p[0])


def _parallel(e, f, deg=20):
    d = abs(_ang(*e) - _ang(*f)) % math.pi
    return min(d, math.pi - d) < math.radians(deg)


def _line_x(e, f):
    """Where the lines through edges e and f cross, or None if they're parallel."""
    (x1, y1), (x2, y2) = e
    (x3, y3), (x4, y4) = f
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return [round(x1 + t * (x2 - x1), 2), round(y1 + t * (y2 - y1), 2)]


def square_openings(pts, cuts, pt_ft, long_ft=1.5, reach=4):
    """The border across a doorway follows the model's rough line, which can be a few degrees
    off the wall and step round the jamb. For each edge that crosses an opening: find the wall
    on each side (the nearest edge over long_ft), take the one running the same way as the
    opening as the wall line, and carry it straight across to the other wall: to where the two
    walls' lines meet, or straight on to the next stretch of the same wall. The vertices in
    between (the jamb steps) go."""
    pts = [list(p) for p in pts]
    for ca, cb in cuts:
        n = len(pts)
        if n < 4:
            break
        m = ((ca[0] + cb[0]) / 2, (ca[1] + cb[1]) / 2)
        i = min(range(n), key=lambda k: _seg_dist(m, pts[k], pts[(k + 1) % n])[0])
        e = (pts[i], pts[(i + 1) % n])
        if _seg_dist(m, *e)[0] > 0.5 * pt_ft:
            continue
        L = long_ft * pt_ft
        prev = next(((k - 1) % n for k in (i - j for j in range(reach)) if math.dist(pts[(k - 1) % n], pts[k % n]) >= L and (k - 1) % n != i), None)
        nxt = next(((i + 1 + j) % n for j in range(0, reach) if math.dist(pts[(i + 1 + j) % n], pts[(i + 2 + j) % n]) >= L), None)
        if prev is None or nxt is None:
            continue
        P = (pts[prev], pts[(prev + 1) % n])
        N = (pts[nxt], pts[(nxt + 1) % n])
        par_p, par_n = _parallel(P, e), _parallel(N, e)
        if par_p and par_n:
            # a doorway in a straight wall: join the two stretches if they're on one line
            far = [P[0][0] + 1e3 * math.cos(_ang(*P)), P[0][1] + 1e3 * math.sin(_ang(*P))]
            off, t = _seg_dist(N[0], P[0], far)
            if off > 0.5 * pt_ft:
                continue
            keep_from, keep_to, new = (prev + 1) % n, nxt, []
            if off > 2 / 12 * pt_ft:                  # the two sides don't line up: carry P's line on
                X = [round(P[0][0] + (far[0] - P[0][0]) * t, 2), round(P[0][1] + (far[1] - P[0][1]) * t, 2)]
                out, k = [], keep_to
                while True:
                    out.append(pts[k])
                    if k == keep_from:
                        break
                    k = (k + 1) % n
                    if len(out) > n:
                        out = None
                        break
                if out is None or len(out) < 3:
                    continue
                out[-1] = X                           # P's end moves along its line to opposite N's start,
                pts = out                             # then the border steps square onto N
                continue
        elif par_p or par_n:
            X = _line_x(P, N)                         # carry the wall's line on to meet the other wall
            if X is None or min(math.dist(X, pts[i]), math.dist(X, pts[(i + 1) % n])) > 4 * pt_ft:
                continue
            keep_from, keep_to, new = (prev + 1) % n, nxt, [X]
        else:
            continue
        # rebuild: ... P[0], (new), N[0] ...  dropping the vertices strictly between
        out, k = [], keep_to
        while True:
            out.append(pts[k])
            if k == keep_from:
                break
            k = (k + 1) % n
            if len(out) > n:
                out = None
                break
        if out is None or len(out) < 3:
            continue
        if new:
            out[-1] = new[0]                          # P's far end moves to the meeting point ...
            out[0] = new[0]                           # ... and so does N's near end
            out = out[:-1]
        pts = out
    return pts


def _cut_openings(pts, cut, s, x0, y0, pt_ft, step_ft=0.25, min_ft=2.5):
    """Stretches of the border with cut-off floor just outside them: [[a, b], ...] in PDF points."""
    H, W = cut.shape
    near = cv2.dilate(cut.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    out = []
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        L = math.dist(a, b)
        if L < min_ft * pt_ft:
            continue
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        nx, ny = uy, -ux                              # outward: the side that isn't inside the room
        if point_in_polygon(mx + nx * 0.5 * pt_ft, my + ny * 0.5 * pt_ft, pts):
            nx, ny = -nx, -ny
        n = max(2, int(L / (step_ft * pt_ft)))
        flags = []
        for k in range(n + 1):
            t = k / n
            px, py = a[0] + (b[0] - a[0]) * t + nx * 0.6 * pt_ft, a[1] + (b[1] - a[1]) * t + ny * 0.6 * pt_ft
            X, Y = int(px * s) - x0, int(py * s) - y0
            flags.append(0 <= X < W and 0 <= Y < H and bool(near[Y, X]))
        k = 0
        while k <= n:
            if not flags[k]:
                k += 1
                continue
            j = k
            while j + 1 <= n and flags[j + 1]:
                j += 1
            ta, tb = k / n, j / n
            if (tb - ta) * L >= min_ft * pt_ft:
                out.append([[round(a[0] + (b[0] - a[0]) * ta, 2), round(a[1] + (b[1] - a[1]) * ta, 2)],
                            [round(a[0] + (b[0] - a[0]) * tb, 2), round(a[1] + (b[1] - a[1]) * tb, 2)]])
            k = j + 1
    return out


def _seg_dist(p, a, b):
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    L2 = (bx - ax) ** 2 + (by - ay) ** 2 or 1e-9
    t = max(0.0, min(1.0, ((p[0] - ax) * (bx - ax) + (p[1] - ay) * (by - ay)) / L2))
    return math.hypot(ax + t * (bx - ax) - p[0], ay + t * (by - ay) - p[1]), t


def merge_segments(segs, pt_ft, gap_ft=1.5, off_ft=1.0):
    """Join stretches that lie along the same line and touch or nearly touch (an opening found
    in two pieces, or found by two methods)."""
    segs = [[list(a), list(b)] for a, b in segs if math.dist(a, b) > 0.1]
    changed = True
    while changed:
        changed = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                a, b = segs[i]
                c, d = segs[j]
                L = math.dist(a, b)
                ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
                vx, vy = d[0] - c[0], d[1] - c[1]
                M = math.hypot(vx, vy)
                if abs(ux * vx + uy * vy) / M < 0.985:          # not parallel (over ~10°)
                    continue
                off = lambda p: abs((p[0] - a[0]) * uy - (p[1] - a[1]) * ux)
                if max(off(c), off(d)) > off_ft * pt_ft:
                    continue
                t = lambda p: (p[0] - a[0]) * ux + (p[1] - a[1]) * uy
                tc, td = sorted((t(c), t(d)))
                if tc > L + gap_ft * pt_ft or td < -gap_ft * pt_ft:
                    continue
                lo, hi = min(0.0, tc), max(L, td)
                segs[i] = [[round(a[0] + ux * lo, 2), round(a[1] + uy * lo, 2)], [round(a[0] + ux * hi, 2), round(a[1] + uy * hi, 2)]]
                segs.pop(j)
                changed = True
                break
            if changed:
                break
    return segs


def _onto_outline(seg, outline, pt_ft, tol_ft=3.0):
    """A marked opening laid onto the outline: each end moved to the nearest point on the border
    (the ends may land on different edges, e.g. where the border steps round the jambs), or None
    if either end is more than tol_ft from it."""
    da, db = seg
    ends = []
    for e in seg:
        best = None
        for k in range(len(outline)):
            p, q = outline[k], outline[(k + 1) % len(outline)]
            d, t = _seg_dist(e, p, q)
            if best is None or d < best[0]:
                best = (d, [round(p[0] + (q[0] - p[0]) * t, 2), round(p[1] + (q[1] - p[1]) * t, 2)])
        if best is None or best[0] > tol_ft * pt_ft:
            return None
        ends.append(best[1])
    if math.dist(*ends) < 2.0 * pt_ft:
        return None
    # the ends can land on different edges; if the line between them runs at an angle to the
    # wall the opening is on, lay both on that wall's line instead (no slanted openings)
    m = ((da[0] + db[0]) / 2, (da[1] + db[1]) / 2)
    k = min(range(len(outline)), key=lambda k: _seg_dist(m, outline[k], outline[(k + 1) % len(outline)])[0])
    p, q = outline[k], outline[(k + 1) % len(outline)]
    if math.dist(p, q) > 1.0 * pt_ft and not _parallel((ends[0], ends[1]), (p, q), 5):
        L = math.dist(p, q)
        ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
        ts = sorted((e[0] - p[0]) * ux + (e[1] - p[1]) * uy for e in (da, db))
        ends = [[round(p[0] + ux * t, 2), round(p[1] + uy * t, 2)] for t in ts]
        if math.dist(*ends) < 2.0 * pt_ft:
            return None
    return ends

def opening_candidates(page, outline, cuts, doors, pt_ft):
    """Every stretch of the border that might be a way in: where the floor fill was cut off at
    a gap, what the model marked, and edges with no wall behind them (access.py). Merged, so an
    opening found in pieces or by several methods is one stretch."""
    from .access import find_access_points
    segs = list(cuts or [])
    for d in doors or []:
        q = _onto_outline(d, outline, pt_ft)
        if q:
            segs.append(q)
    try:
        segs += [[x["a"], x["b"]] for x in find_access_points(page, [{"outline_pt": outline}], pt_ft)]
    except Exception:
        pass
    return [s_ for s_ in merge_segments(segs, pt_ft) if math.dist(*s_) >= 2.0 * pt_ft]


CHECK_MODEL = "claude-haiku-5-5"      # close-up yes/no on each possible opening: cheap and quick
CHECK_PROMPT = """This is a close-up of an architectural floor plan. The thick translucent magenta line marks a stretch of the border of a room being measured for flooring.

What is along that magenta stretch? Look at the drawing under and just beside it:
- "door": a doorway (a door leaf, a door swing arc, or a gap between two wall ends / jambs)
- "open": an open edge with no wall, where the floor carries on into another space (a corridor, a lobby)
- "window": glazing, a storefront, a curtain wall or window frames (thin parallel lines set in the wall, mullions), with no door
- "wall": a solid wall
- "other": anything else

walkable is true only if people can walk through that stretch (door or open)."""
CHECK_SCHEMA = {"type": "object", "properties": {
    "kind": {"type": "string", "enum": ["door", "open", "window", "wall", "other"]},
    "walkable": {"type": "boolean"}}, "required": ["kind", "walkable"], "additionalProperties": False}


def _check_picture(page, seg, pt_ft, px=900):
    a, b = seg
    L = math.dist(a, b)
    m = max(8 * pt_ft, 0.35 * L)
    rect = pymupdf.Rect(min(a[0], b[0]) - m, min(a[1], b[1]) - m, max(a[0], b[0]) + m, max(a[1], b[1]) + m) & page.rect
    z = px / max(rect.width, rect.height)
    with _SNAP_LOCK:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), clip=rect, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    over = img.copy()
    P = lambda p: (int((p[0] - rect.x0) * z), int((p[1] - rect.y0) * z))
    cv2.line(over, P(a), P(b), (255, 0, 255), max(6, int(0.6 * pt_ft * z)))
    img = cv2.addWeighted(over, 0.45, img, 0.55, 0)
    ok, png = cv2.imencode(".png", img)
    return png.tobytes()


def check_openings(page, cands, pt_ft, client, model=CHECK_MODEL):
    """Ask the model, one close-up per candidate, whether people can walk through it.
    Returns (kept [[a, b]...], [kind per candidate], (tokens in, tokens out))."""
    from concurrent.futures import ThreadPoolExecutor

    def one(seg):
        png = _check_picture(page, seg, pt_ft)
        try:
            resp = client.messages.create(
                model=model, max_tokens=4000,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": CHECK_SCHEMA}},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                 "data": base64.standard_b64encode(png).decode()}},
                    {"type": "text", "text": CHECK_PROMPT}]}])
        except Exception as e:                       # can't check it: keep it, the user decides
            print(f"  opening check failed: {type(e).__name__}: {e}", flush=True)
            return {"kind": "unchecked", "walkable": True}, 0, 0
        text = next((b.text for b in resp.content if b.type == "text"), None)
        try:
            data = json.loads(text) if resp.stop_reason not in ("refusal", "max_tokens") else None
        except (TypeError, ValueError):
            data = None
        return data or {"kind": "unchecked", "walkable": True}, resp.usage.input_tokens, resp.usage.output_tokens

    if not cands:
        return [], [], (0, 0)
    with ThreadPoolExecutor(max_workers=min(8, len(cands))) as ex:
        res = list(ex.map(one, cands))
    kept = [seg for seg, (d, _i, _o) in zip(cands, res) if d.get("walkable")]
    return kept, [d.get("kind") for d, _i, _o in res], (sum(r[1] for r in res), sum(r[2] for r in res))


NOTCH_PROMPT = """This is a close-up of an architectural floor plan. The magenta outline marks a spot where the measured floor of a room stops short: the border goes round whatever is drawn there.

What is drawn inside the magenta outline?
- "structure": part of the building: a wall, pilaster, pier, column or column wrap, shaft, chase, or a built-in enclosure. Structure is usually drawn solid, grey-filled or hatched like the walls, and is joined to the wall lines.
- "equipment": something standing on the floor against the wall: a rack, shelf, bench, locker, cabinet, machine, storage, kettlebell or dumbbell rack, or any furniture. Equipment is usually drawn as an outline with empty paper or details inside, often with a label.
- "unsure": can't tell

If anything inside the outline is solid, grey-filled or hatched like the walls, answer "structure"."""
NOTCH_SCHEMA = {"type": "object", "properties": {"kind": {"type": "string", "enum": ["structure", "equipment", "unsure"]}},
                "required": ["kind"], "additionalProperties": False}


def check_notches(page, notches, pt_ft, client, model=CHECK_MODEL):
    """One close-up per notch: is it structure (keep the border round it) or equipment standing
    on the floor (the floor runs under it)? Returns ([kind per notch], (tokens in, tokens out))."""
    from concurrent.futures import ThreadPoolExecutor

    def one(poly):
        P = np.array(poly, float)
        (a0, b0), (a1, b1) = P.min(0), P.max(0)
        m = max(5 * pt_ft, 0.6 * max(a1 - a0, b1 - b0))
        rect = pymupdf.Rect(a0 - m, b0 - m, a1 + m, b1 + m) & page.rect
        z = 900 / max(rect.width, rect.height)
        with _SNAP_LOCK:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), clip=rect, alpha=False)
        img = cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy(),
                           cv2.COLOR_RGB2BGR)
        # outline only: a fill would hide the grey or hatching that marks structure
        cv2.polylines(img, [np.round((P - (rect.x0, rect.y0)) * z).astype(np.int32)], True, (255, 0, 255), 3)
        png = cv2.imencode(".png", img)[1].tobytes()
        try:
            resp = client.messages.create(
                model=model, max_tokens=4000,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": NOTCH_SCHEMA}},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                 "data": base64.standard_b64encode(png).decode()}},
                    {"type": "text", "text": NOTCH_PROMPT}]}])
            text = next((b.text for b in resp.content if b.type == "text"), None)
            kind = json.loads(text)["kind"] if resp.stop_reason not in ("refusal", "max_tokens") else "unsure"
            return kind, resp.usage.input_tokens, resp.usage.output_tokens
        except Exception as e:                       # can't check it: leave the border as traced
            print(f"  notch check failed: {type(e).__name__}: {e}", flush=True)
            return "unsure", 0, 0

    if not notches:
        return [], (0, 0)
    with ThreadPoolExecutor(max_workers=min(8, len(notches))) as ex:
        res = list(ex.map(one, notches))
    return [r[0] for r in res], (sum(r[1] for r in res), sum(r[2] for r in res))


def add_areas(outline, polys, pt_ft):
    """The outline with the given touching areas joined in (floor under equipment)."""
    if not polys:
        return outline
    P = np.array(outline + [p for q in polys for p in q], float)
    x0, y0 = P.min(0) - 2 * pt_ft
    span = max(P.max(0) - P.min(0)) + 4 * pt_ft
    s = min(4.0, 3000.0 / span)
    H, W = (np.ceil((P.max(0) - P.min(0) + 4 * pt_ft) * s).astype(int) + 2)[::-1]
    m = np.zeros((H, W), np.uint8)
    T = lambda q: np.round((np.array(q, float) - (x0, y0)) * s).astype(np.int32)
    cv2.fillPoly(m, [T(outline)], 1)
    for q in polys:
        cv2.fillPoly(m, [T(q)], 1)
    k = max(3, int(round(1.0 * pt_ft * s)) | 1)       # and close the equipment's own thin lines
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cs, _h = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return outline
    c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), max(1.0, 1.0 / 12 * pt_ft * s), True)[:, 0, :]
    out = sharpen_corners([[round(x / s + x0, 2), round(y / s + y0, 2)] for x, y in c], pt_ft)
    a0 = polygon_area(outline)
    return out if len(out) >= 3 and a0 <= polygon_area(out) <= 1.1 * a0 else outline


def cut_at_openings(outline, openings, seed, pt_ft, min_sf=2.0):
    """Trim floor that bulges out past a confirmed opening (into a door swing or a vestibule):
    the outline is cut along each opening's jamb-to-jamb line and only the part holding the room
    is kept. Returns the outline unchanged when nothing comes off."""
    if not openings or len(outline) < 3:
        return outline
    P = np.array(outline, float)
    x0, y0 = P.min(0) - 2 * pt_ft
    span = max(P.max(0) - P.min(0)) + 4 * pt_ft
    s = min(4.0, 3000.0 / span)
    H, W = (np.ceil((P.max(0) - P.min(0) + 4 * pt_ft) * s).astype(int) + 2)[::-1]
    m = np.zeros((H, W), np.uint8)
    T = lambda q: np.round((np.array(q, float) - (x0, y0)) * s).astype(np.int32)
    cv2.fillPoly(m, [T(outline)], 1)
    for a, b in openings:
        L = math.dist(a, b) or 1
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        e = 0.75 * pt_ft                              # run a little past each jamb so the cut closes
        A_ = (a[0] - ux * e, a[1] - uy * e)
        B_ = (b[0] + ux * e, b[1] + uy * e)
        cv2.line(m, tuple(T(A_)), tuple(T(B_)), 0, 2)
    n, lab, stats, _c = cv2.connectedComponentsWithStats(m, connectivity=4)
    if n <= 2:
        return outline
    sx, sy = T(seed)
    main = lab[sy, sx] if 0 <= sy < H and 0 <= sx < W and lab[sy, sx] else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    lost = [i for i in range(1, n) if i != main and stats[i, cv2.CC_STAT_AREA] / s ** 2 / pt_ft ** 2 >= min_sf]
    if not lost:
        return outline
    keep = (lab == main).astype(np.uint8)
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))   # heal the cut lines themselves
    cs, _h = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return outline
    c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), max(1.0, 1.0 / 12 * pt_ft * s), True)[:, 0, :]
    out = [[round(x / s + x0, 2), round(y / s + y0, 2)] for x, y in c]
    if len(out) < 3 or polygon_area(out) < 0.85 * polygon_area(outline):
        return outline
    return out


REVIEW_PROMPT = """This picture is part of an architectural floor plan, for a flooring takeoff. The red dot marks a point inside one room{label}. The thick magenta line is our outline of that room's floor. Check it against the drawing:
- All the way round, it should follow the inside face of the room's walls, stepping round pilasters, piers and columns that are part of the walls, and straight across doorways in line with the wall.
- Where the room is open to another space with no wall, it should follow the line the plan draws for the room's edge (usually a thin line where the floor finish changes; it can run at an angle). If no line is drawn, the edge of the room's floor finish.
- Equipment, furniture, benches, racks and lockers stand on the floor: where they're against a wall, the outline should run behind them along the wall, not round them.
- Tags, labels, dimension lines, door swings and grid lines are not walls.

If the outline is right everywhere (within about 6 inches), set ok to true, leave problems and vertices empty. Otherwise set ok to false, say briefly what is wrong in problems, and give the whole corrected outline in vertices: every corner in order around the room, in this picture's pixel coordinates (x to the right, y down; the numbers along the picture's edges are pixel positions)."""
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "problems": {"type": "array", "items": {"type": "string"}},
        "vertices": {"type": "array", "items": {
            "type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["x", "y"], "additionalProperties": False}},
    },
    "required": ["ok", "problems", "vertices"],
    "additionalProperties": False,
}


def review_outline(page, outline, pt, rect, client, label=None, model=None):
    """Show the model the fitted outline drawn over the plan and ask whether it's right.
    Returns (corrected outline in PDF points or None, problems, tokens in, tokens out)."""
    with _SNAP_LOCK:
        png, z = _picture(page, rect, pt)
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    P = np.round((np.array(outline, float) - (rect.x0, rect.y0)) * z).astype(np.int32)
    cv2.polylines(img, [P], True, (255, 0, 255), 3)
    png = cv2.imencode(".png", img)[1].tobytes()
    model = model or MODEL
    extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"} if model == "claude-opus-5-5" else {}
    resp = client.beta.messages.create(
        model=model, max_tokens=16000, **extra,
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": REVIEW_SCHEMA}},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.standard_b64encode(png).decode()}},
            {"type": "text", "text": REVIEW_PROMPT.format(label=f' (labelled "{label}")' if label else "")}]}])
    text = next((b.text for b in resp.content if b.type == "text"), None)
    try:
        data = json.loads(text) if resp.stop_reason not in ("refusal", "max_tokens") else None
    except (TypeError, ValueError):
        data = None
    i_tok, o_tok = resp.usage.input_tokens, resp.usage.output_tokens
    if not data or data.get("ok") or len(data.get("vertices") or []) < 3:
        return None, (data or {}).get("problems") or [], i_tok, o_tok
    pts = [[rect.x0 + v["x"] / z, rect.y0 + v["y"] / z] for v in data["vertices"]]
    return pts, data.get("problems") or [], i_tok, o_tok


def snap_to_lines(page, pts, pt_ft, tol_ft=1.0):
    """Precision only: move each edge of the model's outline onto the drawn line running the same
    way within tol_ft of it (any line: wall face, finish line; any angle), then rebuild each
    corner where neighbouring edges' lines meet. Edges with no line nearby stay where they are."""
    S = _page_segments(page)
    if not len(S) or len(pts) < 3:
        return pts
    v = S[:, 2:] - S[:, :2]
    Ls = np.hypot(v[:, 0], v[:, 1])
    lines = []
    n = len(pts)
    for i in range(n):
        a, b = np.array(pts[i], float), np.array(pts[(i + 1) % n], float)
        L = np.linalg.norm(b - a)
        if L < 1e-6:
            lines.append((a, b))
            continue
        u = (b - a) / L
        nrm = np.array([u[1], -u[0]])
        par = (Ls > 0.5 * pt_ft) & (np.abs(v @ u) / np.maximum(Ls, 1e-9) > 0.9976)
        best = None
        if par.any():
            P, Q = S[par, :2] - a, S[par, 2:] - a
            off = ((P @ nrm) + (Q @ nrm)) / 2
            t0, t1 = np.minimum(P @ u, Q @ u), np.maximum(P @ u, Q @ u)
            cover = (np.minimum(t1, L) - np.maximum(t0, 0)) / L
            ok = (np.abs(off) <= tol_ft * pt_ft) & (cover >= 0.5)
            if ok.any():
                k = np.flatnonzero(ok)[np.argmin(np.abs(off[ok]))]
                best = off[k]
        lines.append((a + nrm * best, b + nrm * best) if best is not None else (a, b))
    out = []
    for i in range(n):
        e, f = lines[i - 1], lines[i]
        X = _line_x((tuple(e[0]), tuple(e[1])), (tuple(f[0]), tuple(f[1])))
        p0 = pts[i]
        out.append(X if X and math.dist(X, p0) < 2 * tol_ft * pt_ft else [round(float(f[0][0]), 2), round(float(f[0][1]), 2)])
    return out


def _overlap(p, q):
    """Share of the two outlines' combined area they have in common (rasterised)."""
    A_ = np.array(p + q, float)
    x0, y0 = A_.min(0)
    s = 800.0 / max(1e-6, max(A_.max(0) - A_.min(0)))
    W, H = (np.ceil((A_.max(0) - A_.min(0)) * s).astype(int) + 4)
    m1 = np.zeros((H, W), np.uint8)
    m2 = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m1, [np.round((np.array(p) - (x0, y0)) * s).astype(np.int32)], 1)
    cv2.fillPoly(m2, [np.round((np.array(q) - (x0, y0)) * s).astype(np.int32)], 1)
    u = (m1 | m2).sum()
    return (m1 & m2).sum() / u if u else 0.0


def _fit(page, rough, pt, pt_ft, client):
    """Fit a rough outline onto the drawing: the floor between the walls (fill_outline), then
    equipment notches checked (Claude only). Returns (outline, method, cuts, check usage)."""
    with _SNAP_LOCK:                                # the wall masks are cached per page
        cuts, notches = [], []
        out = fill_outline(page, rough, pt, pt_ft, openings=cuts, notches=notches)
        how = "filled"
        if out is None:                             # no vector walls (a scan), or the fill went wrong
            out, how, cuts, notches = snap_outline(page, rough, pt_ft), "snapped", [], []
    ci = co = 0
    kinds = []
    if client is not None and notches:              # where the border cut in round something:
        kinds, (ci, co) = check_notches(page, notches, pt_ft, client)   # structure stays, equipment is floor
        out = add_areas(out, [q for q, k in zip(notches, kinds) if k == "equipment"], pt_ft)
    return out, how, cuts, {"check_input": ci, "check_output": co, "notches": len(notches)}


def inside_point(poly):
    """A point well inside a polygon (farthest from its edges on a grid): where the floor fill
    starts for an outline the user drew."""
    xs, ys = [x for x, _ in poly], [y for _, y in poly]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    best, bd = None, -1.0
    n = 24
    for i in range(1, n):
        for j in range(1, n):
            x, y = x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * j / n
            if not point_in_polygon(x, y, poly):
                continue
            d = min(_seg_dist((x, y), poly[k - 1], poly[k])[0] for k in range(len(poly)))
            if d > bd:
                best, bd = [x, y], d
    return best


def room_at(page, pt, pt_ft, provider="gemini", client=None, log=None, view=None, label=None, model=None,
            review=True, rough=None):
    """{outline_pt, area_sf, rough_pt, usage, method, openings_pt, review} for the room at `pt`, or None.
    With Claude: trace (rough_outline), fit onto the drawing (_fit), then Claude reviews the
    fitted outline drawn over the plan and corrects it if needed, and the correction is fitted
    again. Openings are found and checked on the final outline. openings_pt is None for Gemini."""
    claude = provider == "claude"
    doors = [] if claude else None
    if claude:
        client = client or _client()
    log = log if log is not None else []
    hint = None
    if rough is not None:
        # the user drew a rough outline: it frames the picture and is shown to Claude as a guide
        # to which room and how far it reaches; Claude still traces the walls. Taken literally
        # (snapped onto the walls) it kept every slanted or sloppy edge as drawn.
        hint = rough
        xs, ys = [x for x, _ in rough], [y for _, y in rough]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        pad = max(10 * pt_ft, 0.15 * max(x1 - x0, y1 - y0))
        view = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    rough, usage = rough_outline(page, pt, pt_ft, provider=provider, client=client, log=log, view=view,
                                 label=label, model=model, doors=doors, hint=hint)
    if not rough:
        return None
    fit = _fit
    out, how, cuts, chk = fit(page, rough, pt, pt_ft, client if claude else None)
    usage["check_input"], usage["check_output"] = chk["check_input"], chk["check_output"]
    rev = None
    if claude and review and log:
        rect = pymupdf.Rect(*log[-1]["rect"])
        try:
            fixed, problems, ri, ro = review_outline(page, out, pt, rect, client, label, model)
        except Exception as e:                       # the review failing keeps the fitted outline
            print(f"  review failed: {type(e).__name__}: {e}", flush=True)
            fixed, problems, ri, ro = None, [], 0, 0
        usage["input"] += ri
        usage["output"] += ro
        usage["calls"] += 1
        rev = {"changed": bool(fixed), "problems": problems, "before_pt": out}
        if fixed:
            out2, how2, cuts2, chk2 = fit(page, fixed, pt, pt_ft, client)
            usage["check_input"] += chk2["check_input"]
            usage["check_output"] += chk2["check_output"]
            if out2 and _overlap(out2, fixed) >= 0.97:
                out, how, cuts, rough = out2, how2, cuts2, fixed
            else:
                # the fit would undo the correction: keep Claude's corrected outline, only
                # snapped onto the drawn lines beside its edges
                with _SNAP_LOCK:
                    out, how, cuts, rough = snap_to_lines(page, fixed, pt_ft), "reviewed", [], fixed
                rev["kept_review"] = True
    ops = cands = kinds = None
    if claude:                                       # every possible opening, each checked close up
        with _SNAP_LOCK:
            cands = opening_candidates(page, out, cuts, doors, pt_ft)
        ops, kinds, (ci, co) = check_openings(page, cands, pt_ft, client)
        usage["check_input"] += ci
        usage["check_output"] += co
        usage["checks"] = len(cands)
        out = cut_at_openings(out, ops, pt, pt_ft)
    return {"outline_pt": out, "rough_pt": rough, "area_sf": round(polygon_area(out) / pt_ft ** 2, 1),
            "usage": usage, "method": how, "openings_pt": ops, "doors_pt": doors,
            "candidates_pt": cands, "candidate_kinds": kinds, "review": rev}
