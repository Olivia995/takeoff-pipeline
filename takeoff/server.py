"""Local web app for the whole pipeline: upload a plan, detect rooms, pick rooms,
build the takeoff sheet. Runs on this computer only (127.0.0.1).

    python3 -m takeoff serve            # opens http://127.0.0.1:8765 in your browser
"""
import datetime
import json
import mimetypes
import re
import subprocess
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pymupdf
import yaml

from .detect import render_page, room_area_pt
import math

from . import ai, engine
from .rooms import find_rooms, room_at
from .scale import find_plan_scale, pt_per_ft, scale_from_dimensions
from .sheet import render_sheet

ROOT = Path(__file__).resolve().parent.parent
APP_HTML = ROOT / "app" / "index.html"
PLANS = ROOT / "plans"
JOBS = ROOT / "jobs"
CATALOG = ROOT / "config" / "products.yaml"
COMPANY = ROOT / "config" / "company.yaml"


class ApiError(Exception):
    def __init__(self, msg, status=HTTPStatus.BAD_REQUEST):
        super().__init__(msg)
        self.status = status


def _safe_name(name, default="plan"):
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "", Path(name).stem).strip().replace(" ", "-")
    stem = re.sub(r"-{2,}", "-", stem)[:80]
    return stem or default


def _plan_path(name):
    p = (PLANS / Path(name).name).resolve()
    if p.parent != PLANS.resolve() or p.suffix.lower() != ".pdf" or not p.exists():
        raise ApiError(f"Plan not found: {name}", HTTPStatus.NOT_FOUND)
    return p


def _open_page(plan, page_no):
    doc = pymupdf.open(_plan_path(plan))
    if not 1 <= page_no <= len(doc):
        raise ApiError(f"{plan} has {len(doc)} page(s); page {page_no} is out of range.")
    page = doc[page_no - 1]
    if page.rotation:
        page.set_rotation(0)
    return doc, page


def _render_scale(page):
    # Same formula as detect.render_page, so browser pixels = PDF points * scale.
    w, h = page.rect.width, page.rect.height
    return min(1.5, (10e6 / (w * h)) ** 0.5)


# ---------------------------------------------------------------- API
def api_catalog(_q, _body):
    cat = engine.load()
    return {"products": engine.selectable(cat), "manual_notes": engine.manual_notes(cat)}


def _engine_rooms(rooms, ppf):
    """Rooms from the app (outline + columns in PDF points, layers top first) -> engine input."""
    out = []
    for r in rooms:
        rings = [r["outline_pt"]] + list(r.get("holes_pt") or [])
        perim = sum(math.dist(ring[i - 1], ring[i]) for ring in rings for i in range(len(ring)))
        r["area_sf"] = room_area_pt(r) / ppf ** 2
        r["perimeter_lf"] = perim / ppf
        r["_ppf"] = ppf
        out.append(r)
    return out


def api_plans(_q, _body):
    PLANS.mkdir(exist_ok=True)
    files = sorted(PLANS.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"plans": [p.name for p in files]}


def api_open(q, _body):
    plan, page_no = q["plan"], int(q.get("page", 1))
    doc, page = _open_page(plan, page_no)
    ipf, label = find_plan_scale(page)
    source = "sheet"
    if not ipf:
        ipf, label = scale_from_dimensions(page)
        source = "dimensions"
    s = _render_scale(page)
    return {"plan": plan, "page": page_no, "pages": len(doc),
            "render_scale": s, "width": int(page.rect.width * s), "height": int(page.rect.height * s),
            "scale": {"label": label, "in_per_ft": ipf, "source": source} if ipf else None}


def api_detect(_q, body, ai_rooms=None):
    _doc, page = _open_page(body["plan"], int(body.get("page", 1)))
    ipf = float(body["in_per_ft"])
    ppf = pt_per_ft(ipf)
    if ai_rooms is None:
        rooms = _detect_cached(body["plan"], int(body.get("page", 1)), page, ppf, body)
    else:
        rooms = find_rooms(page, ppf, threshold=int(body.get("threshold", 200)), thin=int(body.get("thin", 1)),
                           gap_ft=float(body.get("gap_ft", 3.75)), min_sf=float(body.get("min_sf", 30)),
                           barriers=body.get("barriers_pt") or [], ai_rooms=ai_rooms)
    return {"rooms": [_room_json(r) for r in rooms]}


_ROOMS_CACHE = {}
_ROOMS_LOCK = threading.Lock()


def _detect_cached(plan, page_no, page, ppf, body):
    """find_rooms for this page and settings, kept so later clicks are instant."""
    key = json.dumps([plan, page_no, round(ppf, 4), body.get("gap_ft", 3.75), body.get("min_sf", 30),
                      body.get("threshold", 200), body.get("thin", 1), body.get("barriers_pt") or []])
    with _ROOMS_LOCK:
        if key not in _ROOMS_CACHE:
            if len(_ROOMS_CACHE) > 8:
                _ROOMS_CACHE.clear()
            _ROOMS_CACHE[key] = find_rooms(page, ppf, threshold=int(body.get("threshold", 200)), thin=int(body.get("thin", 1)),
                                           gap_ft=float(body.get("gap_ft", 3.75)), min_sf=float(body.get("min_sf", 30)),
                                           barriers=body.get("barriers_pt") or [])
        return _ROOMS_CACHE[key]


def _room_json(r):
    return {"name": r.get("name"), "number": r.get("number"), "area_sf": r["area_sf"],
            "outline_pt": r["outline_pt"], "holes_pt": r.get("holes_pt") or [],
            "ai_candidate": bool(r.get("ai_candidate")), "finish_tag": r.get("finish_tag"), "check": r.get("check"),
            "traced_by": r.get("traced_by"), "openings_pt": r.get("openings_pt")}


def api_room_at(_q, body):
    """The room under a click: from the page's detected rooms, else traced out from the point."""
    page_no = int(body.get("page", 1))
    _doc, page = _open_page(body["plan"], page_no)
    ppf = pt_per_ft(float(body["in_per_ft"]))
    rooms = _detect_cached(body["plan"], page_no, page, ppf, body)
    with _ROOMS_LOCK:
        r = room_at(page, body["pt"], ppf, rooms, gap_ft=float(body.get("gap_ft", 3.75)),
                    min_sf=float(body.get("min_sf", 30)), barriers=body.get("barriers_pt") or [])
    return {"room": _room_json(r) if r else None}


_TRACE_LOCK = threading.Lock()


def api_trace(_q, body):
    """The wall face between two corners of a room being drawn (the draw tool)."""
    from .detect import trace_between
    _doc, page = _open_page(body["plan"], int(body.get("page", 1)))
    ppf = pt_per_ft(float(body["in_per_ft"]))
    with _TRACE_LOCK:
        path = trace_between(page, body["a"], body["b"], ppf)
    return {"path": [[round(float(x), 2), round(float(y), 2)] for x, y in path] if path else None}


def api_merge(_q, body):
    """Join touching rooms into one outline (a room split by finish lines)."""
    from .detect import union_polygons
    ppf = pt_per_ft(float(body["in_per_ft"]))
    out = union_polygons(body["polys_pt"], bridge_pt=max(2.0, 0.75 * ppf))
    if out is None:
        raise ApiError("Those rooms don't touch, so they can't be joined into one outline.")
    return {"outline_pt": out}


def api_access(_q, body):
    """Suggested access points (openings onto the buildup) for the included areas."""
    from .access import find_access_points
    _doc, page = _open_page(body["plan"], int(body.get("page", 1)))
    ppf = pt_per_ft(float(body["in_per_ft"]))
    rooms = [r for r in body.get("rooms", []) if len(r.get("outline_pt") or []) >= 3]
    # rooms Claude traced come with their openings (where the floor carried on through a gap
    # Claude marked as a way in); the rest are searched for edges with no wall behind them
    ai_rooms = [r for r in rooms if r.get("openings_pt") is not None]
    others = [r for r in rooms if r.get("openings_pt") is None]
    out = find_access_points(page, others, ppf, all_rooms=rooms) if others else []
    from .detect import point_in_polygon
    from .ai_outline import _seg_dist
    n_ai = 0
    for r in ai_rooms:
        ring = r["outline_pt"]
        for a, b in r["openings_pt"]:
            m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            near = min(_seg_dist(m, ring[i], ring[(i + 1) % len(ring)])[0] for i in range(len(ring)))
            if near > 1.5 * ppf:                     # no longer on the border (the room was joined or edited)
                continue
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dy) or 1
            nx, ny = dy / L * ppf, -dx / L * ppf      # a foot either side: the side that isn't this room
            o = (m[0] + nx, m[1] + ny) if not point_in_polygon(m[0] + nx, m[1] + ny, ring) else (m[0] - nx, m[1] - ny)
            if any(q is not r and point_in_polygon(o[0], o[1], q["outline_pt"]) for q in rooms):
                continue                             # onto another included area: no ramp between them here
            out.append({"a": a, "b": b, "length_ft": round(L / ppf, 2), "room": r.get("name")})
            n_ai += 1
    out.sort(key=lambda x: -x["length_ft"])
    return {"access": out, "from_ai": n_ai}


def api_ai_status(_q, _body):
    return ai.key_status()


def api_ai_models(_q, _body):
    try:
        return {"models": ai.list_models(), "current": ai.get_settings()}
    except ai.AiError as e:
        raise ApiError(str(e))


def api_ai_settings(_q, body):
    try:
        ai.save_settings(body.get("model"), body.get("rpm"), body.get("trace_model"))
    except (ai.AiError, ValueError) as e:
        raise ApiError(str(e))
    return ai.key_status()


def api_ai_key(_q, body):
    try:
        msg, model_ok, prov = ai.save_key(body.get("key"))
    except ai.AiError as e:
        raise ApiError(str(e))
    return dict(ai.key_status(), message=msg, model_available=model_ok, saved=prov)


def api_ai_detect(_q, body):
    """AI first pass: Gemini or Claude reads the sheet, then rooms are traced from its findings."""
    _doc, page = _open_page(body["plan"], int(body.get("page", 1)))
    try:
        ai_rooms, info = ai.read_rooms(page)
    except ai.AiError as e:
        raise ApiError(str(e))
    out = api_detect(_q, body, ai_rooms=ai_rooms)
    out["ai"] = info
    out["ai_rooms"] = len(ai_rooms)
    if ai.provider(info["model"]) == "claude":
        out["traced"] = _claude_borders(page, pt_per_ft(float(body["in_per_ft"])), out["rooms"], ai_rooms)
    return out


def api_ai_trace(_q, body):
    """Claude traces the rooms the user clicked inside: no whole-sheet reading, one request per
    room. body: plan, page, in_per_ft, points (PDF points), plus the detect settings."""
    from concurrent.futures import ThreadPoolExecutor
    from . import ai_outline
    from .detect import point_in_polygon
    if not ai.get_key("claude"):
        raise ApiError("Tracing rooms needs a Claude API key. Add one in AI settings.")
    page_no = int(body.get("page", 1))
    _doc, page = _open_page(body["plan"], page_no)
    ppf = pt_per_ft(float(body["in_per_ft"]))
    pts = [p for p in body.get("points") or [] if len(p) == 2][:30]
    if not pts:
        raise ApiError("Click inside the rooms to trace first.")
    try:                                             # the line-traced rooms frame each picture and give a name
        lined = _detect_cached(body["plan"], page_no, page, ppf, body)
    except Exception:
        lined = []
    client = ai._claude_client(ai.get_key("claude"))
    model = ai.get_settings()["trace_model"]

    def job(p):
        r = next((r for r in lined if point_in_polygon(p[0], p[1], r["outline_pt"])), None)
        view = None
        if r and r["area_sf"] < 20000:
            xs = [x for x, _ in r["outline_pt"]] + [p[0]]
            ys = [y for _, y in r["outline_pt"]] + [p[1]]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            pad = max(15 * ppf, 0.25 * max(x1 - x0, y1 - y0))
            view = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        label = " ".join(x for x in ((r or {}).get("name"), (r or {}).get("number")) if x) or None
        try:
            res = ai_outline.room_at(page, list(p), ppf, provider="claude", client=client, view=view,
                                     label=label, model=model)
        except ai.AiError as e:
            raise ApiError(str(e))
        except Exception as e:
            import anthropic
            if isinstance(e, anthropic.AuthenticationError):
                raise ApiError("Anthropic rejected the Claude API key. Check it in AI settings.")
            if isinstance(e, anthropic.APIStatusError) and "credit" in str(e).lower():
                raise ApiError("The Claude account has no credit left. Add credit at platform.claude.com (Settings → Billing).")
            print(f"  Claude trace at {p}: {type(e).__name__}: {e}", flush=True)
            return None, r
        return res, r

    t0 = datetime.datetime.now()
    with ThreadPoolExecutor(max_workers=min(8, len(pts))) as ex:
        results = list(ex.map(job, pts))
    out, tok_in, tok_out, chk_in, chk_out = [], 0, 0, 0, 0
    for p, (res, r) in zip(pts, results):
        if not res:
            out.append(None)
            continue
        tok_in += res["usage"]["input"]
        tok_out += res["usage"]["output"]
        chk_in += res["usage"].get("check_input", 0)
        chk_out += res["usage"].get("check_output", 0)
        o = res["outline_pt"]
        holes = [h for h in (r or {}).get("holes_pt") or []
                 if point_in_polygon(sum(x for x, _ in h) / len(h), sum(y for _, y in h) / len(h), o)]
        room = {"name": (r or {}).get("name"), "number": (r or {}).get("number"), "outline_pt": o,
                "holes_pt": holes, "traced_by": "claude", "openings_pt": res.get("openings_pt"),
                "doors_pt": res.get("doors_pt"),                 # what Claude marked, for checking
                "candidates_pt": res.get("candidates_pt"), "candidate_kinds": res.get("candidate_kinds")}
        room["area_sf"] = round(room_area_pt(room) / ppf ** 2, 1)
        out.append(room)
    pr, pc = ai.price(model), ai.price(ai_outline.CHECK_MODEL)
    cost = (tok_in * pr[0] + tok_out * pr[1] + chk_in * pc[0] + chk_out * pc[1]) / 1e6 if pr and pc else None
    secs = round((datetime.datetime.now() - t0).total_seconds(), 1)
    print(f"  Claude trace ({model}): {sum(1 for r in out if r)}/{len(pts)} rooms, {secs}s, "
          f"opening checks {chk_in} in / {chk_out} out, "
          f"{tok_in} in / {tok_out} out" + (f", ~${cost:.3f}" if cost is not None else ""), flush=True)
    return {"rooms": out, "model": model, "seconds": secs, "cost_usd": round(cost, 4) if cost is not None else None}


def _claude_borders(page, ppf, rooms, ai_rooms):
    """Claude traces the border of every room the first pass ticked: it marks the rough
    outline around a point inside the room, then each edge is snapped onto the PDF's wall
    lines (ai_outline.py). Replaces the line-traced outline, which often cuts across angled
    walls or stops at revision clouds. Columns inside the new border are kept."""
    from concurrent.futures import ThreadPoolExecutor
    from . import ai_outline
    from .detect import point_in_polygon

    def ai_room(r):
        """The first pass's sighting of this room: by number, else name, else a point inside."""
        for a in ai_rooms:
            if r.get("number") and a.get("number") == r["number"]:
                return a
        for a in ai_rooms:
            if a.get("name") and (r.get("name") or "").startswith(a["name"]):
                return a
        return next((a for a in ai_rooms if point_in_polygon(a["pt"][0], a["pt"][1], r["outline_pt"])), None)

    def view(r, a):
        """Frame the picture around the whole room: the line-traced outline, the AI's rough box
        and its point, with a margin, so walls the line tracing missed are still in view."""
        xs = [x for x, _ in r["outline_pt"]] + [a["box"][0], a["box"][2], a["pt"][0]]
        ys = [y for _, y in r["outline_pt"]] + [a["box"][1], a["box"][3], a["pt"][1]]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        pad = max(15 * ppf, 0.25 * max(x1 - x0, y1 - y0))
        return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)

    jobs = [(r, ai_room(r)) for r in rooms if r.get("ai_candidate")]
    jobs = [(r, a) for r, a in jobs if a]
    if not jobs:
        return {"rooms": 0}
    client = ai._claude_client(ai.get_key("claude"))
    model = ai.get_settings()["trace_model"]

    def run(job):
        try:
            r, a = job
            return ai_outline.room_at(page, list(a["pt"]), ppf, provider="claude", client=client, view=view(r, a),
                                     label=" ".join(x for x in (a.get("name"), a.get("number")) if x) or None,
                                     model=model)
        except Exception as e:                       # one room failing keeps its line-traced outline
            print(f"  Claude border for {job[0].get('name')}: {type(e).__name__}: {e}", flush=True)
            return None

    t0 = datetime.datetime.now()
    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as ex:
        results = list(ex.map(run, jobs))
    tok_in = tok_out = done = chk_in = chk_out = 0
    rough = {}                                       # the model's own outlines, for checking
    for (r, _p), res in zip(jobs, results):
        if not res:
            continue
        tok_in += res["usage"]["input"]
        tok_out += res["usage"]["output"]
        chk_in += res["usage"].get("check_input", 0)
        chk_out += res["usage"].get("check_output", 0)
        out = res["outline_pt"]
        holes = [h for h in r.get("holes_pt") or []
                 if point_in_polygon(sum(x for x, _ in h) / len(h), sum(y for _, y in h) / len(h), out)]
        r.update(outline_pt=out, holes_pt=holes, check=None, traced_by="claude", openings_pt=res.get("openings_pt"))
        rough[r.get("name") or "?"] = res["rough_pt"]
        r["area_sf"] = round(room_area_pt(r) / ppf ** 2, 1)
        done += 1
    pr, pc = ai.price(model), ai.price(ai_outline.CHECK_MODEL)
    cost = (tok_in * pr[0] + tok_out * pr[1] + chk_in * pc[0] + chk_out * pc[1]) / 1e6 if pr and pc else None
    secs = round((datetime.datetime.now() - t0).total_seconds(), 1)
    print(f"  Claude borders ({model}): {done}/{len(jobs)} rooms, {secs}s, {tok_in} in / {tok_out} out"
          + (f", ~${cost:.3f}" if cost is not None else ""), flush=True)
    return {"rooms": done, "tried": len(jobs), "model": model, "seconds": secs,
            "cost_usd": round(cost, 4) if cost is not None else None, "rough": rough}


_WALL_IMG = {}


def _wall_test(plan, page_no, ppf):
    """wall_at(x, y): is the plan drawn dark (wall poché or wall lines) around that point (PDF
    points)? Judged from the rendered page, i.e. what the drawing looks like: hidden fills and
    pale hatching don't count. None if the plan can't be read."""
    try:
        key = (plan, page_no)
        if key not in _WALL_IMG:
            _doc, page = _open_page(plan, page_no)
            s = min(2.0, (30e6 / (page.rect.width * page.rect.height)) ** 0.5)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csGRAY, alpha=False)
            import numpy as np
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
            _WALL_IMG.clear()
            _WALL_IMG[key] = (img < 150, s)
        dark, s = _WALL_IMG[key]
        H, W = dark.shape
        r = max(1, int(round(2 / 12 * ppf * s)))

        def wall_at(x, y):
            X, Y = int(x * s), int(y * s)
            if not (0 <= X < W and 0 <= Y < H):
                return False
            return bool(dark[max(0, Y - r):Y + r + 1, max(0, X - r):X + r + 1].mean() > 0.5)
        return wall_at
    except Exception:
        return None


def _takeoff(body):
    """body: rooms (included, each with outline_pt, holes_pt, layers top first,
    finish_adhesive_by_pliteq), in_per_ft, manual_notes."""
    cat = engine.load()
    ppf = pt_per_ft(float(body["in_per_ft"]))
    rooms = _engine_rooms([r for r in body.get("rooms", []) if r.get("included") is True], ppf)
    if not rooms:
        raise ApiError("No rooms are selected.")
    T = body.get("transitions")
    if T and body.get("plan") and T.get("openings"):
        T = dict(T, _wall_at=_wall_test(body["plan"], int(body.get("page", 1)), ppf))
    try:
        t = engine.compute(cat, rooms, body.get("manual_notes") or [], transitions=T, pt_per_ft=ppf)
    except engine.TakeoffError as e:
        raise ApiError(str(e))
    return cat, t, rooms


def api_takeoff(_q, body):
    _cat, t, _rooms = _takeoff(body)
    from .sheet import ramp_shapes
    ppf = pt_per_ft(float(body["in_per_ft"]))

    def shapes(x):                                   # where the transition sits, for the editor
        try:
            return [[k, [[round(p[0], 2), round(p[1], 2)] for p in poly]] for k, poly in ramp_shapes(x, ppf)] if x.get("a") else []
        except Exception:
            return []
    return {"lines": [ln.text() for ln in t.lines], "notes": t.notes, "note_ids": t.note_ids,
            "warnings": t.warnings, "net_sf": t.net_sf,
            "ramps": [dict({k: v for k, v in x.items() if k not in ("a", "b", "normal", "depth_in")}, shapes=shapes(x))
                      for x in (t.ramps or {}).get("ramps", [])],
            "groups": [{k: g[k] for k in ("key", "layers", "callout", "rooms", "net_sf", "thickness_mm")}
                       for g in t.groups]}


def api_build(_q, body):
    plan, page_no = body["plan"], int(body.get("page", 1))
    ipf = float(body["scale"]["in_per_ft"])
    ppf = pt_per_ft(ipf)
    if not body.get("project", "").strip():
        raise ApiError("Enter a project name.")
    body["in_per_ft"] = ipf
    cat, t, rooms = _takeoff(body)
    company = yaml.safe_load(COMPANY.read_text(encoding="utf-8"))

    dwg = body.get("dwg", "").strip()
    job = _safe_name(dwg or body["project"], "takeoff")
    out_dir = JOBS / job
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{job}.pdf"

    doc, _page = _open_page(plan, page_no)
    meta = {"project": body["project"].strip(), "dwg_no": dwg, "drawn_by": body.get("drawn_by", "").strip(),
            "date": body.get("date") or None, "pt_per_ft": ppf,
            "ramps": t.ramps}                  # transitions to draw (None on a recessed slab)
    try:
        render_sheet(str(pdf_path), doc, page_no, rooms, t, company, meta, cat)
    except RuntimeError as e:
        if "Permission" in str(e) or "save" in str(e).lower():
            raise ApiError(f"Couldn't write {pdf_path.name}. Close it in Preview/Acrobat and try again.")
        raise

    from .cli import write_bom_csv
    write_bom_csv(pdf_path.with_suffix(".csv"), t, rooms)

    saved = {"source": str(_plan_path(plan)), "page": page_no,
             "scale": {"label": body["scale"].get("label"), "in_per_ft": ipf, "pt_per_ft": ppf},
             "barriers_pt": body.get("barriers_pt") or [],
             "takeoff": {k: body.get(k) for k in ("project", "dwg", "drawn_by", "date", "manual_notes", "transitions")},
             "rooms": [{"id": i, "name": r.get("name"), "number": r.get("number"), "included": bool(r.get("included")),
                        "area_sf": round(room_area_pt(r) / ppf ** 2, 1), "outline_pt": r["outline_pt"],
                        "holes_pt": r.get("holes_pt") or [],
                        "layers": [dict(l, code="CUSTOM") if str(l.get("code", "")).startswith("CUSTOM:") else l
                                   for l in r.get("layers") or []],
                        "finish_adhesive_by_pliteq": bool(r.get("finish_adhesive_by_pliteq"))}
                       for i, r in enumerate(body["rooms"], 1)]}
    (out_dir / "rooms.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")

    rel = f"{job}/{job}"
    return {"lines": [ln.text() for ln in t.lines], "notes": t.notes, "net_sf": t.net_sf,
            "folder": str(out_dir), "pdf": f"/files/{rel}.pdf", "csv": f"/files/{rel}.csv",
            "rooms_json": f"/files/{job}/rooms.json"}


def api_reveal(_q, body):
    target = (JOBS / _safe_name(body.get("job", ""), "")).resolve()
    if target.parent != JOBS.resolve() or not target.exists():
        raise ApiError("Folder not found.", HTTPStatus.NOT_FOUND)
    if sys.platform == "darwin":
        subprocess.run(["open", str(target)], check=False)
    elif sys.platform.startswith("win"):
        subprocess.run(["explorer", str(target)], check=False)
    else:
        subprocess.run(["xdg-open", str(target)], check=False)
    return {"ok": True}


GET_API = {"/api/ai/status": api_ai_status, "/api/ai/models": api_ai_models, "/api/catalog": api_catalog, "/api/plans": api_plans, "/api/open": api_open}
POST_API = {"/api/detect": api_detect, "/api/access": api_access, "/api/ai/key": api_ai_key, "/api/ai/settings": api_ai_settings, "/api/ai/detect": api_ai_detect, "/api/ai/trace": api_ai_trace, "/api/takeoff": api_takeoff, "/api/room_at": api_room_at, "/api/merge": api_merge, "/api/trace": api_trace, "/api/build": api_build,
            "/api/reveal": api_reveal}


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "TakeoffApp/1"

    def log_message(self, fmt, *args):
        if args and str(args[1]).startswith(("4", "5")):
            sys.stderr.write("  " + (fmt % args) + "\n")

    def _send(self, status, body, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status=HTTPStatus.OK):
        self._send(status, json.dumps(data).encode(), "application/json")

    def _file(self, path):
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(HTTPStatus.OK, path.read_bytes(), ctype)

    def _run(self, fn, q, body):
        try:
            self._json(fn(q, body))
        except ApiError as e:
            self._json({"error": str(e)}, e.status)
        except (KeyError, ValueError, TypeError) as e:
            self._json({"error": f"Bad request: {e}"}, HTTPStatus.BAD_REQUEST)
        except Exception as e:  # keep the server alive and tell the page what went wrong
            import traceback
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    # ---- only the app's own page may use it ------------------------------------------
    # Other websites open in the browser could otherwise send requests to this local app
    # (and, for example, start AI runs that use the Gemini key). Browsers mark such
    # requests with Origin / Sec-Fetch-Site headers, and a site that points its own domain
    # at 127.0.0.1 still sends its own Host header, so all three are checked.
    def _local_names(self):
        port = self.server.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}

    def _allowed(self, strict):
        names = self._local_names()
        host = (self.headers.get("Host") or "").lower()
        if host not in names:
            return False
        origin = (self.headers.get("Origin") or "").lower()
        if origin and origin != "null":
            if urlparse(origin).netloc not in names:
                return False
        elif origin == "null":
            return False
        site = (self.headers.get("Sec-Fetch-Site") or "").lower()
        if strict and site and site not in ("same-origin", "none"):
            return False
        return True

    def _refuse(self):
        self._json({"error": "Refused: requests must come from the takeoff app itself."}, HTTPStatus.FORBIDDEN)

    def do_GET(self):
        u = urlparse(self.path)
        # pages and images may be opened directly; anything under /api/ or /files/ must be the app's own
        if not self._allowed(strict=u.path.startswith(("/api/", "/files/")) and u.path != "/api/render"):
            return self._refuse()
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            return self._file(APP_HTML)
        if u.path in GET_API:
            return self._run(GET_API[u.path], q, None)
        if u.path == "/api/render":
            try:
                _doc, page = _open_page(q["plan"], int(q.get("page", 1)))
                if q.get("clip"):
                    # a sharp crop of the zoomed-in view: clip in PDF points, scale in px per point
                    x0, y0, x1, y1 = (float(v) for v in q["clip"].split(","))
                    k = max(0.5, min(float(q.get("scale", 4)), 12.0))
                    rect = pymupdf.Rect(x0, y0, x1, y1) & page.rect
                    if rect.is_empty or rect.width * rect.height * k * k > 40e6:
                        raise ApiError("Crop too large")
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(k, k), clip=rect, alpha=False)
                    return self._send(HTTPStatus.OK, pix.tobytes("png"), "image/png")
                img, _s = render_page(page)
                import cv2
                ok, png = cv2.imencode(".png", img[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
                return self._send(HTTPStatus.OK, png.tobytes(), "image/png")
            except ApiError as e:
                return self._json({"error": str(e)}, e.status)
        if u.path == "/api/circles":
            # circles in the drawing (round columns), for the round-column tool
            try:
                from .detect import clear_cache, find_circles
                _doc, page = _open_page(q["plan"], int(q.get("page", 1)))
                cs = find_circles(page)
                clear_cache()
                return self._json({"circles": [[round(x, 2), round(y, 2), round(r, 2)] for x, y, r in cs]})
            except ApiError as e:
                return self._json({"error": str(e)}, e.status)
        if u.path == "/api/lines":
            # horizontal and vertical line work, for snapping corners onto the drawing
            try:
                from .detect import axis_lines, clear_cache, raster_axis_lines
                _doc, page = _open_page(q["plan"], int(q.get("page", 1)))
                hs, vs = axis_lines(page, min_len_pt=2)
                if len(hs) + len(vs) < 50:      # a scanned sheet: find the lines in the picture
                    hs, vs = raster_axis_lines(page)
                clear_cache()
                r2 = lambda t: [round(t[0], 2), round(t[1], 2), round(t[2], 2)]
                return self._json({"h": [r2(t) for t in hs], "v": [r2(t) for t in vs]})
            except ApiError as e:
                return self._json({"error": str(e)}, e.status)
        if u.path == "/api/walls":
            # The clean wall drawing (no text, hatching or markups) used by the click-to-add tool.
            try:
                _doc, page = _open_page(q["plan"], int(q.get("page", 1)))
                import cv2
                from .detect import wall_mask_vector
                s = _render_scale(page)
                m = wall_mask_vector(page, s, min_stroke_pt=float(q.get("min_stroke", 0.3)))
                if m is None:
                    return self._json({"error": "No vector line work on this page"}, HTTPStatus.NOT_FOUND)
                ok, png = cv2.imencode(".png", (1 - m) * 255)
                return self._send(HTTPStatus.OK, png.tobytes(), "image/png")
            except ApiError as e:
                return self._json({"error": str(e)}, e.status)
        if u.path.startswith("/files/"):
            p = (JOBS / unquote(u.path[len("/files/"):])).resolve()
            if JOBS.resolve() in p.parents and p.is_file():
                return self._file(p)
        self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        u = urlparse(self.path)
        if not self._allowed(strict=True):
            return self._refuse()
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if u.path == "/api/upload":
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if not raw.startswith(b"%PDF"):
                return self._json({"error": "That file isn't a PDF."}, HTTPStatus.BAD_REQUEST)
            PLANS.mkdir(exist_ok=True)
            name = _safe_name(q.get("name", "plan.pdf")) + ".pdf"
            (PLANS / name).write_bytes(raw)
            return self._json({"plan": name})
        if u.path in POST_API:
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                return self._json({"error": "Bad JSON"}, HTTPStatus.BAD_REQUEST)
            return self._run(POST_API[u.path], {}, body)
        self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)


def serve(port=8765, open_browser=True):
    httpd = None
    for p in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            break
        except OSError:
            continue
    if httpd is None:
        sys.exit(f"No free port between {port} and {port + 19}.")
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"\n  Takeoff app running at {url}", flush=True)
    print("  Keep this window open while you use it. Press Ctrl+C to stop.\n", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
