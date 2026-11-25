"""Name detected rooms from the PDF text layer (e.g. 'FITNESS' printed above '420')."""
import re

from .detect import point_in_polygon
from .pdftext import text_dict, text_words

ROOM_NUMBER = re.compile(r"^(?:[A-Z]?\d{3,4}|[A-Z]?\d{1,3}[-.]\d{3,4})[A-Z]?$")   # 918, 900N, B102, 10-005, 1.105
NAME_WORD = re.compile(r"^[A-Z][A-Z&/\-']*\.?$")
IGNORE = {"TYP", "TYP.", "ALIGN", "COLOR", "UP", "DN", "EQ", "EQ.", "REF", "UNO", "U.O.N."}
FINISH_TAG = re.compile(r"^[A-Z]{1,4}-\d{1,2}[A-Z]?$")      # RF-1, WD-7, CPT-5, LVT-1


def _finish_name(inside, lines):
    """'RF-1 COLOR B' from the finish tag printed inside an unnamed area."""
    tags = [w for w in inside if FINISH_TAG.match(w[4])]
    if not tags:
        return None
    best = max({w[4] for w in tags}, key=lambda t: sum(w[4] == t for w in tags))
    w = next(x for x in tags if x[4] == best)
    h, cx = w[3] - w[1], (w[0] + w[2]) / 2
    for line in lines:   # a 'COLOR B' line directly under the tag
        toks = [x[4] for x in sorted(line)]
        if toks and toks[0] == "COLOR" and len(toks) <= 3:
            lx0, lx1, ly0 = min(x[0] for x in line), max(x[2] for x in line), min(x[1] for x in line)
            if lx0 - h <= cx <= lx1 + h and 0 <= ly0 - w[3] <= 1.5 * h:
                return f"{best} {' '.join(toks)}"
    return best


def _lines(words):
    out = {}
    for x0, y0, x1, y1, word, b, l, _ in words:
        out.setdefault((b, l), []).append((x0, y0, x1, y1, word))
    return list(out.values())


def room_labels(page):
    """Room name + number pairs printed on the plan, in any text direction.

    Returns [{'name': 'TRAINING FLOOR', 'number': '101', 'pt': (x, y)}] where pt is the
    centre of the number. Names can run over several lines above the number.
    """
    lines = []
    for b in text_dict(page)["blocks"]:
        for l in b.get("lines", []):
            text = " ".join("".join(sp["text"] for sp in l["spans"]).split())
            if not text:
                continue
            dx, dy = l["dir"]
            nx, ny = -dy, dx                        # the "next line" direction
            x0, y0, x1, y1 = l["bbox"]
            cs = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
            us = [x * dx + y * dy for x, y in cs]
            vs = [x * nx + y * ny for x, y in cs]
            lines.append({"text": text, "dir": (round(dx, 2), round(dy, 2)), "u": (min(us), max(us)),
                          "v": (min(vs), max(vs)), "c": ((x0 + x1) / 2, (y0 + y1) / 2)})

    def is_name(t):
        toks = t.split()
        return toks and len(toks) <= 4 and all(NAME_WORD.match(x) for x in toks) and not set(toks) <= IGNORE

    def above(line, pool, strict=False):
        h = line["v"][1] - line["v"][0]
        uc = sum(line["u"]) / 2
        best = None
        for o in pool:
            if o is line or o["dir"] != line["dir"] or not is_name(o["text"]):
                continue
            if o["u"][1] < line["u"][0] - h or o["u"][0] > line["u"][1] + h:
                continue
            if strict and abs(sum(o["u"]) / 2 - uc) > 0.75 * h:
                continue          # extra name lines are centred over the line below them
            gap = line["v"][0] - o["v"][1]
            if -0.6 * h <= gap <= (0.6 if strict else 1.5) * h and (best is None or gap < best[0]):
                best = (gap, o)
        return best[1] if best else None

    out, seen = [], set()
    for ln in lines:
        if not ROOM_NUMBER.match(ln["text"]):
            continue
        parts, cur = [], ln
        for _ in range(3):
            cur = above(cur, lines, strict=bool(parts))
            if cur is None or cur["text"] in parts:
                break
            parts.insert(0, cur["text"])
        if not parts:
            continue
        key = (ln["text"], round(ln["c"][0]), round(ln["c"][1]))
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": " ".join(parts), "number": ln["text"], "pt": ln["c"]})
    return out


def name_rooms(page, rooms, words=None, labels=None):
    """Set room['name'] (and 'number') from the room label printed inside each outline."""
    words = text_words(page) if words is None else words
    labels = room_labels(page) if labels is None else labels
    lines = _lines(words)
    for i, room in enumerate(rooms, 1):
        poly = room["outline_pt"]
        inside = [lb for lb in labels if point_in_polygon(lb["pt"][0], lb["pt"][1], poly)]
        if inside and not room.get("number"):
            lb = inside[0]
            room["name"], room["number"] = f"{lb['name']} {lb['number']}", lb["number"]
        elif not room.get("number"):
            ws = [w for w in words if point_in_polygon((w[0] + w[2]) / 2, (w[1] + w[3]) / 2, poly)]
            tag = _finish_name(ws, lines)
            room["name"] = room.get("name") or (f"{tag} (area {i})" if tag else f"Room {i}")
    return rooms


def keep_plan_rooms(rooms, margin=0.25):
    """Drop 'rooms' found in the title block, legend or details, away from the named rooms.

    Keeps rooms whose label point lies within the box around the rooms that have a
    room number, grown by `margin` of its size. Does nothing with fewer than 3 named rooms.
    """
    from .detect import bbox, label_point
    named = [r for r in rooms if r.get("number")]
    if len(named) < 3:
        return rooms
    x0, y0, x1, y1 = bbox([p for r in named for p in r["outline_pt"]])
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    out = []
    for r in rooms:
        lx, ly = label_point(r["outline_pt"])
        if x0 - mx <= lx <= x1 + mx and y0 - my <= ly <= y1 + my:
            out.append(r)
    return out


def name_and_filter(page, rooms):
    """Name rooms from the text layer, drop ones off the plan, and number the rest in order."""
    words = text_words(page)
    labels = room_labels(page)
    rooms = name_rooms(page, rooms, words, labels)
    rooms = _split_named_parts(page, rooms, words, labels)
    rooms = keep_plan_rooms(rooms)
    from .detect import bbox
    rooms.sort(key=lambda r: (round(bbox(r["outline_pt"])[1] / 100), bbox(r["outline_pt"])[0]))
    for r in rooms:
        if not r.get("number"):
            r["name"] = None
    return name_rooms(page, rooms, words, labels)


def _split_named_parts(page, rooms, words, labels):
    """A detected area that holds another numbered room (e.g. YOGA 420 open to FITNESS 422)
    is split: the other room becomes its own room and is cut out of the host."""
    from .detect import polygon_area, subtract_polygons
    out = []
    for host in rooms:
        parts = host.pop("parts", [])
        if not parts:
            out.append(host)
            continue
        name_rooms(page, parts, words, labels)
        own = [p for p in parts if p.get("number") and p["number"] != host.get("number")]
        if not own or not host.get("number"):
            out.append(host)
            continue
        rest = subtract_polygons(host["outline_pt"], [p["outline_pt"] for p in own])
        if rest:
            k = host["area_sf"] / polygon_area(host["outline_pt"]) if host.get("area_sf") else None
            host["outline_pt"] = rest
            if k:
                host["area_sf"] = round(polygon_area(rest) * k, 1)
            out.append(host)
        out.extend(own)
    return out
