"""Find the rooms on a plan, using the room labels printed on it to check the result.

Wall tracing alone can't know whether a 12 ft opening is a doorway (close it) or part of
the room (leave it). The plan's own room labels settle it: every numbered room should end
up in an area of its own. So rooms are traced with several wall rules and opening widths,
and each numbered room keeps the first outline that holds only its own number.
"""
from .detect import (attach_columns, axis_lines, bbox, clear_cache, detect_rooms, filled_polygons, find_columns,
                     label_point, point_in_polygon, polygon_area, refine_outline, render_page,
                     room_area_pt, subtract_polygons)
from .labels import keep_plan_rooms, name_rooms, room_labels
from .pdftext import text_dict, text_words

# Wall rules, tried in order. Plans differ in how they draw walls: some use the heaviest
# line weight for walls (the first rule), others draw walls in a medium weight and use
# heavier lines for equipment, which the long-straight-lines rule handles.
WALL_RULES = [
    {"label": "line weight", "opts": {}},
    {"label": "medium lines", "opts": {"min_stroke_pt": 0.3, "fine_stroke_pt": 0}},
    {"label": "long lines", "opts": {"orth_only": "long", "min_stroke_pt": 0.2, "fine_stroke_pt": 0,
                                     "min_len_ft": 1.0}},
]
GAP_STEPS = (0.75, 1.0, 1.5, 2.1, 3.0)         # x the base opening width (3.75 ft): ~2.8 ft to ~11 ft


def _labels_in(poly, labels):
    return {lb["uid"] for lb in labels if point_in_polygon(lb["pt"][0], lb["pt"][1], poly)}


def _clean(r):
    return {k: v for k, v in r.items() if k != "parts"}


def _overlaps(a, b):
    """Rough overlap test between two room outlines (each one's label point inside the other)."""
    return (point_in_polygon(*label_point(a["outline_pt"]), b["outline_pt"])
            or point_in_polygon(*label_point(b["outline_pt"]), a["outline_pt"]))


def _named(r, lb):
    r = _clean(r)
    r["name"], r["number"], r["uid"] = f"{lb['name']} {lb['number']}", lb["number"], lb["uid"]
    return r


def _resolve(page, labels, pt_ft, gap_ft, opts, detect_kw, min_sf):
    runs = [detect_rooms(page, pt_ft=pt_ft, gap_ft=gap_ft * k, **opts, **detect_kw) for k in GAP_STEPS]
    by_num = {}
    for lb in labels:
        by_num.setdefault(lb["uid"], lb)
    min_pt2 = min_sf * pt_ft ** 2

    # 1. each numbered room: the first outline (narrowest openings first) holding only its number
    resolved = {}
    for num, lb in by_num.items():
        for run in runs:
            hit = next((r for r in run if point_in_polygon(lb["pt"][0], lb["pt"][1], r["outline_pt"])), None)
            if hit and _labels_in(hit["outline_pt"], labels) == {num}:
                resolved[num] = _named(hit, lb)
                break

    final = []
    placed = set()
    for B in runs[0]:
        S = _labels_in(B["outline_pt"], labels)
        if not S:
            continue
        if len(S) == 1:
            num = next(iter(S))
            final.append(_named(B, by_num[num]))
            placed.add(num)
            continue
        # 2. an area shared by several numbers: cut out the separated rooms, plus unnumbered
        #    spaces that came apart at wider openings, and give each room what is left around it
        inner = [resolved[n] for n in S if n in resolved]
        extra = []
        for run in runs[1:]:
            for r in run:
                if _labels_in(r["outline_pt"], labels):
                    continue
                if not point_in_polygon(*label_point(r["outline_pt"]), B["outline_pt"]):
                    continue
                if any(_overlaps(r, q) for q in inner + extra):
                    continue
                extra.append(_clean(r))
        final.extend(extra)
        done = []
        for n in sorted(S, key=lambda n: n not in resolved):     # resolved rooms first
            lb = by_num[n]
            if any(point_in_polygon(lb["pt"][0], lb["pt"][1], q["outline_pt"]) for q in done):
                continue
            holes = [q["outline_pt"] for q in inner + extra + done if q.get("uid") != n]
            piece = subtract_polygons(B["outline_pt"], holes, keep_point=lb["pt"])
            room = None
            if n in resolved:
                R = resolved[n]
                ok = piece and polygon_area(R["outline_pt"]) <= polygon_area(piece) <= 1.3 * polygon_area(R["outline_pt"])
                room = dict(R, outline_pt=piece) if ok else R
            elif piece and polygon_area(piece) >= min_pt2:
                room = _named({"outline_pt": piece}, lb)
                others = [by_num[m] for m in S if m != n and m not in resolved
                          and point_in_polygon(by_num[m]["pt"][0], by_num[m]["pt"][1], piece)]
                if others:
                    # rooms that open into each other with no door to tell them apart: one area,
                    # named after all of them, for the estimator to split if needed
                    seen = []
                    for o in [lb] + others:
                        nm = f"{o['name']} {o['number']}"
                        if nm not in seen:
                            seen.append(nm)
                    room["name"] = " + ".join(seen)
                    room["merged"] = [lb["uid"]] + [o["uid"] for o in others]
                    room["check"] = ("Open to " + ", ".join(seen[1:]) + " with no door between: split it (X) "
                                     "or close the opening (B) if only part is in the takeoff")
            if room:
                done.append(room)
                placed.add(n)
        final.extend(done)

    # 3. numbered rooms whose label sat on a wall at the narrowest openings
    for n, R in resolved.items():
        if n not in placed and not any(_overlaps(R, q) for q in final):
            final.append(R)
            placed.add(n)

    # 4. unnumbered spaces (corridors, restrooms whose label is printed outside them),
    #    narrowest openings first, where they don't overlap a room already found
    for run in runs:
        for r in run:
            if _labels_in(r["outline_pt"], labels) or any(_overlaps(r, q) for q in final):
                continue
            final.append(_clean(r))
    return final, len(placed)


def _fill_rooms(page, labels, pt_ft, min_sf):
    """Rooms the architect drew as filled shapes (colour-fill plans): for each room label,
    the smallest filled shape around it that holds no other room's label."""
    W, H = page.rect.width, page.rect.height
    polys = filled_polygons(page, min_sf * pt_ft ** 2, 0.25 * W * H)
    out = []
    for lb in labels:
        x, y = lb["pt"]
        cands = [p for p in polys if point_in_polygon(x, y, p)]
        for p in sorted(cands, key=polygon_area):
            others = [o for o in labels if o["uid"] != lb["uid"] and point_in_polygon(o["pt"][0], o["pt"][1], p)]
            if not others:
                out.append(_named({"outline_pt": p, "from_fill": True}, lb))
            break                                  # only the smallest shape around the label counts
    return out


def _borrow(rooms, other, labels):
    """Rooms another wall rule separated that this result left merged or missing.

    Plans mix wall weights (heavy demising walls, thin glass fronts), so one rule rarely
    gets every room. A room the other rule found on its own replaces this result's share of it."""
    def single(r):
        return r.get("uid") is not None and not r.get("merged")

    have = {r["uid"] for r in rooms if single(r)}
    out = list(rooms)
    for r in other:
        if not single(r) or r["uid"] in have:
            continue
        lb = next(lb for lb in labels if lb["uid"] == r["uid"])
        if any(_overlaps(r, q) for q in out if single(q)):
            continue
        a = polygon_area(r["outline_pt"])
        nxt = []
        for q in out:
            if point_in_polygon(lb["pt"][0], lb["pt"][1], q["outline_pt"]) or _overlaps(r, q):
                rest = subtract_polygons(q["outline_pt"], [r["outline_pt"]], keep_point=None)
                if rest and polygon_area(rest) > 0.1 * a:
                    q = dict(q, outline_pt=rest)
                    inl = [x for x in labels if x["uid"] != r["uid"]
                           and point_in_polygon(x["pt"][0], x["pt"][1], rest)]
                    if q.get("merged"):
                        q["merged"] = [x["uid"] for x in inl]
                        if len(inl) == 1:
                            q = _named({k: v for k, v in q.items() if k not in ("merged", "check")}, inl[0])
                        elif inl:
                            q["name"] = " + ".join(dict.fromkeys(f"{x['name']} {x['number']}" for x in inl))
                    nxt.append(q)
                continue
            nxt.append(q)
        out = nxt + [r]
        have.add(r["uid"])
    return out


def _text_cover(r, words):
    """Share of an outline covered by words: notes, legends and title-block boxes are full of
    text, rooms aren't."""
    a = polygon_area(r["outline_pt"])
    if not a:
        return 0
    x0, y0, x1, y1 = bbox(r["outline_pt"])
    ws = [w for w in words if x0 <= w[0] and w[2] <= x1 and y0 <= w[1] and w[3] <= y1
          and point_in_polygon((w[0] + w[2]) / 2, (w[1] + w[3]) / 2, r["outline_pt"])]
    if len(ws) < 40:                 # a room with a few lines of notes in it is still a room
        return 0
    return sum((w[2] - w[0]) * (w[3] - w[1]) for w in ws) / a


def _ai_labels(text_labels, ai_rooms):
    """AI-read rooms as extra room labels, for rooms the PDF's text layer doesn't already label."""
    import math
    nums = {lb["number"] for lb in text_labels}
    out = []
    for i, a in enumerate(ai_rooms or []):
        if a.get("number") and a["number"] in nums:
            continue                                        # the text layer has it, exactly
        if any(math.dist(a["pt"], lb["pt"]) < 60 and a["name"] and a["name"] in lb["name"] for lb in text_labels):
            continue
        display = " ".join(x for x in (a.get("name"), a.get("number")) if x)
        out.append({"name": a.get("name") or "ROOM", "number": a.get("number") or f"~{i}", "pt": tuple(a["pt"]),
                    "display": display, "ai": a})
    return out


def find_rooms(page, pt_ft, gap_ft=3.75, min_sf=30, barriers=(), threshold=200, thin=1, refine=True,
               ai_rooms=None):
    """Rooms on the page, named and numbered in reading order.

    Each room has outline_pt (PDF points), holes_pt (columns inside it) and area_sf
    (outline minus columns). ai_rooms (from ai.read_rooms) add room labels the text layer
    lacks (e.g. on scans), mark likely takeoff areas, and give a fallback outline."""
    clear_cache()
    labels = room_labels(page)
    labels = labels + _ai_labels(labels, ai_rooms)
    labels = [dict(lb, uid=i) for i, lb in enumerate(labels)]   # the same number can label two views of a room
    words = text_words(page)
    detect_kw = {"min_sf": min_sf, "barriers": barriers, "threshold": threshold, "thin": thin}
    best = None
    rule_opts = WALL_RULES[0]["opts"]
    if labels:
        results = []
        for rule in WALL_RULES:
            rooms, score = _resolve(page, labels, pt_ft, gap_ft, rule["opts"], detect_kw, min_sf)
            results.append((score, rooms, rule["opts"]))
            if score == len(labels):
                break
        results.sort(key=lambda t: -t[0])
        rooms, rule_opts = results[0][1], results[0][2]
        for _score, other, _opts in results[1:]:
            rooms = _borrow(rooms, other, labels)
        fills = _fill_rooms(page, labels, pt_ft, min_sf)
        if fills:
            rooms = _borrow(rooms, fills, labels)   # only where tracing left the room merged or missing
    else:   # no room labels (e.g. a scan): plain tracing
        rooms = detect_rooms(page, pt_ft=pt_ft, gap_ft=gap_ft, **detect_kw)
        for r in rooms:
            r.pop("parts", None)

    # AI outlines: the AI says where each room is, the drawing's walls near that outline say
    # exactly where (noise inside the room is ignored). They replace the traced rooms they cover.
    from .guided import covered_by, guided_rooms
    guided = guided_rooms(page, pt_ft, ai_rooms, barriers=barriers, gap_ft=gap_ft,
                          label_pts=[lb["pt"] for lb in labels]) if ai_rooms else []
    if guided:
        rooms = [r for r in rooms if not covered_by(r, guided)] + guided

    # AI-read rooms that no traced outline could isolate: use the AI's box, snapped to the drawing
    placed = {r.get("number") for r in rooms}
    for lb in labels:
        if lb.get("ai") and lb["ai"].get("polys"):
            continue                                  # handled by the guided outlines
        if lb.get("ai") and lb["number"] not in placed:
            x0, y0, x1, y1 = lb["ai"]["box"]
            if (x1 - x0) * (y1 - y0) < min_sf * pt_ft ** 2:
                continue
            rooms.append({"outline_pt": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "name": lb["display"],
                          "number": lb["number"], "check": "Outline from the AI's rough box: check it"})

    rooms = keep_plan_rooms(rooms)
    page_pt2 = page.rect.width * page.rect.height
    rooms = [r for r in rooms if polygon_area(r["outline_pt"]) < 0.25 * page_pt2]   # the sheet border isn't a room
    rooms = [r for r in rooms if r.get("number") or _text_cover(r, words) < 0.2]
    if refine:
        _refine(page, rooms, pt_ft, rule_opts)
    else:
        for r in rooms:
            r.setdefault("holes_pt", [])
    clear_cache()

    rooms.sort(key=lambda r: (round(bbox(r["outline_pt"])[1] / 100), bbox(r["outline_pt"])[0]))
    for r in rooms:
        if r.get("guided"):
            r["_ai_name"] = r["name"]
        if not r.get("number"):
            r["name"] = None
    name_rooms(page, rooms, words, [])
    for r in rooms:                                   # the AI's reading wins over a generic name
        if r.get("guided") and (not r.get("name") or str(r["name"]).startswith("Room ")):
            r["name"] = r["_ai_name"]
        r.pop("_ai_name", None)
    by_num = {lb["number"]: lb for lb in labels}
    for r in rooms:
        lb = by_num.get(r.get("number"))
        if lb and lb.get("display"):                 # an AI-read name ("~n" = no printed number)
            r["name"] = lb["display"]
            if r["number"].startswith("~"):
                r["number"] = None
        r["area_sf"] = round(room_area_pt(r) / pt_ft ** 2, 1)
    # what the AI thinks belongs in the takeoff, and the finish tag it read
    for r in rooms:
        if r.get("guided"):
            a = ai_rooms[r["ai_index"]]
            r["ai_candidate"] = a.get("candidate", False)
            r["finish_tag"] = a.get("finish_tag")
    for a in ai_rooms or []:
        for r in rooms:
            if r.get("guided"):
                continue
            if (a.get("number") and a["number"] == r.get("number")) or point_in_polygon(a["pt"][0], a["pt"][1], r["outline_pt"]):
                r["ai_candidate"] = r.get("ai_candidate") or a.get("candidate", False)
                r["finish_tag"] = r.get("finish_tag") or a.get("finish_tag")
                break
    return rooms


def _refine(page, rooms, pt_ft, rule_opts):
    """Exact corners from the PDF's wall lines, and free-standing columns cut out."""
    if len(page.get_drawings()) < 200:          # scanned sheet: nothing to snap to
        for r in rooms:
            r.setdefault("holes_pt", [])
        return
    _img, s = render_page(page)
    hs, vs = axis_lines(page)
    tol = max(1.5, 2.6 / s)                     # about two pixels of the tracing
    for r in rooms:
        r["outline_pt"] = refine_outline(r["outline_pt"], hs, vs, tol)
    mask_opts = {"min_stroke_pt": rule_opts.get("min_stroke_pt", 0.5),
                 "orth_only": rule_opts.get("orth_only", False),
                 "min_len_pt": rule_opts.get("min_len_ft", 0) * pt_ft if rule_opts.get("orth_only") else 0}
    cols = [refine_outline(c, hs, vs, tol) for c in find_columns(page, s, pt_ft, **mask_opts)]
    attach_columns(rooms, cols)


def room_at(page, pt, pt_ft, rooms=None, gap_ft=3.75, min_sf=30, barriers=(), threshold=200, thin=1):
    """The room under one point (the click-to-add tool).

    First the rooms the full detection found (`rooms`, from find_rooms); if none holds the
    point, trace outward from it: each wall rule, narrowest openings first, and keep the
    first enclosed space that holds at most one room label."""
    x, y = pt

    def inside(r):
        return point_in_polygon(x, y, r["outline_pt"]) and not any(point_in_polygon(x, y, h) for h in r.get("holes_pt") or [])

    hits = [r for r in rooms or [] if inside(r)]
    if hits:
        return min(hits, key=room_area_pt)
    labels = [dict(lb, uid=i) for i, lb in enumerate(room_labels(page))]
    detect_kw = {"min_sf": min_sf, "barriers": barriers, "threshold": threshold, "thin": thin}
    try:
        for rule in WALL_RULES:
            for k in GAP_STEPS:
                for r in detect_rooms(page, pt_ft=pt_ft, gap_ft=gap_ft * k, **rule["opts"], **detect_kw):
                    if not point_in_polygon(x, y, r["outline_pt"]):
                        continue
                    inl = [lb for lb in labels if point_in_polygon(lb["pt"][0], lb["pt"][1], r["outline_pt"])]
                    if len(inl) > 1:
                        break
                    r = _named(r, inl[0]) if inl else _clean(r)
                    _refine(page, [r], pt_ft, rule["opts"])
                    r["area_sf"] = round(room_area_pt(r) / pt_ft ** 2, 1)
                    if not r.get("name"):
                        r["name"] = None
                    return r
    finally:
        clear_cache()
    return None
