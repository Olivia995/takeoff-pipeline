"""AI-guided room outlines.

Plain tracing (detect.py) treats every heavy line on the sheet as a possible wall, so
furniture, hatching, equipment and text inside a room can carve it up or seal it off.
Here the AI says roughly where each room is (a polygon, a foot or two off) and the
drawing says exactly where its walls are:

1. Only line work in a band around the AI outline can be a wall for that room;
   everything well inside the room (the noise) is ignored.
2. The room is the open space inside that band, bounded by those walls, with doorways
   bridged as in normal tracing.
3. Where the AI's edge has no wall near it but empty floor carries on to a real wall within
   EXTEND_FT, the room is carried on to that wall (the AI outline was short). Where nothing
   stops it, or the space beyond holds another room's label, the AI's edge is kept.
4. Edges snap onto the PDF's exact wall lines.
5. A result that disagrees badly with the AI (much bigger or smaller) falls back to the
   AI outline, snapped where it can be, and is flagged "check".
"""
import cv2
import numpy as np

from .detect import (axis_lines, bbox, drop_specks, polygon_area, raster_axis_lines, refine_outline,
                     render_page, wall_mask, wall_mask_vector)

BAND_FT = 3.0          # how far a real wall may sit from the AI's edge
MIN_BAND_PT = 18       # ... but never less than this on paper (AI error is in image terms)
MAX_BAND_SHARE = 0.3   # ... nor more than this share of the room's smaller side
AREA_OK = (0.6, 1.6)   # traced area / AI area outside this range -> use the AI outline
EXTEND_FT = 15.0       # an open edge may carry on this far to the next real wall ...
MAX_EXTEND_SHARE = 0.75  # ... but no more than this share of the room's smaller side
LEAK_RETRY = 0.03      # open space cut back > this share of the room: retry with lighter wall lines


def _union_mask(polys, to_px, shape):
    m = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.array([to_px(x, y) for x, y in p], np.int32)], 1)
    return m


def _walls(page, s, pt_ft, barriers):
    """Wall mask for the whole page at render scale s (vector line work, or the image for scans)."""
    img, _ = render_page(page)
    H, W = img.shape[:2]
    walls = wall_mask_vector(page, s, min_stroke_pt=0.5)
    vector = walls is not None
    lighter = None
    if walls is None:
        walls = wall_mask(img)
    else:
        # walls drawn in medium-weight lines (e.g. BFT): long straight lines too. Only used for a
        # room the heavy lines don't close (more lines also means more chances to catch noise)
        long_lines = wall_mask_vector(page, s, min_stroke_pt=0.2, orth_only="long", min_len_pt=1.0 * pt_ft)
        if long_lines is not None:
            lighter = np.maximum(walls, long_lines)

    def finish(m):
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m = drop_specks(m, 2.0 * pt_ft * s)
        for (a, b) in barriers:
            cv2.line(m, (int(a[0] * s), int(a[1] * s)), (int(b[0] * s), int(b[1] * s)), 1, thickness=5)
        return m
    return finish(walls), (finish(lighter) if lighter is not None else None), vector


def _trace(walls, P, band, ext, gap_px, taken, others):
    """The room's open space, bounded by the walls near / beyond the AI outline P (crop masks).
    Returns (C, wall mask used, pixels cut back at open edges)."""
    outer = cv2.dilate(P, np.ones((2 * ext + 1, 2 * ext + 1), np.uint8))
    inner = cv2.erode(P, np.ones((2 * band + 1, 2 * band + 1), np.uint8))
    w = walls & outer & (1 - inner)                   # walls well inside the room (the noise) don't count
    w = np.maximum(w, taken)                          # a neighbour already placed is a wall too
    closed = cv2.dilate(w, np.ones((2 * gap_px + 1, 2 * gap_px + 1), np.uint8))
    free = outer & (1 - closed)
    n, lab = cv2.connectedComponents(free, connectivity=4)
    if n <= 1:
        return None, w, 0
    core = cv2.erode(P, np.ones((3, 3), np.uint8))
    counts = np.bincount(lab[core > 0], minlength=n)  # the open space that overlaps the AI's room most
    counts[0] = 0
    best = int(counts.argmax())
    if counts[best] == 0:
        return None, w, 0
    C = (lab == best).astype(np.uint8)
    C = cv2.dilate(C, np.ones((2 * gap_px + 1, 2 * gap_px + 1), np.uint8))   # back out to the wall faces
    C &= outer
    C[w > 0] = 0
    # beyond the AI outline: keep space that a real wall closes off (the AI was short); cut back
    # space that runs on with nothing to stop it, or that holds another room's label
    near = cv2.dilate(P, np.ones((2 * band + 1, 2 * band + 1), np.uint8))
    rim = outer & (1 - cv2.erode(outer, np.ones((3, 3), np.uint8)))
    extra = C & (1 - P)
    m, elab = cv2.connectedComponents(extra, connectivity=4)
    cut = set(np.unique(elab[(rim > 0) & (extra > 0)]).tolist())
    H, W = P.shape
    for (ox, oy) in others:
        if 0 <= oy < H and 0 <= ox < W and not near[oy, ox]:
            cut.add(int(elab[oy, ox]))
    cut.discard(0)
    trimmed = 0
    for j in cut:
        sel = elab == j
        trimmed += int(sel.sum())
        C[sel] = 0
    C = cv2.morphologyEx(C, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return C, w, trimmed


def guided_rooms(page, pt_ft, ai_rooms, barriers=(), gap_ft=3.75, label_pts=()):
    """Outlines for the AI-read rooms that came with a polygon.

    label_pts: where other rooms' labels sit (PDF points), so an outline isn't carried
    into a neighbouring room. Returns rooms like find_rooms makes them ({outline_pt, name,
    number, ...}) plus "ai_index" (which ai_rooms entry it came from) and, when the result
    is a fallback, "check"."""
    todo = [(i, a) for i, a in enumerate(ai_rooms or []) if a.get("polys")]
    if not todo:
        return []
    img, s = render_page(page)
    H, W = img.shape[:2]
    heavy, lighter, vector = _walls(page, s, pt_ft, barriers)
    hs, vs = axis_lines(page) if vector else raster_axis_lines(page)
    snap_tol = max(1.5, 2.6 / s)
    gap_px = max(1, int(round(gap_ft * pt_ft * s / 2)))
    taken = np.zeros((H, W), np.uint8)          # pixels already given to a room
    all_pts = list(label_pts) + [tuple(a["pt"]) for a in ai_rooms or [] if a.get("pt")]

    # smaller rooms first, so a big room's rough outline can't swallow a small neighbour
    def ai_area(a):
        return sum(polygon_area(p) for p in a["polys"])
    todo.sort(key=lambda t: ai_area(t[1]))

    out = []
    for i, a in todo:
        xs = [x for p in a["polys"] for x, _ in p]
        ys = [y for p in a["polys"] for _, y in p]
        side_pt = min(max(xs) - min(xs), max(ys) - min(ys))
        band_pt = min(max(BAND_FT * pt_ft, MIN_BAND_PT), MAX_BAND_SHARE * side_pt)
        band = max(2, int(round(band_pt * s)))
        ext = max(band, int(round(min(EXTEND_FT * pt_ft, MAX_EXTEND_SHARE * side_pt) * s)))
        pad = ext + gap_px + 4
        x0, y0 = max(0, int(min(xs) * s) - pad), max(0, int(min(ys) * s) - pad)
        x1, y1 = min(W, int(max(xs) * s) + pad), min(H, int(max(ys) * s) + pad)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        to_px = lambda x, y: (x * s - x0, y * s - y0)
        shape = (y1 - y0, x1 - x0)

        P = _union_mask(a["polys"], to_px, shape)
        P = cv2.morphologyEx(P, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))   # seal seams between sections
        ai_px = float(P.sum())
        others = []
        for (px_, py_) in all_pts:                   # labels of OTHER rooms (not inside this AI outline)
            qx, qy = int(px_ * s - x0), int(py_ * s - y0)
            if 0 <= qy < shape[0] and 0 <= qx < shape[1] and not P[qy, qx]:
                others.append((qx, qy))
        tk = taken[y0:y1, x0:x1]

        C, w, trimmed = _trace(heavy[y0:y1, x0:x1], P, band, ext, gap_px, tk, others)
        ok = C is not None and AREA_OK[0] <= C.sum() / max(ai_px, 1) <= AREA_OK[1]
        if lighter is not None and (not ok or trimmed > LEAK_RETRY * ai_px):
            # heavy lines don't close this room: try with the long, lighter lines as well
            C2, w2, t2 = _trace(lighter[y0:y1, x0:x1], P, band, ext, gap_px, tk, others)
            ok2 = C2 is not None and AREA_OK[0] <= C2.sum() / max(ai_px, 1) <= AREA_OK[1]
            if ok2 and (not ok or t2 < trimmed):
                C, w, trimmed, ok = C2, w2, t2, ok2
        if C is None:
            C = np.zeros(shape, np.uint8)

        cnts, _ = cv2.findContours(C, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        room = {"name": " ".join(x for x in (a.get("name"), a.get("number")) if x) or "ROOM",
                "number": a.get("number"), "ai_index": i, "guided": True}
        ai_px = float(P.sum())
        traced = max(cnts, key=cv2.contourArea) if cnts else None
        ratio = (cv2.contourArea(traced) / ai_px) if traced is not None and ai_px else 0
        if traced is not None and AREA_OK[0] <= ratio <= AREA_OK[1]:
            c = cv2.approxPolyDP(traced, max(1.5, 1.6 * s), True).reshape(-1, 2).astype(float)
            pts = [[round((px + x0 + 0.5) / s, 2), round((py + y0 + 0.5) / s, 2)] for px, py in c]
            pts = refine_outline(pts, hs, vs, snap_tol)
            mask = C
        else:
            # the drawing doesn't back the AI up: keep the AI outline, nudged onto nearby walls
            cnts, _ = cv2.findContours(P, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            c = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), max(2.0, 2.5 * s), True).reshape(-1, 2).astype(float)
            pts = [[round((px + x0 + 0.5) / s, 2), round((py + y0 + 0.5) / s, 2)] for px, py in c]
            pts = refine_outline(pts, hs, vs, band_pt)
            room["check"] = "Outline from the AI (the drawing's walls didn't match): check it"
            mask = P
        if len(pts) < 3 or polygon_area(pts) < 4 * pt_ft ** 2:
            continue
        room["outline_pt"] = pts
        taken[y0:y1, x0:x1] |= cv2.dilate(mask, np.ones((3, 3), np.uint8))
        out.append(room)
    return out


def covered_by(room, guided, share=0.5):
    """True if more than `share` of a traced room's area lies inside one of the guided rooms."""
    x0, y0, x1, y1 = bbox(room["outline_pt"])
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return False
    k = min(1.0, 400 / max(x1 - x0, y1 - y0))
    shape = (int((y1 - y0) * k) + 3, int((x1 - x0) * k) + 3)
    to_px = lambda x, y: ((x - x0) * k + 1, (y - y0) * k + 1)
    m = _union_mask([room["outline_pt"]], to_px, shape)
    total = m.sum()
    if not total:
        return False
    for g in guided:
        gx0, gy0, gx1, gy1 = bbox(g["outline_pt"])
        if gx1 < x0 or gx0 > x1 or gy1 < y0 or gy0 > y1:
            continue
        gm = _union_mask([g["outline_pt"]], to_px, shape)
        if (m & gm).sum() > share * total:
            return True
    return False
