"""Detect enclosed rooms on a plan page by tracing its wall lines.

All geometry returned is in PDF points (page coordinates), so it does not
depend on the resolution used for detection.
"""
import math

import cv2
import numpy as np
import pymupdf


_CACHE = {}


def clear_cache():
    _CACHE.clear()


def _cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def _page_key(page):
    return (page.parent.name, id(page.parent), page.number, page.rotation)


def render_page(page, max_pixels=10e6, max_scale=1.5):
    return _cached(("render", _page_key(page), max_pixels, max_scale),
                   lambda: _render_page(page, max_pixels, max_scale))


def _render_page(page, max_pixels=10e6, max_scale=1.5):
    """Render a page to an RGB array. Returns (img, scale) with px = pt * scale."""
    w, h = page.rect.width, page.rect.height
    scale = min(max_scale, (max_pixels / (w * h)) ** 0.5)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    return img.copy(), scale


def wall_mask(img, threshold=200, thin=1, max_saturation=70):
    """Dark, uncoloured strokes thicker than (2*thin+1) px."""
    rgb = img.astype(np.int16)
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    sat = rgb.max(-1) - rgb.min(-1)
    dark = ((lum < threshold) & (sat < max_saturation)).astype(np.uint8)
    if thin > 0:
        k = np.ones((2 * thin + 1, 2 * thin + 1), np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
    return dark


def _gray(c):
    return c is not None and len(c) >= 3 and max(c) - min(c) < 0.3


def wall_mask_vector(page, scale, min_stroke_pt=0.5, max_fill_lum=0.85, max_stroke_lum=0.5,
                     orth_only=False, min_len_pt=0, curves=True):
    key = ("walls", _page_key(page), scale, min_stroke_pt, max_fill_lum, max_stroke_lum, orth_only, min_len_pt, curves)
    m = _cached(key, lambda: _wall_mask_vector(page, scale, min_stroke_pt, max_fill_lum, max_stroke_lum,
                                               orth_only, min_len_pt, curves))
    return None if m is None else m.copy()


def _drawings(page):
    return _cached(("drawings", _page_key(page)), page.get_drawings)


def _x_mark_piece(d, max_pt=24.0):
    """Is this filled shape (part of) an X symbol, e.g. "change in floor finish"? Such symbols are
    drawn at paper size (under about 1/3" whatever the scale) as small filled shapes whose edges
    all run at about 45° (two V halves on IA310). Wall poché and columns run square."""
    r = d["rect"]
    if max(r.width, r.height) > max_pt or min(r.width, r.height) < 2.0:
        return False
    items = d["items"]
    if not 3 <= len(items) <= 12 or any(it[0] != "l" for it in items):
        return False
    for it in items:
        a, b = it[1], it[2]
        dx, dy = abs(b.x - a.x), abs(b.y - a.y)
        if dx + dy < 0.2:
            continue
        ang = math.degrees(math.atan2(dy, dx))
        if not 35 <= ang <= 55:
            return False
    return True


def _dash_strokes(drawings, max_len=30.0, min_run=4):
    """ids of the drawings that are dashes of a dashed line drawn dash by dash (IA310's heavy
    finish boundary: 18 pt dashes): single short straight strokes, each in line with another of
    about the same length a gap of up to 3 dash lengths away, in runs of at least min_run. A lone
    short stroke, or two wall pieces either side of a doorway, stay."""
    segs = []
    for d in drawings:
        its = d["items"]
        if d.get("type") != "s" or len(its) != 1 or its[0][0] != "l":
            continue
        a, b = its[0][1], its[0][2]
        L = math.hypot(b.x - a.x, b.y - a.y)
        if 0.3 < L <= max_len:
            segs.append((id(d), a, b, L))
    cell = 4 * max_len
    grid = {}
    for k, (_i, a, b, _L) in enumerate(segs):
        grid.setdefault((int((a.x + b.x) / 2 // cell), int((a.y + b.y) / 2 // cell)), []).append(k)
    parent = list(range(len(segs)))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for k, (_i, a, b, L) in enumerate(segs):
        ux, uy = (b.x - a.x) / L, (b.y - a.y) / L
        mx, my = (a.x + b.x) / 2, (a.y + b.y) / 2
        gx, gy = int(mx // cell), int(my // cell)
        for nx in (gx - 1, gx, gx + 1):
            for ny in (gy - 1, gy, gy + 1):
                for j in grid.get((nx, ny), ()):
                    if j <= k:
                        continue
                    _i2, a2, b2, L2 = segs[j]
                    if abs(L2 - L) > 0.25 * max(L, L2):
                        continue
                    if abs((b2.x - a2.x) / L2 * uy - (b2.y - a2.y) / L2 * ux) > 0.05:
                        continue                      # not parallel
                    vx, vy = (a2.x + b2.x) / 2 - mx, (a2.y + b2.y) / 2 - my
                    along, off = abs(vx * ux + vy * uy), abs(vx * uy - vy * ux)
                    if off < 0.3 and L < along <= 4 * L:
                        parent[find(j)] = find(k)
    runs = {}
    for k in range(len(segs)):
        runs.setdefault(find(k), []).append(k)
    return {segs[k][0] for ks in runs.values() if len(ks) >= min_run for k in ks}


def _wall_mask_vector(page, scale, min_stroke_pt, max_fill_lum, max_stroke_lum, orth_only, min_len_pt, curves=True,
                      skip=None):
    """Wall mask drawn from the PDF's vector line work instead of the rendered image.

    Text is never drawn, and thin strokes (hatching, dimensions, door swings, tag
    bubbles, furniture) and coloured markups are skipped. Only heavy dark strokes
    and dark/grey fills (wall poché, columns) are kept.
    Returns None if the page has little vector line work (e.g. a scanned sheet).
    """
    drawings = _drawings(page)
    if len(drawings) < 200:
        return None
    out = pymupdf.open()
    q = out.new_page(width=page.rect.width, height=page.rect.height)
    sh = q.new_shape()
    kept = 0
    erase_min = 0.0003 * page.rect.width * page.rect.height
    big_fill = 0.01 * page.rect.width * page.rect.height
    dashes = _dash_strokes(drawings)
    thin_white = []                                   # thin "white" fills: some PDFs (IA310) report a
    for d in drawings:                                # special print colour as white: checked below
        if id(d) in dashes:
            continue                                  # a heavy dashed line (a finish/zone boundary), not a wall
        col, fill, w = d.get("color"), d.get("fill"), d.get("width") or 0
        r_ = d["rect"]
        if (d.get("type") == "f" and fill and min(fill) >= 0.97 and min(r_.width, r_.height) <= 1.0
                and max(r_.width, r_.height) >= 3.0):
            thin_white.append(d)
            continue
        stroke_ok = _gray(col) and w >= min_stroke_pt and sum(col) / 3 <= max_stroke_lum
        fill_ok = _gray(fill) and sum(fill) / 3 <= max_fill_lum
        if (not stroke_ok and fill and min(fill) >= 0.97 and (d.get("fill_opacity") or 1) >= 0.99
                and d["rect"].width * d["rect"].height >= erase_min):
            # a large white fill paints over what was drawn before it (e.g. room fills laid on a
            # grey background sheet): erase, so the background doesn't read as a solid wall
            for it in d["items"]:
                if it[0] == "l":
                    sh.draw_line(it[1], it[2])
                elif it[0] == "re":
                    sh.draw_rect(it[1])
                elif it[0] == "qu":
                    sh.draw_quad(it[1])
                elif it[0] == "c":
                    sh.draw_bezier(it[1], it[2], it[3], it[4])
            sh.finish(color=None, fill=(1, 1, 1), closePath=True)
            continue
        if fill_ok and len(d["items"]) <= 8 and d["rect"].width * d["rect"].height >= big_fill:
            # a big plain grey shape (a background tone, a shaded zone) is not a wall: keep its edge only
            fill_ok = False
            stroke_ok = True
        if not (stroke_ok or fill_ok):
            continue
        if fill_ok and _x_mark_piece(d):
            continue                                  # a "change in floor finish" X, not a wall
        for it in d["items"]:
            if skip is not None and stroke_ok and not fill_ok and skip(it):
                continue                              # e.g. a tag's outline (ai_outline.tag_filter)
            if orth_only and stroke_ok and not fill_ok:
                # walls run square to the sheet; hatching is diagonal, equipment is short/curved
                if it[0] == "l":
                    a, b = it[1], it[2]
                    square = abs(a.x - b.x) < 0.5 or abs(a.y - b.y) < 0.5
                    if (square or orth_only == "long") and abs(a - b) >= min_len_pt:
                        sh.draw_line(a, b)
                elif it[0] == "re" and max(it[1].width, it[1].height) >= min_len_pt:
                    sh.draw_rect(it[1])
                continue
            if it[0] == "l":
                sh.draw_line(it[1], it[2])
            elif it[0] == "re":
                sh.draw_rect(it[1])
            elif it[0] == "qu":
                sh.draw_quad(it[1])
            elif it[0] == "c" and (curves or fill_ok):     # curves=False drops door swings
                sh.draw_bezier(it[1], it[2], it[3], it[4])
        sh.finish(color=(0, 0, 0) if stroke_ok else None, fill=(0, 0, 0) if fill_ok else None,
                  width=max(w, 0.6), closePath=d.get("closePath", False))
        kept += 1
    if kept < 50:
        return None
    sh.commit()
    pix = q.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False, colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    mask = (img < 160).astype(np.uint8)
    if thin_white:
        # thin lines the file calls white: where the rendered sheet shows them dark (wall lines
        # printed in a special colour), they're walls
        q2 = out.new_page(width=page.rect.width, height=page.rect.height)
        sh2 = q2.new_shape()
        for d in thin_white:
            sh2.draw_rect(d["rect"])
        sh2.finish(color=(0, 0, 0), fill=(0, 0, 0), width=0.6)
        sh2.commit()
        p2 = q2.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False, colorspace=pymupdf.csGRAY)
        where = np.frombuffer(p2.samples, np.uint8).reshape(p2.height, p2.width) < 160
        p3 = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False, colorspace=pymupdf.csGRAY)
        real = np.frombuffer(p3.samples, np.uint8).reshape(p3.height, p3.width)
        if real.shape == mask.shape:
            mask |= (where & (real < 128)).astype(np.uint8)
    return mask


def drop_specks(mask, max_side_px):
    """Remove small isolated marks (label underlines, symbols, stray dots) that aren't walls."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) < max_side_px
    out = mask.copy()
    out[small[lab]] = 0
    return out


def drop_thick(mask, max_px):
    """Clear solid areas thicker than max_px (about 3.5 ft): no wall is that thick, so it's a
    shaded zone (a floor tinted to show the scope, a hidden fill), not a wall. Its edge is kept
    as a boundary: a shaded floor stops there."""
    k = max(3, int(round(max_px)) | 1)
    core = cv2.erode(mask, np.ones((k, k), np.uint8))
    if not core.any():
        return mask
    blob = (cv2.dilate(core, np.ones((k, k), np.uint8)) > 0) & (mask > 0)
    out = mask.copy()
    out[blob] = 0
    edge = blob & ~(cv2.erode(blob.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    out[edge] = 1                                     # its edge is where the shaded room stops
    return out


def detect_rooms(page, threshold=200, thin=1, gap_ft=3.75, pt_ft=None, gap_px=None,
                 min_sf=30, barriers=(), max_pixels=10e6, mode="auto", min_stroke_pt=0.5,
                 fine_stroke_pt=0.3, speck_ft=2.0, orth_only=False, min_len_ft=2.0):
    """Find enclosed spaces on a page.

    mode:     "vector" traces walls from the PDF line work (ignores text and hatching),
              "raster" from the rendered image (for scans), "auto" picks vector when it can.
    In vector mode two passes run: walls drawn at >= min_stroke_pt first, then any
    area that leaked is filled in from a pass that also uses lighter lines
    (>= fine_stroke_pt), which seals more rooms but can split a room along finish lines.
    gap_ft:   openings up to about this width (feet) are bridged while searching.
    barriers: extra wall segments [[[x0,y0],[x1,y1]], ...] in PDF points,
              used to close doorways that leak.
    Returns a list of dicts with 'outline_pt' and 'area_sf' (if pt_ft is known).
    """
    img, s = render_page(page, max_pixels=max_pixels)
    H, W = img.shape[:2]
    if gap_px is None:
        gap_px = int(round(gap_ft * pt_ft * s / 2)) if pt_ft else int(round(13.5 * s))
    gap_px = max(0, gap_px)

    def prep(walls):
        if walls.shape != (H, W):
            walls = cv2.resize(walls, (W, H), interpolation=cv2.INTER_NEAREST)
        if speck_ft and pt_ft:
            walls = drop_specks(walls, speck_ft * pt_ft * s)
        if pt_ft:
            walls = drop_thick(walls, 3.5 * pt_ft * s)
        for (a, b) in barriers:
            cv2.line(walls, (int(a[0] * s), int(a[1] * s)), (int(b[0] * s), int(b[1] * s)), 1, thickness=5)
        return walls

    vopts = {"orth_only": orth_only, "min_len_pt": (min_len_ft * pt_ft if pt_ft else 18) if orth_only else 0}
    coarse = wall_mask_vector(page, s, min_stroke_pt=min_stroke_pt, **vopts) if mode in ("auto", "vector") else None
    if coarse is None:
        rooms = _trace_rooms(prep(wall_mask(img, threshold, thin)), s, gap_px, pt_ft, min_sf)
    else:
        rooms = _trace_rooms(prep(coarse), s, gap_px, pt_ft, min_sf)
        if fine_stroke_pt and fine_stroke_pt < min_stroke_pt:
            fine = wall_mask_vector(page, s, min_stroke_pt=fine_stroke_pt, **vopts)
            for r in _trace_rooms(prep(fine), s, gap_px, pt_ft, min_sf):
                lx, ly = label_point(r["outline_pt"])
                host = next((q for q in rooms if point_in_polygon(lx, ly, q["outline_pt"])), None)
                if host is None:
                    rooms.append(r)
                else:
                    host.setdefault("parts", []).append(r)   # labels.py may split these out

    rooms.sort(key=lambda r: (round(bbox(r["outline_pt"])[1] / 100), bbox(r["outline_pt"])[0]))
    return rooms


def _trace_rooms(walls, s, gap_px, pt_ft, min_sf):
    H, W = walls.shape
    closed = cv2.dilate(walls, np.ones((2 * gap_px + 1, 2 * gap_px + 1), np.uint8)) if gap_px else walls
    free = (closed == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)

    px_per_ft = pt_ft * s if pt_ft else None
    min_px = min_sf * px_per_ft ** 2 if px_per_ft else 2800 * (s / 1.25) ** 2
    grow = np.ones((2 * gap_px + 1, 2 * gap_px + 1), np.uint8)
    eps = max(1.5, 1.6 * s)
    rooms = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue  # leaks to the sheet edge: not an enclosed room
        if area < 0.5 * min_px or area > 0.25 * W * H:
            continue
        pad = gap_px + 3
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        m = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        if gap_px:
            m = cv2.dilate(m, grow)          # grow back out to the wall faces
        m[walls[y0:y1, x0:x1] > 0] = 0
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < min_px:
            continue
        poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float)
        poly[:, 0] = (poly[:, 0] + x0 + 0.5) / s
        poly[:, 1] = (poly[:, 1] + y0 + 0.5) / s
        pts = [[round(float(px), 2), round(float(py), 2)] for px, py in poly]
        rooms.append({"outline_pt": pts,
                      "area_sf": round(polygon_area(pts) / pt_ft ** 2, 1) if pt_ft else None})
    return rooms


def subtract_polygons(poly, holes, k=3.0, eps_pt=0.75, keep_point=None):
    """poly minus holes (all in PDF points), as the largest remaining polygon,
    or the one containing keep_point."""
    x0, y0, x1, y1 = bbox(poly)
    w, h = int((x1 - x0) * k) + 4, int((y1 - y0) * k) + 4
    to_px = lambda pts: np.array([[(x - x0) * k + 2, (y - y0) * k + 2] for x, y in pts], np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [to_px(poly)], 1)
    hm = np.zeros_like(m)
    cv2.fillPoly(hm, [to_px(q) for q in holes], 1)
    hm = cv2.dilate(hm, np.ones((5, 5), np.uint8))          # no slivers along shared walls
    m[hm > 0] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    if keep_point is not None:
        kp = ((keep_point[0] - x0) * k + 2, (keep_point[1] - y0) * k + 2)
        cnts = [c for c in cnts if cv2.pointPolygonTest(c, kp, True) > -3 * k]
        if not cnts:
            return None
    c = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), eps_pt * k, True).reshape(-1, 2).astype(float)
    return [[round((px - 2) / k + x0, 2), round((py - 2) / k + y0, 2)] for px, py in c]


def polygon_area(pts):
    a = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i - 1]
        x2, y2 = pts[i]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


def point_in_polygon(x, y, pts):
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def label_point(pts):
    """A point inside the polygon, near its centre, for placing text."""
    a = cx = cy = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        f = x0 * y1 - x1 * y0
        a += f
        cx += (x0 + x1) * f
        cy += (y0 + y1) * f
    if a:
        cx, cy = cx / (3 * a), cy / (3 * a)
        if point_in_polygon(cx, cy, pts):
            return cx, cy
    x0, y0, x1, y1 = bbox(pts)
    y = (y0 + y1) / 2
    xs = sorted(pts[i - 1][0] + (y - pts[i - 1][1]) * (pts[i][0] - pts[i - 1][0]) / (pts[i][1] - pts[i - 1][1])
                for i in range(len(pts)) if (pts[i][1] > y) != (pts[i - 1][1] > y))
    best, bw = ((x0 + x1) / 2, y), -1
    for i in range(0, len(xs) - 1, 2):
        if xs[i + 1] - xs[i] > bw:
            bw, best = xs[i + 1] - xs[i], ((xs[i] + xs[i + 1]) / 2, y)
    return best


# ---------------------------------------------------------------- columns
def room_area_pt(room):
    """Floor area of a room in square points: its outline minus its columns."""
    return polygon_area(room["outline_pt"]) - sum(polygon_area(h) for h in room.get("holes_pt") or [])


def find_columns(page, scale, pt_ft, min_side_ft=0.6, max_side_ft=4.0, max_aspect=2.2, min_solid=0.7,
                 **mask_opts):
    """Free-standing columns: solid, roughly square marks between ~7 in and 4 ft across,
    drawn as filled shapes (poché). Columns built into a wall are already part of the
    room outline. Returns polygons in PDF points."""
    m = wall_mask_vector(page, scale, **mask_opts)
    if m is None:
        return []
    k = pt_ft * scale                                  # px per ft
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        lo, hi = min(w, h) / k, max(w, h) / k
        if lo < min_side_ft or hi > max_side_ft or hi / max(lo, 1e-6) > max_aspect:
            continue
        if area / float(w * h) < min_solid:
            continue                                   # an outline (equipment, symbol), not a solid column
        sub = np.pad((lab[y:y + h, x:x + w] == i).astype(np.uint8), 2)
        r = max(1, int(round(0.15 * k)))              # trim thin bits touching it (arrowheads, leaders)
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
        x, y = x - 2, y - 2
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        poly = cv2.approxPolyDP(c, max(1.0, 0.8 * scale), True).reshape(-1, 2).astype(float)
        if len(poly) < 3:
            continue
        pts = [[round((px + x + 0.5) / scale, 2), round((py + y + 0.5) / scale, 2)] for px, py in poly]
        out.append(pts)
    return out


def attach_columns(rooms, columns):
    """Add each column lying inside a room to that room's holes_pt."""
    for r in rooms:
        r.setdefault("holes_pt", [])
    for col in columns:
        cx = sum(p[0] for p in col) / len(col)
        cy = sum(p[1] for p in col) / len(col)
        for r in rooms:
            if point_in_polygon(cx, cy, r["outline_pt"]) and all(point_in_polygon(p[0], p[1], r["outline_pt"]) for p in col):
                r["holes_pt"].append(col)
                break
    return rooms


# ---------------------------------------------------------------- exact edges
def find_circles(page, min_r_pt=1.5, max_r_pt=300):
    """Circles in the drawing's vector line work (round columns), as (cx, cy, r) in PDF points.
    A circle is a closed path of Bezier curves (usually 4) or of many short lines whose points
    all sit the same distance from the middle."""
    import math
    out = []
    for d in _drawings(page):
        items = d["items"]
        if len(items) < 4:
            continue
        pts = []
        for it in items:
            if it[0] == "c":
                pts += [(it[1].x, it[1].y), (it[4].x, it[4].y)]
                # the curve's midpoint (t = 0.5) as well, so a lens or rounded rectangle doesn't pass
                mx = (it[1].x + 3 * it[2].x + 3 * it[3].x + it[4].x) / 8
                my = (it[1].y + 3 * it[2].y + 3 * it[3].y + it[4].y) / 8
                pts.append((mx, my))
            elif it[0] == "l":
                pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
            else:
                pts = []
                break
        kinds = {it[0] for it in items}
        if not pts or (kinds == {"l"} and len(items) < 12):
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if w <= 0 or h <= 0 or not 0.9 < w / h < 1.1:
            continue
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        ds = [math.hypot(x - cx, y - cy) for x, y in pts]
        r = sum(ds) / len(ds)
        if not min_r_pt <= r <= max_r_pt or max(abs(x - r) for x in ds) > 0.06 * r:
            continue
        if not any(abs(cx - a) < 0.5 and abs(cy - b) < 0.5 and abs(r - c) < 0.5 for a, b, c in out):
            out.append((cx, cy, r))
    return out


def axis_lines(page, min_len_pt=4, min_width=0.2):
    """Horizontal and vertical line work on the page, as (coord, lo, hi) sorted by coord.
    Used to snap traced outlines onto the drawing's exact wall faces."""
    def build():
        hs, vs = [], []

        def add(a, b):
            if abs(a[1] - b[1]) < 0.3 and abs(a[0] - b[0]) >= min_len_pt:
                hs.append(((a[1] + b[1]) / 2, min(a[0], b[0]), max(a[0], b[0])))
            elif abs(a[0] - b[0]) < 0.3 and abs(a[1] - b[1]) >= min_len_pt:
                vs.append(((a[0] + b[0]) / 2, min(a[1], b[1]), max(a[1], b[1])))

        for d in _drawings(page):
            col, fill, w = d.get("color"), d.get("fill"), d.get("width") or 0
            if not ((_gray(col) and w >= min_width) or _gray(fill)):
                continue
            for it in d["items"]:
                if it[0] == "l":
                    add((it[1].x, it[1].y), (it[2].x, it[2].y))
                elif it[0] == "re":
                    r = it[1]
                    for a, b in (((r.x0, r.y0), (r.x1, r.y0)), ((r.x1, r.y0), (r.x1, r.y1)),
                                 ((r.x1, r.y1), (r.x0, r.y1)), ((r.x0, r.y1), (r.x0, r.y0))):
                        add(a, b)
                elif it[0] == "qu":
                    q = it[1]
                    pts = [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)]
                    for j in range(4):
                        add(pts[j], pts[(j + 1) % 4])
        hs.sort()
        vs.sort()
        return hs, vs
    return _cached(("axis", _page_key(page), min_len_pt, min_width), build)


def _snap(c, lo, hi, segs, tol):
    """The coordinate of the nearest line within tol of c that runs alongside [lo, hi]."""
    import bisect
    keys = _cached(("keys", id(segs)), lambda: [s[0] for s in segs])
    i0, i1 = bisect.bisect_left(keys, c - tol), bisect.bisect_right(keys, c + tol)
    span = hi - lo
    best = None
    for s in segs[i0:i1]:
        overlap = min(hi, s[2]) - max(lo, s[1])
        if overlap >= 0.3 * span or overlap >= 0.8 * (s[2] - s[1]) and overlap > 0:
            d = abs(s[0] - c)
            if best is None or d < best[0]:
                best = (d, s[0])
    return best[1] if best else c


def refine_outline(pts, hsegs, vsegs, tol_pt):
    """Square up near-horizontal/vertical edges and move them onto the nearest wall line
    from the PDF, then rebuild the corners as exact intersections."""
    n = len(pts)
    if n < 3:
        return pts
    edges = []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        if abs(dy) <= 0.06 * abs(dx):
            c = _snap((a[1] + b[1]) / 2, min(a[0], b[0]), max(a[0], b[0]), hsegs, tol_pt)
            edges.append(("h", c, a, b))
        elif abs(dx) <= 0.06 * abs(dy):
            c = _snap((a[0] + b[0]) / 2, min(a[1], b[1]), max(a[1], b[1]), vsegs, tol_pt)
            edges.append(("v", c, a, b))
        else:
            edges.append(("g", None, a, b))
    # merge neighbouring edges that ended up on the same line
    merged = []
    for e in edges:
        if merged and e[0] in "hv" and merged[-1][0] == e[0] and abs(merged[-1][1] - e[1]) < 0.05:
            merged[-1] = (e[0], merged[-1][1], merged[-1][2], e[3])
        else:
            merged.append(e)
    if len(merged) > 1 and merged[0][0] in "hv" and merged[0][0] == merged[-1][0] and abs(merged[0][1] - merged[-1][1]) < 0.05:
        last = merged.pop()
        merged[0] = (last[0], last[1], last[2], merged[0][3])
    if len(merged) < 3:
        return pts

    def line(e):
        if e[0] == "h":
            return (0.0, 1.0, e[1])          # y = c
        if e[0] == "v":
            return (1.0, 0.0, e[1])          # x = c
        (x1, y1), (x2, y2) = e[2], e[3]
        a, b = y2 - y1, x1 - x2
        return (a, b, a * x1 + b * y1)

    out = []
    m = len(merged)
    for i in range(m):
        p, q = merged[i - 1], merged[i]
        a1, b1, c1 = line(p)
        a2, b2, c2 = line(q)
        det = a1 * b2 - a2 * b1
        if abs(det) > 1e-6 * (abs(a1) + abs(b1)) * (abs(a2) + abs(b2)):
            out.append([round((c1 * b2 - c2 * b1) / det, 2), round((a1 * c2 - a2 * c1) / det, 2)])
        else:   # parallel neighbours (a jog): keep both ends at the original corner
            v = q[2]
            if p[0] == "h":
                out += [[round(v[0], 2), round(p[1], 2)], [round(v[0], 2), round(q[1], 2)]]
            elif p[0] == "v":
                out += [[round(p[1], 2), round(v[1], 2)], [round(q[1], 2), round(v[1], 2)]]
            else:
                out.append([round(v[0], 2), round(v[1], 2)])
    # a corner that moved a long way means the snap went wrong: keep the traced shape
    if polygon_area(out) < 0.85 * polygon_area(pts) or polygon_area(out) > 1.15 * polygon_area(pts):
        return pts
    return out


def raster_axis_lines(page, min_len_pt=12, max_band_pt=30, max_pixels=40e6):
    """Snap lines for scanned sheets (a picture, no line work): the straight horizontal and
    vertical strokes found in the image. Thick strokes (walls) give both faces; thin ones
    give their centre line. Same format as axis_lines: (coord, lo, hi) in PDF points."""
    def build():
        # render near the scan's own resolution so edges land on real pixels
        native = max((i["width"] / max(1e-6, i["bbox"][2] - i["bbox"][0]) for i in page.get_image_info()), default=1.5)
        w, h = page.rect.width, page.rect.height
        s = min(native, 4.0, (max_pixels / (w * h)) ** 0.5)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(s, s), alpha=False, colorspace=pymupdf.csGRAY)
        g = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
        thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = (g < min(thr, 170)).astype(np.uint8)
        L = max(9, int(min_len_pt * s))
        hs, vs = [], []
        for horiz in (True, False):
            k = np.ones((1, L), np.uint8) if horiz else np.ones((L, 1), np.uint8)
            runs = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
            n, lab, st, _ = cv2.connectedComponentsWithStats(runs, connectivity=4)
            for i in range(1, n):
                x, y, bw, bh, _a = st[i]
                band = bh if horiz else bw
                if band / s > max_band_pt:
                    continue                       # a filled area, not a line
                if horiz:
                    lo, hi = x / s, (x + bw) / s
                    faces = [(y + bh / 2) / s] if bh <= 2 else [y / s, (y + bh) / s]
                    hs += [(round(c, 2), round(lo, 2), round(hi, 2)) for c in faces]
                else:
                    lo, hi = y / s, (y + bh) / s
                    faces = [(x + bw / 2) / s] if bw <= 2 else [x / s, (x + bw) / s]
                    vs += [(round(c, 2), round(lo, 2), round(hi, 2)) for c in faces]
        hs.sort()
        vs.sort()
        return hs, vs
    return _cached(("raster-axis", _page_key(page), min_len_pt, max_band_pt), build)


def union_polygons(polys, bridge_pt=6.0, k=3.0, eps_pt=0.75):
    """One outline for rooms that touch (e.g. a room split by a finish line), or None if
    they don't meet. Gaps up to bridge_pt (a drawn line between them) are closed."""
    pts = [p for poly in polys for p in poly]
    x0, y0, x1, y1 = bbox(pts)
    pad = bridge_pt + 4
    w, h = int((x1 - x0 + 2 * pad) * k), int((y1 - y0 + 2 * pad) * k)
    to_px = lambda q: np.array([[(x - x0 + pad) * k, (y - y0 + pad) * k] for x, y in q], np.int32)
    m = np.zeros((h, w), np.uint8)
    for poly in polys:
        cv2.fillPoly(m, [to_px(poly)], 1)
    r = max(1, int(round(bridge_pt * k / 2)))
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cnts) != 1:
        return None
    c = cv2.approxPolyDP(cnts[0], eps_pt * k, True).reshape(-1, 2).astype(float)
    return [[round(px / k + x0 - pad, 2), round(py / k + y0 - pad, 2)] for px, py in c]


def filled_polygons(page, min_area_pt2, max_area_pt2):
    """Closed filled shapes on the page, as [outline, ...] in PDF points.

    Colour-fill plans (Revit room fills, finish zones) draw each room as one filled shape
    that follows its walls exactly. Each subpath becomes its own polygon."""
    out = []
    for d in _drawings(page):
        if d.get("fill") is None or (d.get("fill_opacity") or 1) <= 0.02:
            continue
        r = d["rect"]
        if r.width * r.height < min_area_pt2:
            continue
        subs, cur = [], []
        for it in d["items"]:
            if it[0] == "re":
                q = it[1]
                subs.append([[q.x0, q.y0], [q.x1, q.y0], [q.x1, q.y1], [q.x0, q.y1]])
                continue
            if it[0] == "qu":
                q = it[1]
                subs.append([[p.x, p.y] for p in (q.ul, q.ur, q.lr, q.ll)])
                continue
            a, b = it[1], it[-1]
            if cur and abs(cur[-1][0] - a.x) + abs(cur[-1][1] - a.y) > 0.5:
                subs.append(cur)
                cur = []
            if not cur:
                cur.append([a.x, a.y])
            if it[0] == "c":                       # a curve: a few points along it
                p0, p1, p2, p3 = it[1], it[2], it[3], it[4]
                for t in (0.25, 0.5, 0.75):
                    u = 1 - t
                    cur.append([u ** 3 * p0.x + 3 * u * u * t * p1.x + 3 * u * t * t * p2.x + t ** 3 * p3.x,
                                u ** 3 * p0.y + 3 * u * u * t * p1.y + 3 * u * t * t * p2.y + t ** 3 * p3.y])
            cur.append([b.x, b.y])
        if cur:
            subs.append(cur)
        for s in subs:
            if len(s) >= 3 and min_area_pt2 <= polygon_area(s) <= max_area_pt2:
                out.append([[round(x, 2), round(y, 2)] for x, y in s])
    return out


_TRACE_MASK = {}


def _trace_mask(page, s):
    """Wall mask for the draw tool's tracing: kept between requests (one page at a time)."""
    key = (page.parent.name, page.number, page.rotation, round(s, 4))
    if key not in _TRACE_MASK:
        _TRACE_MASK.clear()
        _TRACE_MASK[key] = _wall_mask_vector(page, s, 0.3, 0.85, 0.5, False, 0, curves=False)
    return _TRACE_MASK[key]


def trace_between(page, a, b, pt_ft, max_ratio=1.6):
    """The wall face from corner a to corner b (PDF points), for drawing a room by its corners.

    Follows the inside face of the walls between the two clicks, around pilasters, jogs and
    recesses, and squares each run onto the PDF's own wall line. Returns the points from a to
    b (a and b included), or None when no single wall face joins them (an opening between
    them, or the clicks are on different walls): the edge is then drawn straight."""
    W, H = page.rect.width, page.rect.height
    s = min(3.0, max(1.0, 30.0 / pt_ft), (40e6 / (W * H)) ** 0.5)
    m = _trace_mask(page, s)
    if m is None:
        return None
    d_pt = math.dist(a, b)
    if d_pt < 0.25 * pt_ft:
        return None
    pad = 3 * pt_ft
    x0 = int(max(0, (min(a[0], b[0]) - pad) * s))
    y0 = int(max(0, (min(a[1], b[1]) - pad) * s))
    x1 = int(min(m.shape[1], (max(a[0], b[0]) + pad) * s))
    y1 = int(min(m.shape[0], (max(a[1], b[1]) + pad) * s))
    wall = m[y0:y1, x0:x1]
    free = (wall == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(free, connectivity=4)
    face = free & cv2.dilate(wall, np.ones((3, 3), np.uint8))
    r = max(4, int(0.6 * pt_ft * s))              # a click within ~7" of the wall face

    def near(p):
        """{component label: (distance, (x, y))} for face pixels near p."""
        px, py = p[0] * s - x0, p[1] * s - y0
        ix, iy = int(px), int(py)
        wy0, wy1, wx0, wx1 = max(0, iy - r), min(face.shape[0], iy + r + 1), max(0, ix - r), min(face.shape[1], ix + r + 1)
        ys, xs = np.nonzero(face[wy0:wy1, wx0:wx1])
        out = {}
        for yy, xx in zip(ys + wy0, xs + wx0):
            dd = math.hypot(xx - px, yy - py)
            L = lab[yy, xx]
            if dd <= r and (L not in out or dd < out[L][0]):
                out[L] = (dd, (xx, yy))
        return out
    na, nb = near(a), near(b)
    both = [L for L in na if L in nb]
    if not both:
        return None
    L = min(both, key=lambda k: na[k][0] + nb[k][0])
    pa, pb = np.array(na[L][1]), np.array(nb[L][1])
    contours, _ = cv2.findContours((lab == L).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    best = None
    for c in contours:
        c = c[:, 0, :]
        if len(c) < 4:
            continue
        ia = int(np.argmin(((c - pa) ** 2).sum(1)))
        ib = int(np.argmin(((c - pb) ** 2).sum(1)))
        da, db = np.hypot(*(c[ia] - pa)), np.hypot(*(c[ib] - pb))
        if da > 2.5 or db > 2.5:
            continue
        N = len(c)
        fwd = c[np.arange(ia, ia + ((ib - ia) % N) + 1) % N]
        bwd = c[np.arange(ia, ia - ((ia - ib) % N) - 1, -1) % N]
        for arc in (fwd, bwd):
            ln = np.hypot(*np.diff(arc, axis=0).T).sum()
            if best is None or ln < best[0]:
                best = (ln, arc)
    if best is None:
        return None
    ln, arc = best
    if ln > max_ratio * d_pt * s + pt_ft * s:
        return None
    hgt, wid = wall.shape
    edge = ((arc[:, 0] <= 0) & (x0 > 0)) | ((arc[:, 1] <= 0) & (y0 > 0)) | \
           ((arc[:, 0] >= wid - 1) & (x1 < m.shape[1])) | ((arc[:, 1] >= hgt - 1) & (y1 < m.shape[0]))
    if edge.any():                                 # ran off the searched area: not one wall
        return None
    simp = cv2.approxPolyDP(arc.reshape(-1, 1, 2).astype(np.int32), 1.2, False)[:, 0, :]
    pts = [((x + x0 + 0.5) / s, (y + y0 + 0.5) / s) for x, y in simp]
    pts[0], pts[-1] = tuple(a), tuple(b)
    pts = _drop_blips(pts, 0.4 * pt_ft)
    if len(pts) <= 2:
        return [list(a), list(b)]
    return _square_path(page, pts, tol_pt=max(2.5 / s, 0.06 * pt_ft))


def _drop_blips(pts, size):
    """Cut out small bumps along a wall (an outlet or a pipe touching the face): any run of
    points that leaves the wall and comes back within a box `size` across."""
    out = list(pts)
    i = 0
    while i < len(out) - 2:
        cut = None
        for j in range(min(len(out) - 1, i + 8), i + 1, -1):
            xs = [p[0] for p in out[i:j + 1]]
            ys = [p[1] for p in out[i:j + 1]]
            if max(xs) - min(xs) <= size and max(ys) - min(ys) <= size:
                cut = j
                break
        if cut and i > 0 and cut < len(out) - 1:
            # keep the wall's line: the points either side of the bump stay
            del out[i + 1:cut]
        i += 1
    return out


def _square_path(page, pts, tol_pt):
    """Square up an open path's near-horizontal/vertical runs onto the PDF's own lines and
    rebuild its inner corners as exact intersections. The two ends stay where they are."""
    hs, vs = axis_lines(page, min_len_pt=2)
    runs = []
    for p, q in zip(pts, pts[1:]):
        dx, dy = q[0] - p[0], q[1] - p[1]
        if abs(dy) <= 0.08 * abs(dx):
            runs.append(["h", _snap((p[1] + q[1]) / 2, min(p[0], q[0]), max(p[0], q[0]), hs, tol_pt), p, q])
        elif abs(dx) <= 0.08 * abs(dy):
            runs.append(["v", _snap((p[0] + q[0]) / 2, min(p[1], q[1]), max(p[1], q[1]), vs, tol_pt), p, q])
        else:
            runs.append(["g", None, p, q])
    merged = []
    for r_ in runs:
        if merged and r_[0] in "hv" and merged[-1][0] == r_[0] and abs(merged[-1][1] - r_[1]) < 0.3:
            merged[-1][3] = r_[3]
        else:
            merged.append(r_)

    def coef(e):
        if e[0] == "h":
            return 0.0, 1.0, e[1]
        if e[0] == "v":
            return 1.0, 0.0, e[1]
        (xa, ya), (xb, yb) = e[2], e[3]
        A, B = yb - ya, xa - xb
        return A, B, A * xa + B * ya
    out = [list(pts[0])]
    for p, q in zip(merged, merged[1:]):
        a1, b1, c1 = coef(p)
        a2, b2, c2 = coef(q)
        det = a1 * b2 - a2 * b1
        if abs(det) > 1e-6 * (abs(a1) + abs(b1)) * (abs(a2) + abs(b2)):
            out.append([round((c1 * b2 - c2 * b1) / det, 2), round((a1 * c2 - a2 * c1) / det, 2)])
        else:
            out.append([round(q[2][0], 2), round(q[2][1], 2)])
    out.append(list(pts[-1]))
    # a corner that moved far from the traced path means a snap went wrong: keep the trace
    if any(math.dist(o, min(pts, key=lambda t: math.dist(t, o))) > 4 * tol_pt for o in out[1:-1]):
        return [list(p) for p in pts]
    return out
