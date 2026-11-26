"""Render a one-page takeoff sheet: BOM header, outlined plan, title block."""
import datetime
import math

import pymupdf

from . import template
from .detect import bbox, label_point, point_in_polygon

PAGE_W, PAGE_H = 792, 612          # 11 x 8.5 in landscape
INK = (0, 0, 0)
FILL = (0.80, 0.90, 0.97)           # light blue room fill
FILLS = [FILL, (0.99, 0.86, 0.68), (0.80, 0.93, 0.78), (0.92, 0.83, 0.96), (0.97, 0.80, 0.80)]   # one per buildup
SERIF = "tiro"                      # Times-Roman (built in)
SANS = "helv"


def _esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _text_width(text, size, font=SERIF):
    return pymupdf.get_text_length(text, fontname=font, fontsize=size)


def _hits(box, polys):
    """Does any polygon (page coordinates) touch the rect? Vertices inside it, or edges through it."""
    def seg_hits(a, b):
        for k in range(21):
            t = k / 20
            if box.contains(pymupdf.Point(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)):
                return True
        return False
    for poly in polys:
        if any(seg_hits(poly[i - 1], poly[i]) for i in range(len(poly))):
            return True
        if all(point_in_polygon(c[0], c[1], poly) for c in ((box.x0, box.y0), (box.x1, box.y1))):
            return True
    return False


def _fit_rect(clip, area):
    """Largest rect with clip's aspect ratio centred inside area. Returns (rect, scale)."""
    s = min(area.width / clip.width, area.height / clip.height)
    w, h = clip.width * s, clip.height * s
    x0 = area.x0 + (area.width - w) / 2
    y0 = area.y0 + (area.height - h) / 2
    return pymupdf.Rect(x0, y0, x0 + w, y0 + h), s


def _cuts(g, v, axis):
    """Where the line y=v (axis 0) or x=v (axis 1) crosses a room's outline and columns."""
    cuts = []
    for poly in g:
        for i in range(len(poly)):
            a, b = poly[i - 1], poly[i]
            pa, pb = (a[1], b[1]) if axis == 0 else (a[0], b[0])
            if (pa > v) != (pb > v):
                t = (v - pa) / (pb - pa)
                cuts.append(a[0] + t * (b[0] - a[0]) if axis == 0 else a[1] + t * (b[1] - a[1]))
    return sorted(cuts)


def _brick_segments(groups, course, length, origin):
    """Running-bond (BRICK) hatch: courses `course` apart, joints every `length`, offset by
    half a length on alternate courses. Clipped to each room minus its columns."""
    segs = []
    xs = [p[0] for g in groups for poly in g for p in poly]
    ys = [p[1] for g in groups for poly in g for p in poly]
    k0 = math.ceil((min(ys) - origin[1]) / course)
    y = origin[1] + k0 * course
    row = k0
    while y < max(ys):
        for g in groups:
            c = _cuts(g, y, 0)
            for j in range(0, len(c) - 1, 2):
                segs.append(((c[j], y), (c[j + 1], y)))
        y += course
        row += 1
    # joints: vertical pieces within each course
    y = origin[1] + (k0 - 1) * course
    row = k0 - 1
    while y < max(ys):
        off = (length / 2) if row % 2 else 0.0
        m0 = math.floor((min(xs) - origin[0] - off) / length)
        x = origin[0] + off + m0 * length
        while x < max(xs):
            for g in groups:
                c = _cuts(g, x, 1)
                for j in range(0, len(c) - 1, 2):
                    lo, hi = max(c[j], y), min(c[j + 1], y + course)
                    if hi > lo:
                        segs.append(((x, lo), (x, hi)))
            x += length
        y += course
        row += 1
    return segs


def _diagonal_segments(groups, spacing, origin, angle=45):
    """Parallel-line hatch (like AutoCAD ANSI31) at `angle` degrees, clipped to each room
    minus its columns: rotate the rooms so the lines are horizontal, cut, rotate back."""
    a = math.radians(angle)
    ca, sa = math.cos(a), math.sin(a)

    def rot(p, c, s_):
        x, y = p[0] - origin[0], p[1] - origin[1]
        return (x * c - y * s_, x * s_ + y * c)

    turned = [[[rot(p, ca, sa) for p in poly] for poly in g] for g in groups]
    ys = [p[1] for g in turned for poly in g for p in poly]
    segs = []
    y = math.ceil(min(ys) / spacing) * spacing
    while y < max(ys):
        for g in turned:
            c = _cuts(g, y, 0)
            for j in range(0, len(c) - 1, 2):
                p0, p1 = rot((c[j] + origin[0], y + origin[1]), ca, -sa), rot((c[j + 1] + origin[0], y + origin[1]), ca, -sa)
                segs.append(((p0[0] + origin[0], p0[1] + origin[1]), (p1[0] + origin[0], p1[1] + origin[1])))
        y += spacing
    return segs


# AutoCAD pattern definitions (acad.pat, inches): (angle, base, offset, dashes) per line family
PATTERNS = {
    "DOLMIT": [(0.0, (0.0, 0.0), (0.0, 0.25), []),
               (45.0, (0.0, 0.0), (-0.5, 0.5), [0.35355339, -0.70710678])],
}


def _pattern_segments(groups, families, unit, angle, origin):
    """An AutoCAD hatch pattern (see PATTERNS) clipped to each room minus its columns.
    `unit` = plan points per pattern unit (scale included); `angle` is the hatch angle in
    degrees as AutoCAD measures it (y up). Plan points run y down, so work flipped."""
    flip = [[[(p[0] - origin[0], origin[1] - p[1]) for p in poly] for poly in g] for g in groups]
    segs = []
    ra = math.radians(angle)
    cr, sr = math.cos(ra), math.sin(ra)
    for ang, base, off, dashes in families:
        a = math.radians(ang + angle)
        ca, sa = math.cos(a), math.sin(a)
        b = ((base[0] * cr - base[1] * sr) * unit, (base[0] * sr + base[1] * cr) * unit)
        o = ((off[0] * cr - off[1] * sr) * unit, (off[0] * sr + off[1] * cr) * unit)
        # into the line frame: u along the lines, v across them
        def fr(p):
            return (p[0] * ca + p[1] * sa, -p[0] * sa + p[1] * ca)
        turned = [[[fr(p) for p in poly] for poly in g] for g in flip]
        bu, bv = fr(b)
        ou, ov = fr(o)
        if abs(ov) < 1e-9:
            continue
        vs = [p[1] for g in turned for poly in g for p in poly]
        k0, k1 = math.ceil((min(vs) - bv) / ov), math.floor((max(vs) - bv) / ov)
        if k0 > k1:
            k0, k1 = k1, k0
        dashes = [d * unit for d in dashes]
        period = sum(abs(d) for d in dashes)
        for k in range(k0, k1 + 1):
            v = bv + k * ov
            phase = bu + k * ou
            for g in turned:
                c = _cuts(g, v, 0)
                for j in range(0, len(c) - 1, 2):
                    lo, hi = c[j], c[j + 1]
                    pieces = []
                    if not dashes:
                        pieces = [(lo, hi)]
                    else:
                        n = math.floor((lo - phase) / period)
                        start = phase + n * period
                        while start < hi:
                            at = start
                            for d in dashes:
                                if d > 0:
                                    p0, p1 = max(at, lo), min(at + d, hi)
                                    if p1 > p0:
                                        pieces.append((p0, p1))
                                at += abs(d)
                            start += period
                    for p0, p1 in pieces:
                        pts = []
                        for u in (p0, p1):
                            x, y = u * ca - v * sa, u * sa + v * ca
                            pts.append((x + origin[0], origin[1] - y))
                        segs.append(tuple(pts))
    return segs


TILE_GRID_FAMILIES = ("fit_tile", "xfit")
NOT_PLITEQ = ("finished_floor", "plywood", "pmi", "concrete", "custom")
DIAGONAL_FT = 4.0        # ANSI31 at scale 384 = lines 48" apart
ROW_FAMILIES = ("rst", "ff", "ff_panel")
ROW_FT = 4.0             # horizontal rows for RST / FF


def hatch_layer(catalog, codes):
    """The layer a buildup is hatched with: the TOP-MOST Pliteq material (finished floor,
    plywood and perimeter isolation are skipped)."""
    for c in codes:
        p = catalog["products"][c]
        if p.get("components"):                      # a system: hatch from its top component
            p = catalog["products"][p["components"][0]]
        if p["family"] not in NOT_PLITEQ and p.get("in_bom", True):
            return p
    return None


def hatch_kind(p):
    """FIT / xFIT / VersaDUEL tiles get the 2' tile grid (ANSI34), planks get BRICK,
    RST and FF (rolls, sheets and panels) get horizontal rows, everything else
    (VersaTREAD, VersaDUEL rolls, ...) gets diagonal lines (ANSI31)."""
    if p is None:
        return None
    if p["family"] == "fit_roll":                 # FIT08: rolls, so horizontal rows like RST
        return "rows"
    if (p.get("hatch_pattern") or "").upper() == "BRICK":
        return "brick"
    if (p.get("hatch_pattern") or "").upper() in PATTERNS:   # Envy / PeakForce: DOLMIT
        return "pattern"
    if p["family"] in TILE_GRID_FAMILIES or (p["code"].startswith("VersaDUEL") and p["unit"] == "tile"):
        return "grid"
    if p["family"] in ROW_FAMILIES:
        return "rows"
    return "diagonal"


def _grid_segments(groups, spacing, origin):
    """Tile grid lines clipped to each room: its outline minus its columns (even-odd)."""
    segs = []
    xs = [p[0] for g in groups for poly in g for p in poly]
    ys = [p[1] for g in groups for poly in g for p in poly]
    for axis in (0, 1):
        lo, hi = (min(ys), max(ys)) if axis == 0 else (min(xs), max(xs))
        o = origin[1] if axis == 0 else origin[0]
        k = math.ceil((lo - o) / spacing)
        v = o + k * spacing
        while v < hi:
            for g in groups:
                cuts = []
                for poly in g:
                    for i in range(len(poly)):
                        a, b = poly[i - 1], poly[i]
                        pa, pb = (a[1], b[1]) if axis == 0 else (a[0], b[0])
                        if (pa > v) != (pb > v):
                            t = (v - pa) / (pb - pa)
                            cuts.append(a[0] + t * (b[0] - a[0]) if axis == 0 else a[1] + t * (b[1] - a[1]))
                cuts.sort()
                for j in range(0, len(cuts) - 1, 2):
                    if axis == 0:
                        segs.append(((cuts[j], v), (cuts[j + 1], v)))
                    else:
                        segs.append(((v, cuts[j]), (v, cuts[j + 1])))
            v += spacing
    return segs


def ramp_shapes(r, ppf, room=None):
    """Polygons (PDF points) for one opening's ramps: [(kind, polygon)], kind "ada", "std",
    "ada-wb", "std-wb" (tiers on wood blocking) or "landing".
    Ramps sit on the buildup, starting at the opening (or at the far edge of a FIT08
    landing) and running into the room. A stacked ramp's tiers run one after another, the
    first (on the floor, cut shorter if it's a part tier) at the opening (Visa FCM-1). ADA
    ramps are cut to the width of the opening (or one ramp, centred or flush to a wall, with
    standard wings filling the rest); each exposed side gets a triangular wing per tier,
    widest where the ramps start and as long as the ADA ramps up to that tier, nested
    (THL-1, FCM-1), and each side of a landing gets a standard ramp per tier along it (PTC-1)."""
    (ax, ay), (bx, by) = r["a"], r["b"]
    L = math.hypot(bx - ax, by - ay)
    if L <= 0:
        return []
    ux, uy = (bx - ax) / L, (by - ay) / L
    nx, ny = -r["normal"][0], -r["normal"][1]       # "normal" points off the buildup: go the other way
    ft = ppf / 12                                 # points per inch
    d_std, d_ada = r["depth_in"]["standard"] * ft, r["depth_in"]["ada"] * ft
    D = (r.get("landing_in") or 0) * ft           # FIT08 landing depth
    sides = r.get("sides", 2)
    scale = r.get("tier_scale") or [1.0] * max(1, r.get("tiers", 1))
    kinds = ["" if k == 0 else "-wb" for k in range(len(scale))]
    W, A = [0.0], [0.0]                           # running standard / ADA depths, tier by tier
    for f in scale:
        W.append(W[-1] + d_std * f)
        A.append(A[-1] + d_ada * f)
    T = len(scale)

    def P(t, d):
        return [ax + ux * t + nx * d, ay + uy * t + ny * d]

    def rect(t0, t1, d0, d1):
        return [P(t0, d0), P(t1, d0), P(t1, d1), P(t0, d1)]
    out = []
    if D:
        out.append(("landing", rect(0, L, 0, D)))
        for k in range(T):                        # first tier against the landing, the rest outside it
            if sides >= 1:
                out.append(("std" + kinds[k], rect(L + W[k], L + W[k + 1], 0, D)))
            if sides == 2:
                out.append(("std" + kinds[k], rect(-W[k + 1], -W[k], 0, D)))
    if not r["ada"]:
        # a standard ramp's ends are cut back 45 degrees: full length on the room side, the
        # edge at the opening shorter by one ramp depth at each end, with a triangular side
        # piece filling each corner so the whole opening is covered (BFTL-1)
        # An end that runs into the next ramp is cut to meet it instead (cut_a / cut_b, from
        # ramps._joins): mitred at a corner, along the wing beside an ADA ramp.
        def on_cut(cut, d):
            from .ramps import _cross_lines
            (p0, p1) = cut
            return list(_cross_lines(P(0, d), (ux, uy), p0, (p1[0] - p0[0], p1[1] - p0[1])) or P(0, d))
        ca, cb = r.get("cut_a"), r.get("cut_b")
        for k in range(T):
            d0, d1 = D + W[k], D + W[k + 1]
            c = min(d1 - d0, L / 2 - 0.01)
            kind = "std" + kinds[k]
            a0, a1 = (on_cut(ca, d0), on_cut(ca, d1)) if ca else (P(c, d0), P(0, d1))
            b0, b1 = (on_cut(cb, d0), on_cut(cb, d1)) if cb else (P(L - c, d0), P(L, d1))
            out.append((kind, [a0, b0, b1, a1]))
            if not ca:
                out.append((kind, [P(0, d0), P(c, d0), P(0, d1)]))
            if not cb:
                out.append((kind, [P(L, d0), P(L, d1), P(L - c, d0)]))
        return out
    # ADA ramps are cut to the opening (Parkview PTC-1: a 48" door is ordered as 2 x 42" but
    # drawn 48" wide); "one ramp + wings" keeps one 42" ramp, cut down if the opening is narrower
    w = L if r.get("ada_width", "full") == "full" else min(42 * ft, L)
    t0 = (L - w) / 2 if sides == 2 else 0.0
    t1 = t0 + w
    left, right = t0 > 0.5, L - t1 > 0.5
    for k in range(T):
        out.append(("ada" + kinds[k], rect(t0, t1, D + A[k], D + A[k + 1])))
        if left:
            out.append(("std" + kinds[k], rect(0, t0, D + W[k], D + W[k + 1])))
        if right:
            out.append(("std" + kinds[k], rect(t1, L, D + W[k], D + W[k + 1])))
    if not D and not left and not right:
        # as drafted: one triangular wing a side, back to the ADA ramp's far corner. Where a
        # standard run carries on from that side, its tip sits on the run's inner edge, one ramp
        # depth out (Lifetime Sky); against a bare wall it lies along the wall, a right triangle
        # one standard depth long (Cottages Cardio)
        # A wing with the doorway's wall beside it fills to the wall: its short edge runs along
        # the doorway line to where the wall turns away (the jamb's corner), its long edge from
        # there back to the ADA ramp's far corner (drafted: an ADA ramp beside a slanted wall)
        def to_wall(t, sgn):
            if not room:
                return None
            tol = 0.75
            def off(p):                           # distance off the doorway line, and along it
                return (abs((p[0] - ax) * nx + (p[1] - ay) * ny), (p[0] - ax) * ux + (p[1] - ay) * uy)
            reach, grew = 0.0, True
            while grew:
                grew = False
                for i in range(len(room)):
                    (d0, s0), (d1, s1) = off(room[i - 1]), off(room[i])
                    if d0 > tol or d1 > tol:
                        continue
                    lo, hi = sorted(((s0 - t) * sgn, (s1 - t) * sgn))
                    if lo <= reach + tol and hi > reach + tol:
                        reach, grew = hi, True
            return reach if reach > 0.5 else None
        if r.get("wing_r", sides >= 1):
            d = W[T] if r.get("std_at_b") else 0.0
            s = W[T] if d else min(to_wall(t1, 1) or W[T], A[T])
            out.append(("std", [P(t1, 0), P(t1 + s, d), P(t1, A[T])]))
        if r.get("wing_l", sides == 2):
            d = W[T] if r.get("std_at_a") else 0.0
            s = W[T] if d else min(to_wall(t0, -1) or W[T], A[T])
            out.append(("std", [P(t0, 0), P(t0 - s, d), P(t0, A[T])]))
        return out
    for k in range(T):
        kind = "std" + kinds[k]
        if r.get("wing_r", sides >= 1):
            d0 = D + (W[T] if right else 0)
            out.append((kind, [P(t1 + W[k], d0), P(t1 + W[k + 1], d0), P(t1, D + A[k + 1]), P(t1, max(d0, D + A[k]))]))
        if r.get("wing_l", sides == 2):
            d0 = D + (W[T] if left else 0)
            out.append((kind, [P(t0 - W[k], d0), P(t0 - W[k + 1], d0), P(t0, D + A[k + 1]), P(t0, max(d0, D + A[k]))]))
    return out


def clip_convex(subject, clip):
    """Sutherland-Hodgman: the part of polygon `subject` (any shape) inside the convex
    polygon `clip`. Used to trim a ramp to its room: subject = room outline, clip = ramp."""
    def area2(poly):
        return sum(poly[i - 1][0] * poly[i][1] - poly[i][0] * poly[i - 1][1] for i in range(len(poly)))
    sign = 1 if area2(clip) > 0 else -1

    def inside(p, a, b):
        return sign * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= -1e-9

    def cross(p, q, a, b):
        x1, y1, x2, y2 = p[0], p[1], q[0], q[1]
        x3, y3, x4, y4 = a[0], a[1], b[0], b[1]
        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(d) < 1e-12:
            return list(q)
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
        return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)]
    out = [list(p) for p in subject]
    for i in range(len(clip)):
        a, b = clip[i - 1], clip[i]
        inp, out = out, []
        if not inp:
            break
        for j in range(len(inp)):
            p, q = inp[j - 1], inp[j]
            if inside(q, a, b):
                if not inside(p, a, b):
                    out.append(cross(p, q, a, b))
                out.append(q)
            elif inside(p, a, b):
                out.append(cross(p, q, a, b))
    return out


def _clusters(rooms, touch):
    """Rooms that touch or nearly touch (split pieces, neighbours) stay together."""
    n = len(rooms)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    boxes = [bbox(r["outline_pt"]) for r in rooms]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            if a[0] - touch <= b[2] and b[0] - touch <= a[2] and a[1] - touch <= b[3] and b[1] - touch <= a[3]:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _pack_rooms(rooms, meta, areas):
    """Move groups of rooms that are far apart on the plan next to each other.

    Returns shifted copies of the rooms (and access points) in a new coordinate space;
    scale and orientation are unchanged, so tile grids and dimensions still match."""
    ppf = meta.get("pt_per_ft") or 9.0
    groups = _clusters(rooms, touch=4 * ppf)
    if len(groups) < 2:
        return rooms, meta
    boxes = []
    for g in groups:
        pts = [p for i in g for p in rooms[i]["outline_pt"]]
        boxes.append(bbox(pts))
    order = sorted(range(len(groups)), key=lambda k: ((boxes[k][1] + boxes[k][3]) / 2, boxes[k][0]))
    big = max(max(b[2] - b[0], b[3] - b[1]) for b in boxes)
    gap = max(6 * ppf, 0.06 * big)

    def layout(nrows):
        # split the groups, in reading order, into rows of similar total width
        widths = [boxes[k][2] - boxes[k][0] for k in order]
        target_w = (sum(widths) + gap * (len(order) - 1)) / nrows
        rows, cur, w = [], [], 0.0
        for k, wk in zip(order, widths):
            if cur and w + gap + wk > target_w * 1.15 and len(rows) < nrows - 1:
                rows.append(cur)
                cur, w = [], 0.0
            cur.append(k)
            w += (gap if len(cur) > 1 else 0) + wk
        rows.append(cur)
        pos, y, W = {}, 0.0, 0.0
        for row in rows:
            h = max(boxes[k][3] - boxes[k][1] for k in row)
            x = 0.0
            for k in row:
                bw, bh = boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]
                pos[k] = (x - boxes[k][0], y + (h - bh) / 2 - boxes[k][1])     # shift for this group
                x += bw + gap
            W = max(W, x - gap)
            y += h + gap
        return pos, W, y - gap

    best = None
    for nrows in range(1, len(groups) + 1):
        pos, W, H = layout(nrows)
        for a in areas or [pymupdf.Rect(0, 0, 700, 400)]:
            sc = min(a.width / W, a.height / H)
            if best is None or sc > best[0] * 1.02:
                best = (sc, pos)
    pos = best[1]
    shift = {}
    for k, g in enumerate(groups):
        for i in g:
            shift[i] = pos[k]
    out = []
    for i, r in enumerate(rooms):
        dx, dy = shift[i]
        q = dict(r)
        q["outline_pt"] = [[x + dx, y + dy] for x, y in r["outline_pt"]]
        q["holes_pt"] = [[[x + dx, y + dy] for x, y in h] for h in r.get("holes_pt") or []]
        out.append(q)
    meta = dict(meta)
    acc = []
    for a in meta.get("access") or []:
        mx, my = (a["a"][0] + a["b"][0]) / 2, (a["a"][1] + a["b"][1]) / 2
        i = min(range(len(rooms)), key=lambda i: _dist_box((mx, my), bbox(rooms[i]["outline_pt"])))
        dx, dy = shift[i]
        acc.append(dict(a, a=[a["a"][0] + dx, a["a"][1] + dy], b=[a["b"][0] + dx, a["b"][1] + dy]))
    meta["access"] = acc
    if meta.get("ramps"):
        moved = []
        for x in meta["ramps"].get("ramps") or []:
            if not x.get("a"):
                moved.append(x)
                continue
            mx, my = (x["a"][0] + x["b"][0]) / 2, (x["a"][1] + x["b"][1]) / 2
            i = x.get("room_index")
            if i is None or not 0 <= i < len(rooms):
                i = min(range(len(rooms)), key=lambda i: _dist_box((mx, my), bbox(rooms[i]["outline_pt"])))
            dx, dy = shift[i]
            cut = {c: [[q[0] + dx, q[1] + dy] for q in x[c]] for c in ("cut_a", "cut_b") if x.get(c)}
            moved.append(dict(x, a=[x["a"][0] + dx, x["a"][1] + dy], b=[x["b"][0] + dx, x["b"][1] + dy], **cut))
        meta["ramps"] = dict(meta["ramps"], ramps=moved)
    return out, meta


def _dist_box(p, b):
    dx = max(b[0] - p[0], 0, p[0] - b[2])
    dy = max(b[1] - p[1], 0, p[1] - b[3])
    return math.hypot(dx, dy)


def _free_layout(polys, shapes, rooms, groups, ramp_labels, fonts, cs, cpitch, frame, notes_top,
                 bom_box, turn, col_scale, ppf):
    """The plan drawn as large as the sheet allows, turned whichever way suits it, with each
    callout in the open sheet around the drawing (never on it), near what it points at (as the drafters lay the sheets out).
    Returns (quarter turns, target rect, scale, clip, pad, [(tip, lines, spot, which)]) when
    that draws the plan bigger than with a callout column, else None."""
    import numpy as np
    full = pymupdf.Rect(frame.x0 + 12, frame.y0 + 10, frame.x1 - 12, notes_top - 6)
    # what each callout points at, in plan coordinates
    wants = []
    for x, kind, poly in shapes:
        key = "landing" if kind == "landing" else x.get("label_" + kind.replace("-", "_"), x["label_std"])
        for lab in ramp_labels:
            if lab["key"] == key:
                c = (sum(q[0] for q in poly) / len(poly), sum(q[1] for q in poly) / len(poly))
                hit = next((w for w in wants if w[0] == key), None)
                if hit:
                    hit[2].append(c)
                else:
                    wants.append([key, lab["text"].split("\n"), [c], None])
    for g in sorted(groups, key=lambda g: -sum(r.get("area_sf") or 0 for r in rooms if r.get("_key") == g["key"])):
        rs = [r for r in rooms if r.get("_key") == g["key"]]
        if rs:
            big = max(rs, key=lambda r: r.get("area_sf") or 0)
            wants.append([g["key"], g["callout"].split("\n"), [label_point(big["outline_pt"])], big])
    sizes = [(max(fonts.width(l, cs) for l in lines), len(lines) * cpitch) for _k, lines, _t, _r in wants]
    W, H = int(math.ceil(frame.x1)) + 2, int(math.ceil(frame.y1)) + 2

    def crosses(p1, p2, q1, q2):
        def o(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return o(p1, p2, q1) * o(p1, p2, q2) < 0 and o(q1, q2, p1) * o(q1, q2, p2) < 0

    def place(k, target, s, clip):
        def tx(p):
            q = turn(p, k)
            return (target.x0 + (q[0] - clip.x0) * s, target.y0 + (q[1] - clip.y0) * s)
        # where callouts can't go: rooms, ramps, the BOM, outside the frame, the notes
        doc = pymupdf.open()
        pg = doc.new_page(width=W, height=H)
        for poly in polys + [sp[2] for sp in shapes]:
            pg.draw_polyline([tx(q) for q in poly], color=(0, 0, 0), fill=(0, 0, 0), width=1.5, closePath=True)
        busy = np.frombuffer(pg.get_pixmap(colorspace=pymupdf.csGRAY, alpha=False).samples, np.uint8)
        busy = (busy.reshape(-1, W)[:H] < 128).astype(np.int32) if busy.size >= W * H else None
        if busy is None:
            return None
        busy[:, :int(frame.x0 + 8)] = 1
        busy[:, int(frame.x1 - 8):] = 1
        busy[:int(frame.y0 + 8), :] = 1
        busy[int(notes_top - 2):, :] = 1
        busy[:int(bom_box.y1 + 2), :int(bom_box.x1 + 2)] = 1
        # callouts go around the drawing, never on it (the leaders cross into it)
        busy[max(0, int(target.y0)):int(target.y1 + 1), max(0, int(target.x0)):int(target.x1 + 1)] = 1
        texts, legs, out = [], [], []
        for (key, lines, tips, room), (w, h) in zip(wants, sizes):
            ii = np.zeros((H + 1, W + 1), np.int64)
            ii[1:, 1:] = busy.cumsum(0).cumsum(1)
            M = 7                                        # clear of the outlines all round
            pw, ph = int(w + 2 * M), int(h + 2 * M)
            if pw >= W or ph >= H:
                return None
            sums = ii[ph:, pw:] - ii[:-ph, pw:] - ii[ph:, :-pw] + ii[:-ph, :-pw]
            ys, xs = np.nonzero(sums[::2, ::2] == 0)
            if not len(xs):
                return None
            X, Y = xs * 2 + float(M), ys * 2 + float(M)  # the text's top-left
            ptips = [tx(t) for t in tips]
            others = [[tx(q) for q in r["outline_pt"]] for r in rooms
                      if room is None or r.get("_key") != room.get("_key")] if room is not None else []
            if room is not None:                         # or anywhere well inside the room (short leaders)
                poly = [tx(q) for q in room["outline_pt"]]
                holes = [[tx(q) for q in h_] for h_ in room.get("holes_pt") or []]
                bx = bbox(poly)
                step = max(12.0, max(bx[2] - bx[0], bx[3] - bx[1]) / 14)

                def clear(px, py):
                    best_ = 1e9
                    for ring in [poly] + holes:
                        for i in range(len(ring)):
                            a_, b_ = ring[i - 1], ring[i]
                            dx, dy = b_[0] - a_[0], b_[1] - a_[1]
                            t = max(0.0, min(1.0, ((px - a_[0]) * dx + (py - a_[1]) * dy) / ((dx * dx + dy * dy) or 1e-9)))
                            best_ = min(best_, math.hypot(a_[0] + dx * t - px, a_[1] + dy * t - py))
                    return best_
                yy = bx[1] + step / 2
                while yy < bx[3]:
                    xx = bx[0] + step / 2
                    while xx < bx[2]:
                        if point_in_polygon(xx, yy, poly) and not any(point_in_polygon(xx, yy, h_) for h_ in holes) \
                                and clear(xx, yy) >= 14:
                            ptips.append((xx, yy))
                        xx += step
                    yy += step
            best = None
            for tp in ptips:
                for which in ("left", "right"):          # text left (leader from its right end) or right of the tip
                    sx = X + w + 4 if which == "left" else X - 4
                    ex = sx + 22 if which == "left" else sx - 22
                    sy = Y + cs * 0.5
                    ok = (ex <= tp[0] + 4) if which == "left" else (ex >= tp[0] - 4)
                    cost = 22 + np.hypot(tp[0] - ex, tp[1] - sy) + np.where(ok, 0, 1e9)
                    order = np.argsort(cost)[:300]
                    for i in order:
                        if cost[i] >= 1e8 or (best and cost[i] >= best[0]):
                            break
                        leg = ((sx[i], sy[i]), (ex[i], sy[i]), tp)
                        # an arrow across another room reads as pointing at it: avoid when there's room
                        pen = 0.0
                        if others:
                            pen = 120.0 * sum(crosses(a, b, c, d) for a, b in ((leg[0], leg[1]), (leg[1], leg[2]))
                                              for ring in others for c, d in zip(ring, ring[1:] + ring[:1])) / 2
                            if best and cost[i] + pen >= best[0]:
                                continue
                        # leaders don't run through other callouts or cross other leaders
                        if any(r.intersects(pymupdf.Rect(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]) + .1, max(a[1], b[1]) + .1))
                               and any(r.contains(pymupdf.Point(a[0] + (b[0] - a[0]) * t / 20, a[1] + (b[1] - a[1]) * t / 20))
                                       for t in range(21))
                               for r in texts for a, b in ((leg[0], leg[1]), (leg[1], leg[2]))):
                            continue
                        if any(crosses(a, b, c, d) for L in legs for a, b in ((L[0], L[1]), (L[1], L[2]))
                               for c, d in ((leg[0], leg[1]), (leg[1], leg[2]))):
                            continue
                        best = (cost[i] + pen, pymupdf.Rect(X[i], Y[i], X[i] + w, Y[i] + h), which, leg, tp)
                        if pen == 0:
                            break
            if not best or best[0] > 0.5 * (frame.x1 - frame.x0) + 120:
                return None
            _c, spot, which, leg, tp = best
            texts.append(spot + (-3, -3, 3, 3))
            legs.append(leg)
            busy[max(0, int(spot.y0 - 4)):int(spot.y1 + 5), max(0, int(spot.x0 - 6)):int(spot.x1 + 7)] = 1
            out.append((pymupdf.Point(*tp), lines, spot, which))
        return out

    found = None
    for k in range(4):
        pts = [turn(q, k) for poly in polys + [sp[2] for sp in shapes] for q in poly]
        x0, y0, x1, y1 = bbox(pts)
        pad = max(x1 - x0, y1 - y0) * 0.03 + 4
        clip = pymupdf.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        s_full = min(full.width / clip.width, full.height / clip.height)
        floor_ = max(col_scale * 1.02, found[2] * 1.02 if found else 0)
        f = 1.0
        while s_full * f >= floor_:
            s = s_full * f
            w, h = clip.width * s, clip.height * s
            done = None
            for ax in (1.0, 0.5, 0.0):
                for ay in (1.0, 0.5):
                    tl = (full.x0 + (full.width - w) * ax, full.y0 + (full.height - h) * ay)
                    target = pymupdf.Rect(tl[0], tl[1], tl[0] + w, tl[1] + h)
                    if target.intersects(bom_box) and _hits(bom_box, [[(target.x0 + (turn(q, k)[0] - clip.x0) * s,
                                                                         target.y0 + (turn(q, k)[1] - clip.y0) * s)
                                                                        for q in poly] for poly in polys]):
                        continue
                    got = place(k, target, s, clip)
                    if got:
                        done = (k, target, s, clip, pad, got)
                        break
                if done:
                    break
            if done:
                found = done
                break
            f -= 0.03
    return found


def render_sheet(out_path, plan_doc, page_no, rooms, takeoff, company, meta, catalog):
    """Write the takeoff sheet PDF.

    rooms:   included rooms (dicts with outline_pt), in plan PDF points
    takeoff: rules.Takeoff
    meta:    {project, dwg_no, drawn_by, date, pt_per_ft}
    """
    # border, title block, logo and fixed text come from Pliteq's AutoCAD template
    # (config/sheet_template.json, made from the DXF by takeoff.template)
    tpl = template.load()
    doc = pymupdf.open()
    page = doc.new_page(width=tpl["page"][0], height=tpl["page"][1])
    fonts = template.Fonts(page)
    font = fonts.name
    frame = pymupdf.Rect(*tpl["frame"])

    # ---- BOM header, where the template has it
    bf = tpl["fields"]["bom"]
    bom = [ln.text() for ln in takeoff.lines]
    size = bf["size"]
    lh = size * template.CAP * template.LINE_PITCH
    y = bf["y"] + size * template.CAP
    bom_at = []                                   # drawn after the plan (which may reach up beside it)
    for line in bom:
        bom_at.append(((bf["x"], y), line))
        y += lh
    bom_bottom = y - lh + 6
    bom_right = bf["x"] + max((fonts.width(t, size) for t in bom), default=0)

    # ---- plan area: the selected rooms over the plan drawing (faded 70%, see below). Rooms far
    # apart on the plan are moved next to each other, all at the same scale and orientation.
    warranty = next((t for t in tpl["texts"] if t["text"].startswith("Scaled according")), None)
    notes_top = (warranty["y"] if warranty else frame.y1 - 140) - 4
    cs = tpl.get("callout_size") or 12.0            # callout text size
    cpitch = cs * 1.22
    ramp_labels = (meta.get("ramps") or {}).get("labels") or []
    col_w = max((fonts.width(l, cs) for t in [g["callout"] for g in takeoff.groups] + [x["text"] for x in ramp_labels]
                 for l in t.split("\n")), default=0) + 46
    below = pymupdf.Rect(frame.x0 + 12, bom_bottom + 10, frame.x1 - 12 - col_w, notes_top - 6)
    beside = pymupdf.Rect(bom_right + 20, frame.y0 + 10, frame.x1 - 12 - col_w, notes_top - 6)
    # the full height, as on the drafted sheets: the plan reaches up beside the BOM wherever its
    # outline leaves that corner empty (Maroon Creek MCC-1); only kept when no room or ramp
    # would run into the BOM (checked in layout below)
    tall = pymupdf.Rect(frame.x0 + 12, frame.y0 + 10, frame.x1 - 12 - col_w, notes_top - 6)
    bom_box = pymupdf.Rect(frame.x0, frame.y0, bom_right + 12, bom_bottom + 6)
    areas = [a for a in (below, beside, tall) if a.width > 50 and a.height > 50]
    plan_rooms = rooms                              # where the rooms are on the plan, before packing
    rooms, meta = _pack_rooms(rooms, meta, areas)
    polys = [r["outline_pt"] for r in rooms]
    ppf = meta.get("pt_per_ft") or 9.0
    shapes = []                                   # (opening, kind, polygon)
    for x in (meta.get("ramps") or {}).get("ramps") or []:
        if x.get("a"):
            room = polys[x["room_index"]] if 0 <= x.get("room_index", -1) < len(polys) else None
            shapes += [(x, k, poly) for k, poly in ramp_shapes(x, ppf, room)]
    # The plan is drawn as is, or turned a quarter turn (clockwise) when that draws it larger:
    # a tall room on this landscape sheet fills the space, and the leaders get shorter.
    def turn(p, k):                               # k quarter turns clockwise
        x_, y_ = p
        for _ in range(int(k) % 4):
            x_, y_ = -y_, x_
        return (x_, y_)

    def unturn(p, k):
        x_, y_ = p
        for _ in range(int(k) % 4):
            x_, y_ = y_, -x_
        return (x_, y_)
    turned_polys = []

    def layout(rot):
        nonlocal turned_polys
        turned_polys = [[turn(q, rot) for q in poly] for poly in polys + [sp[2] for sp in shapes]]
        pts = [q for poly in turned_polys for q in poly]
        x0, y0, x1, y1 = bbox(pts)
        pad = max(x1 - x0, y1 - y0) * 0.03 + 4
        clip = pymupdf.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        # each buildup's callout goes on the side its room is on; keep a column free on that side
        side = {}
        for g in takeoff.groups:
            rs = [r for r in rooms if r.get("_key") == g["key"]]
            if rs:
                big = max(rs, key=lambda r: r.get("area_sf") or 0)
                side[g["key"]] = "left" if turn(label_point(big["outline_pt"]), rot)[0] < (x0 + x1) / 2 else "right"
        if len(set(side.values())) == 1:
            side = {k: "left" for k in side}      # drafted sheets: plan on the right, callouts to its left
        # callouts on the side their room is on, all on the right, or all on the left: whichever
        # leaves the rooms drawn largest (the right-hand column is already kept free in `areas`)
        best, own = None, None
        for n, sides in enumerate((side, {k: "right" for k in side}, {k: "left" for k in side})):
            these = areas
            if "left" in sides.values():
                these = [pymupdf.Rect(a.x0 + col_w, a.y0, a.x1 + (col_w if "right" not in sides.values() else 0), a.y1)
                         for a in areas]
                these = [a for a in these if a.width > 50] or these
            for a in these:
                t_, s_ = _fit_rect(clip, a)
                if t_.intersects(bom_box) and _hits(bom_box, [[(t_.x0 + (q[0] - clip.x0) * s_, t_.y0 + (q[1] - clip.y0) * s_)
                                                               for q in poly] for poly in turned_polys]):
                    continue
                if best is None or s_ > best[1] * 1.02:
                    best = (t_, s_, sides, a)
                if n == 0 and (own is None or s_ > own[1]):
                    own = (t_, s_, sides, a)
        # each callout on its own room's side unless one side draws the rooms a third bigger
        # (callouts all on one side means leaders crossing the nearer rooms)
        if own and own[1] >= 0.75 * best[1]:
            best = own
        return best[:3] + (best[3], clip, pad)

    plain, turned = layout(0), layout(1)
    rot = 1 if turned[1] > plain[1] * 1.08 else 0   # turn when it draws the plan noticeably bigger (Maroon Creek: 13%)
    target, s, side, area, clip, pad = turned if rot else plain
    # the drawing goes to the right of the page (as on the drafted sheets); callouts sit just
    # beside it, so the leaders are short
    if "right" not in side.values() or "left" not in side.values():
        target = target + (area.x1 - target.x1, 0, area.x1 - target.x1, 0)

    # As the drafters do: no column kept for the callouts, they go in the open sheet around the
    # rooms (beside the BOM, in a notch of the plan), and the plan may be turned any way round.
    # Used whenever it draws the plan noticeably bigger and every callout finds a spot.
    free = _free_layout(polys, shapes, rooms, takeoff.groups, ramp_labels, fonts, cs, cpitch, frame,
                        notes_top, bom_box, turn, s, ppf)
    if free:
        rot, target, s, clip, pad, free = free

    x0, y0, x1, y1 = bbox([q for poly in polys for q in poly])     # plan coordinates (hatch origin)

    def tx(p):
        q = turn(p, rot)
        return pymupdf.Point(target.x0 + (q[0] - clip.x0) * s, target.y0 + (q[1] - clip.y0) * s)

    # ---- the plan drawing behind the rooms, faded 70% (the page's own drawing only: no PDF
    # markups/annotations and none of the app's labels). Rooms moved next to each other bring
    # their own piece of the plan with them.
    try:
        src_rect = plan_doc[page_no - 1].rect
        moves = {}
        for orig, moved in zip(plan_rooms, rooms):
            d = (round(moved["outline_pt"][0][0] - orig["outline_pt"][0][0], 2),
                 round(moved["outline_pt"][0][1] - orig["outline_pt"][0][1], 2))
            moves.setdefault(d, []).append(orig)
        for (dx, dy), rs in moves.items():
            if len(moves) == 1:                   # the whole drawn area, back in plan coordinates
                corners = [unturn(q, rot) for q in ((clip.x0, clip.y0), (clip.x1, clip.y1))]
                xs, ys = [q[0] - dx for q in corners], [q[1] - dy for q in corners]
                src = pymupdf.Rect(min(xs), min(ys), max(xs), max(ys))
            else:
                bx0, by0, bx1, by1 = bbox([q for r in rs for q in r["outline_pt"]])
                src = pymupdf.Rect(bx0 - pad, by0 - pad, bx1 + pad, by1 + pad)
            src &= src_rect
            if src.is_empty:
                continue
            cs_ = [tx((src.x0 + dx, src.y0 + dy)), tx((src.x1 + dx, src.y1 + dy))]
            dest = pymupdf.Rect(min(c.x for c in cs_), min(c.y for c in cs_), max(c.x for c in cs_), max(c.y for c in cs_))
            page.show_pdf_page(dest, plan_doc, page_no - 1, clip=src, rotate=(-90 * rot) % 360)
            page.draw_rect(dest, color=None, fill=(1, 1, 1), fill_opacity=0.7, width=0)
    except Exception:                             # a plan that can't be placed: rooms only, as before
        pass
    if target.intersects(bom_box):                 # the faded plan reaches up beside the BOM: keep it clear
        page.draw_rect(pymupdf.Rect(frame.x0 + 4, frame.y0 + 4, bom_right + 6, bom_bottom + 2), color=None, fill=(1, 1, 1), width=0)
    for at, line in bom_at:
        page.insert_text(at, line, fontname=font, fontsize=size)

    # one buildup per system: its own fill colour, tile grid and callout
    buildups = takeoff.groups
    sys_rooms = {g["key"]: [r for r in rooms if r.get("_key") == g["key"]] for g in buildups}
    ppf = meta.get("pt_per_ft")
    for k, g in enumerate(buildups):
        rs = sys_rooms[g["key"]]
        rings = [[r["outline_pt"]] + list(r.get("holes_pt") or []) for r in rs]
        sh = page.new_shape()
        # colour fills tell buildups apart; a single buildup is left unfilled, as on the drafted sheets
        for ring in rings if len(buildups) > 1 else []:
            for poly in ring:
                sh.draw_polyline([tx(p) for p in poly] + [tx(poly[0])])
            sh.finish(color=None, fill=FILLS[k % len(FILLS)], fill_opacity=0.5, closePath=True, width=0, even_odd=True)
        sh.commit()
        # hatch from the top-most PLITEQ layer (see hatch_kind); lines closer than ~1.2 pt
        # on paper would be a solid smear, so they're skipped
        top = hatch_layer(catalog, g["layers"])
        kind = hatch_kind(top)
        size = top.get("unit_size") if top else None
        segs = []
        if ppf and rings and kind:
            if kind == "brick" and size:
                course = math.sqrt(size / 4) * ppf          # planks about 1:4
                if course * s > 1.2:
                    segs = _brick_segments(rings, course, size * ppf * ppf / course, (x0, y0))
            elif kind == "grid":
                spacing = math.sqrt(size or 4) * ppf        # 2' x 2' tiles
                if spacing * s > 1.2:
                    segs = _grid_segments(rings, spacing, (x0, y0))
            elif kind == "pattern":
                # drafted sheets: DOLMIT at the product's scale, lines running up the sheet
                unit = (top.get("hatch_scale") or 1) * ppf / 12
                fams = PATTERNS[top["hatch_pattern"].upper()]
                if 0.25 * unit * s > 1.2:
                    segs = _pattern_segments(rings, fams, unit, 0 if rot % 2 else 90, (x0, y0))
            elif kind in ("diagonal", "rows"):
                spacing = (DIAGONAL_FT if kind == "diagonal" else ROW_FT) * ppf
                if spacing * s > 1.2:
                    segs = _diagonal_segments(rings, spacing, (x0, y0), 45 if kind == "diagonal" else 0)
        if segs:
            shape = page.new_shape()
            for a, b in segs:
                shape.draw_line(tx(a), tx(b))
            shape.finish(color=INK, width=0.35)
            shape.commit()

    for poly in polys:
        page.draw_polyline([tx(p) for p in poly] + [tx(poly[0])], color=INK, width=1.1, closePath=True)
    for r in rooms:
        for hole in r.get("holes_pt") or []:            # columns, left out of the takeoff
            page.draw_polyline([tx(p) for p in hole] + [tx(hole[0])], color=INK, width=0.6, closePath=True)

    # ---- transitions: ramps on the buildup at each opening, drawn over the tiles (they replace them there)
    tips = {}
    for x, kind, poly in shapes:
        pts = [tx(p) for p in poly]
        # ramps sit on the buildup: trim them to the room's outline (walls, poles at door jambs)
        ri = x.get("room_index")
        shape = poly
        if ri is not None and 0 <= ri < len(rooms):
            c = clip_convex(rooms[ri]["outline_pt"], poly)
            if len(c) >= 3 and abs(sum(c[i - 1][0] * c[i][1] - c[i][0] * c[i - 1][1] for i in range(len(c)))) > 1e-6:
                shape = c
        sp = []
        for p in shape:                           # drop repeated corners
            q = tx(p)
            if not sp or abs(q - sp[-1]) > 0.01:
                sp.append(q)
        if len(sp) > 1 and abs(sp[0] - sp[-1]) <= 0.01:
            sp.pop()
        # round joins: a thin wing's sharp tip would otherwise draw a long mitre spike past the corner
        page.draw_polyline(sp + [sp[0]], color=INK, fill=(1, 1, 1), width=0.8, closePath=True, lineJoin=1)
        if kind == "landing":                     # the door's swing over the landing
            hinge, end, open_ = pts[0], pts[1], pts[3]
            rad = abs(end - hinge)
            a0 = math.atan2(end.y - hinge.y, end.x - hinge.x)
            a1 = math.atan2(open_.y - hinge.y, open_.x - hinge.x)
            da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
            arc = [pymupdf.Point(hinge.x + rad * math.cos(a0 + da * i / 24), hinge.y + rad * math.sin(a0 + da * i / 24))
                   for i in range(25)]
            arc = [q for q in arc if abs(q - hinge) <= rad + 0.01]
            page.draw_polyline(arc, color=INK, width=0.4)
        key = "landing" if kind == "landing" else x.get("label_" + kind.replace("-", "_"), x["label_std"])
        tips.setdefault(key, []).append(pymupdf.Point(sum(p.x for p in sp) / len(sp), sum(p.y for p in sp) / len(sp)))
    if shapes:                                    # columns inside a room cut the ramps too
        for r in rooms:
            for hole in r.get("holes_pt") or []:
                hp = [tx(p) for p in hole]
                page.draw_polyline(hp + [hp[0]], color=INK, fill=(1, 1, 1), width=0.6, closePath=True)

    # ---- one callout per buildup in the right-hand column, ordered like the rooms they point to
    items = []
    for g in (buildups if not free else []):
        rs = sys_rooms[g["key"]]
        if not rs:
            continue
        big = max(rs, key=lambda r: r.get("area_sf") or 0)
        lp = label_point(big["outline_pt"])
        tip = tx(lp)
        which = side.get(g["key"], "right")
        # point just inside the room's edge that faces the callout, level with the room's middle
        poly = [tx(q) for q in big["outline_pt"]]
        xs = []
        for i in range(len(poly)):
            a_, b_ = poly[i - 1], poly[i]
            if (a_.y > tip.y) != (b_.y > tip.y):
                xs.append(a_.x + (tip.y - a_.y) * (b_.x - a_.x) / (b_.y - a_.y))
        if xs:
            edge = max(xs) if which == "right" else min(xs)
            width = max(xs) - min(xs)
            inset = min(30.0, 0.25 * width)
            cand = pymupdf.Point(edge - inset if which == "right" else edge + inset, tip.y)
            holes = [[tx(q) for q in h] for h in big.get("holes_pt") or []]
            def clear(p):                           # how far a point is from the room's outline
                best_ = 1e9
                for i in range(len(poly)):
                    a_, b_ = poly[i - 1], poly[i]
                    dx, dy = b_.x - a_.x, b_.y - a_.y
                    t = max(0.0, min(1.0, ((p.x - a_.x) * dx + (p.y - a_.y) * dy) / ((dx * dx + dy * dy) or 1e-9)))
                    best_ = min(best_, math.hypot(a_.x + dx * t - p.x, a_.y + dy * t - p.y))
                return best_
            # only if it's well inside (not on an edge running level with it, e.g. beside a notch
            # at the door: Maroon Creek), else the room's middle
            if point_in_polygon(cand.x, cand.y, [(q.x, q.y) for q in poly]) and clear(cand) >= inset * 0.6 and \
                    not any(point_in_polygon(cand.x, cand.y, [(q.x, q.y) for q in h]) for h in holes):
                tip = cand
        items.append((tip, g["callout"].split("\n"), which))
    cx = target.x0 + (clip.x0 + clip.width / 2 - clip.x0) * s
    left_ok = "left" in side.values()
    for lab in (ramp_labels if not free else []):
        cands = tips.get(lab["key"])
        if cands:
            only_left = "right" not in side.values()
            which = "left" if left_ok and (only_left or min(c.x for c in cands) < cx) else "right"
            # point at the piece nearest the callout's side, so the leader stays short
            tip = min(cands, key=lambda c: c.x) if which == "left" else max(cands, key=lambda c: c.x)
            items.append((tip, lab["text"].split("\n"), which))

    def leader(which, cw, ch, yc, tip):
        """The callout's leader: (start, elbow, tip), as drawn below."""
        if which == "left":
            x1 = max(frame.x0 + 12 + cw, target.x0 - 34)
            start = pymupdf.Point(x1 + 4, yc - ch / 2 + cs * 0.5)
            return start, pymupdf.Point(start.x + 22, start.y), tip
        x0 = min(frame.x1 - 12 - cw, target.x1 + 34)
        start = pymupdf.Point(x0 - 4, yc - ch / 2 + cs * 0.5)
        return start, pymupdf.Point(start.x - 22, start.y), tip

    def crosses(p1, p2, q1, q2):
        def o(a, b, c):
            return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        return o(p1, p2, q1) * o(p1, p2, q2) < 0 and o(q1, q2, p1) * o(q1, q2, p2) < 0

    def stack(col, which, first_bottom):
        prev_bottom, these = first_bottom, []
        for tip, lines, _ in col:
            cw = max(fonts.width(l, cs) for l in lines)
            ch = len(lines) * cpitch
            yc = max(tip.y, prev_bottom + 10 + ch / 2)
            these.append([tip, lines, cw, ch, yc, which])
            prev_bottom = yc + ch / 2
        over = prev_bottom - (notes_top - 8)        # stack runs past the bottom: shift it up
        if over > 0:
            for it in these:
                it[4] -= over
        return these

    def tangle(these):
        """(leader crossings, total leader length) for a stacked column."""
        legs = [leader(it[5], it[2], it[3], it[4], it[0]) for it in these]
        n = sum(crosses(a[i], a[i + 1], b[j], b[j + 1]) for x, a in enumerate(legs) for b in legs[x + 1:]
                for i in range(2) for j in range(2))
        return n, sum(abs(a[1] - a[0]) + abs(a[2] - a[1]) for a in legs)

    placed = []
    for which in ("left", "right"):
        col = sorted([it for it in items if it[2] == which], key=lambda it: it[0].y)
        first_bottom = target.y0 - 1e9
        if which == "left" and col and target.x0 - 34 - max(fonts.width(l, cs) for it in col for l in it[1]) < bom_right + 10:
            first_bottom = bom_bottom + 4         # don't run into the BOM
        best = stack(col, which, first_bottom)
        if 1 < len(col) <= 7:                     # the order with the fewest crossing leaders
            import itertools
            score = tangle(best)
            for order in itertools.permutations(col):
                these = stack(list(order), which, first_bottom)
                sc = tangle(these)
                if sc < score:
                    best, score = these, sc
        placed += best
    spots = []
    for tip, lines, cw, ch, yc, which in placed:
        # callouts sit just beside the drawing (not out at the border), so leaders stay short
        if which == "left":
            x1 = max(frame.x0 + 12 + cw, target.x0 - 34)
            spot = pymupdf.Rect(x1 - cw, yc - ch / 2, x1, yc + ch / 2)
        else:
            x0 = min(frame.x1 - 12 - cw, target.x1 + 34)
            spot = pymupdf.Rect(x0, yc - ch / 2, x0 + cw, yc + ch / 2)
        spots.append((tip, lines, spot, which))
    for tip, lines, spot, which in (free or spots):
        if free:                                  # over the faded plan: keep the words readable
            page.draw_rect(spot + (-2, -1, 2, 1), color=None, fill=(1, 1, 1), width=0)
        for i, l in enumerate(lines):
            # like the template's leaders: left-side callouts are right-aligned against their leader
            lx = spot.x1 - fonts.width(l, cs) if which == "left" else spot.x0
            page.insert_text((lx, spot.y0 + cs * 0.8 + i * cpitch), l, fontname=font, fontsize=cs)
        if which == "left":
            start = pymupdf.Point(spot.x1 + 4, spot.y0 + cs * 0.5)
            elbow = pymupdf.Point(start.x + 22, start.y)
        else:
            start = pymupdf.Point(spot.x0 - 4, spot.y0 + cs * 0.5)
            elbow = pymupdf.Point(start.x - 22, start.y)
        page.draw_polyline([start, elbow, tip], color=INK, width=0.9)
        ang = math.atan2(tip.y - elbow.y, tip.x - elbow.x)
        head = [tip, pymupdf.Point(tip.x - 7 * math.cos(ang - 0.3), tip.y - 7 * math.sin(ang - 0.3)),
                pymupdf.Point(tip.x - 7 * math.cos(ang + 0.3), tip.y - 7 * math.sin(ang + 0.3))]
        page.draw_polyline(head, color=INK, fill=INK, closePath=True, width=0.3)

    # ---- template: border, title block, logo, fixed text and the filled-in fields
    template.draw(page, tpl, {"project": meta.get("project", ""), "drawn_by": meta.get("drawn_by", ""),
                              "date": meta.get("date") or datetime.date.today().isoformat(),
                              "dwg_no": meta.get("dwg_no", ""), "reviewed_by": company.get("reviewed_by", "")}, fonts)

    # ---- notes, in the template's Notes box (shrunk to fit when there are many)
    nf = tpl["fields"]["notes"]
    nb = pymupdf.Rect(nf["x"], nf["y"], nf["x"] + nf["width"], frame.y1 - 6)
    html = "".join(f"<p>{_esc(n)}</p>" for n in takeoff.notes)
    if fonts.file:
        from pathlib import Path as _P
        arch = pymupdf.Archive(str(_P(fonts.file).parent))
        css = (f"@font-face {{font-family: G; src: url({_P(fonts.file).name});}} "
               f"p{{font-family:G;font-size:{nf['size']}pt;margin:0 0 {nf['size'] * 0.9:.1f}pt 0;line-height:1.2}}")
        page.insert_htmlbox(nb, html, css=css, archive=arch, scale_low=0)
    else:
        page.insert_htmlbox(nb, html, css=f"p{{font-family:serif;font-size:{nf['size']}pt;margin:0 0 4pt 0}}", scale_low=0)

    doc.save(out_path, garbage=3, deflate=True)
    doc.close()


def render_preview(out_path, plan_doc, page_no, rooms):
    """Copy of the plan page with every detected room outlined and numbered."""
    src = plan_doc[page_no - 1]
    doc = pymupdf.open()
    page = doc.new_page(width=src.rect.width, height=src.rect.height)
    page.show_pdf_page(page.rect, plan_doc, page_no - 1)
    for r in rooms:
        pts = [pymupdf.Point(*p) for p in r["outline_pt"]]
        color = (0.9, 0.55, 0.0) if r.get("included") else (0.18, 0.42, 0.56)
        page.draw_polyline(pts + [pts[0]], color=color, fill=color, fill_opacity=0.12,
                           width=3 if r.get("included") else 2, closePath=True)
        for h in r.get("holes_pt") or []:
            hp = [pymupdf.Point(*p) for p in h]
            page.draw_polyline(hp + [hp[0]], color=color, fill=(1, 1, 1), width=1, closePath=True)
        cx, cy = label_point(r["outline_pt"])
        tag = str(r["id"])
        w = pymupdf.get_text_length(tag, fontname="hebo", fontsize=22)
        page.draw_rect(pymupdf.Rect(cx - w / 2 - 5, cy - 19, cx + w / 2 + 5, cy + 7), color=color, fill=(1, 1, 1), width=1.5)
        page.insert_text((cx - w / 2, cy), tag, fontname="hebo", fontsize=22, color=color)
    doc.save(out_path, garbage=3, deflate=True)
    doc.close()
