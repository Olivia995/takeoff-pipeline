"""Transitions (ramps): where people walk onto a buildup that sits above the floor beside it.

Rules from the takeoff readme and Pliteq's answers (Nov 2026):

- The ramp's height is the buildup height (sum of the layers' thicknesses), less the
  floor on the other side: another buildup where two touch, or a recess depth.
    up to 30 mm        FIT30 ramps (trimmed below 30 mm)
    over 30, to 70 mm  FIT70 ramps, trimmed (cut down to the height, so shorter in proportion)
    over 70 mm         h / 70 rounded up FIT70 tiers in a row. The first one, at the
                       opening, sits on the floor and takes what's left over (trimmed,
                       and drawn shorter in proportion); every tier after it is a full
                       70 mm on wood blocking (supplied by others). The sheet labels them
                       "GenieMat FIT70 ... Transition" and "2x GenieMat FIT70 ...
                       Transition over Wood Blocking". Past sheets: Lifetime Sky (85 mm),
                       Park Hyatt (86 mm) and Westerly (120 mm) are 1 + 1 over blocking,
                       Visa Foster City (178 mm) is 1 + 2x over blocking (10.5 LF ADA).
- Each tier's ADA ramp has standard wings down both sides, as long as all the ADA ramps up
  to and including it (they nest, the outermost running the whole length: Visa FCM-1,
  68 LF standard).
  A height that isn't a full tier adds the "shimmed/trimmed" note; blocking adds the
  wood-blocking note.
- Ramps sit on the buildup (the high side), starting at the opening.
- An ADA doorway is covered by ADA ramps across its whole width, in 42" pieces (a 48"
  door takes 2 = 7 LF, Parkview PTC-1). On a wide open edge the drafter can choose one
  ADA ramp instead, with standard ramps (wings) filling the rest wall to wall. Each exposed
  side of the ADA ramp gets a triangular wing cut from a piece of its own (The Heartwood
  Library THL-1: one ADA ramp = 3.5 LF ADA + 8 LF standard). Any other opening gets
  standard ramps along its length.
- Standard pieces are 48"; each edge rounds up to the next 6". Edges of 42" or less can be cut
  from a shared piece: up to 3 of them, using no more than 87.5% (42") of it together.
- A door that swings onto the buildup gets a FIT08 landing in front of it, door width x
  (width + 2"). The ramp starts at the far edge of the landing, and each side of the
  landing gets a standard ramp along it (Parkview PTC-1: 4 standard ramps around the door).
- Standard ramps come in 48" pieces. Each edge is rounded up to the next 6"; edges of 42"
  or less can share a piece, up to 3 per piece. rules.yaml transition_overage adds spare
  pieces. ADA ramps get no overage.
- Every tier is counted: a 2x ramp is twice the LF.
- Adhesive (FAS1): (LF x depth / 12) / (((150/2)/75)*125*4), i.e. SF / 500; depth 32.5" FIT70 ADA,
  18" FIT70 standard, 13" FIT30 ADA, 12" FIT30 standard.
"""
import math

from .detect import point_in_polygon

ADA_IN = 42
ROUND_IN = 6
ADA_TOL_IN = 3          # up to 45" is still one 42" ADA ramp, trimmed to fit (Cottages Cardio: 44.4" = 1 ADA)


def buildup_mm(cat, layers):
    """Height of an expanded buildup in mm, and the layers with no thickness on file."""
    P = cat["products"]
    total, missing = 0.0, []
    for l in layers:
        p = P[l["code"]]
        if p["family"] == "pmi":
            continue
        if p.get("thickness_mm") is None:
            missing.append(p["display"])
        else:
            total += p["thickness_mm"]
    return total, missing


def ramp_for(h):
    """(series, tiers, trimmed/shimmed, on wood blocking) for a ramp height in mm."""
    if h <= 30 + 0.5:
        return "FIT30", 1, h < 30 - 0.5, False
    if h <= 70 + 0.5:
        return "FIT70", 1, h < 70 - 0.5, False
    n = math.ceil(h / 70 - 0.5 / 70)
    return "FIT70", n, h - 70 * (n - 1) < 70 - 0.5, True


def tier_scale(h, tiers, full_mm=70):
    """Each tier's depth as a share of a full ramp, from the opening inward. A ramp taller
    than the step it covers is trimmed: cut down to the height and so shorter in proportion
    (a 70 mm ramp on a 45 mm buildup). In a stack only the first tier, at the opening, is
    trimmed; the full tiers sit on top (Visa FCM-1)."""
    first = min(1.0, max(0.05, (h - full_mm * (tiers - 1)) / full_mm))
    return [first] + [1.0] * (tiers - 1)


def name(series, ada):
    return f"GenieMat {series} {'ADA' if ada else 'Standard'} Transition"


def label(series, ada, tiers, blocking=False, over=None):
    """Callout for one kind of ramp on the sheet. over: what the ramp sits on, e.g. the
    higher room's underlayment carried on under a ramp between two buildups."""
    t = (f"{tiers}x " if tiers > 1 else "") + name(series, ada)
    if blocking:
        return t + " over\nWood Blocking (supplied by others)"
    return t + (" over\n" + over if over else "")


def _seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy or 1.0
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.hypot(a[0] + t * dx - p[0], a[1] + t * dy - p[1])


def _ring_dist(p, ring):
    return min(_seg_dist(p, ring[i - 1], ring[i]) for i in range(len(ring)))


def sides(o, rooms, ppf):
    """Which rooms an opening (a, b in PDF points) lies between: (high-side candidates).
    Returns (room index on one side or None, room index on the other side or None, unit
    normal pointing from the first side to the second)."""
    (ax, ay), (bx, by) = o["a"], o["b"]
    L = math.hypot(bx - ax, by - ay) or 1.0
    nx, ny = -(by - ay) / L, (bx - ax) / L
    mx, my = (ax + bx) / 2, (ay + by) / 2
    d = 0.75 * ppf

    def inside(x, y):
        for i, r in enumerate(rooms):
            if point_in_polygon(x, y, r["outline_pt"]):
                return i
        return None
    s1, s2 = inside(mx + nx * d, my + ny * d), inside(mx - nx * d, my - ny * d)
    if s1 is None and s2 is None:                       # marked a little off the outline
        near = min(range(len(rooms)), key=lambda i: _ring_dist((mx, my), rooms[i]["outline_pt"]), default=None)
        if near is None or _ring_dist((mx, my), rooms[near]["outline_pt"]) > 3 * ppf:
            return None, None, (nx, ny)
        # the room's side is the one whose sample point is nearer its inside
        c = rooms[near]["outline_pt"]
        cx, cy = sum(p[0] for p in c) / len(c), sum(p[1] for p in c) / len(c)
        if (cx - mx) * nx + (cy - my) * ny > 0:
            return near, None, (-nx, -ny)
        return near, None, (nx, ny)
    if s1 == s2:
        s2 = None
    if s1 is None:
        return s2, None, (nx, ny)
    return s1, s2, (-nx, -ny)


def _depths(x):
    """Total standard and ADA depth (inches) of a ramp record, all tiers."""
    scale = x.get("tier_scale") or [1.0] * max(1, x.get("tiers", 1))
    return x["depth_in"]["standard"] * sum(scale), x["depth_in"]["ada"] * sum(scale)


def _joins(ramps, ppf, rooms=None):
    """How each end of a standard ramp is cut, as drafted (Lifetime Sky LFS-3 DWG):
    - into the next standard ramp: mitred at a square corner; at a slanted wall the straighter
      run ends square and the slanted one meets it; square where the edge carries straight on
    - into an ADA ramp: a 45 degree cut from the ADA ramp's corner at the wall out to the
      inner edge (the ADA ramp's triangular wing fills the rest)
    - against a wall that turns into the room (an inside corner): butted along that wall
    - otherwise (a door jamb) None: the 45 degree cut-back with its triangle piece (BFTL-1).
    Sets cut_a / cut_b (a line, two PDF points) on each standard ramp."""
    tol = 2.0
    ft = ppf / 12
    for r in ramps:
        r["cut_a"] = r["cut_b"] = None
        r["corner_a"] = r["corner_b"] = False
        r["std_at_a"] = r["std_at_b"] = False
    for r in ramps:
        if r["ada"] or not r.get("a") or r.get("landing_in") or r.get("outside"):
            continue
        ra, rb = r["a"], r["b"]
        L = math.dist(ra, rb) or 1.0
        u = ((rb[0] - ra[0]) / L, (rb[1] - ra[1]) / L)
        n = (-r["normal"][0], -r["normal"][1])             # into the room
        W = _depths(r)[0] * ft
        for end, pt in (("a", ra), ("b", rb)):
            for q in ramps:
                if q is r or q["room_index"] != r["room_index"] or not q.get("a") or q.get("landing_in"):
                    continue
                qend = "a" if math.dist(q["a"], pt) <= tol else "b" if math.dist(q["b"], pt) <= tol else None
                if not qend:
                    continue
                qa, qb = q["a"], q["b"]
                QL = math.dist(qa, qb) or 1.0
                qu = ((qb[0] - qa[0]) / QL, (qb[1] - qa[1]) / QL)
                qn = (-q["normal"][0], -q["normal"][1])
                square = [pt, (pt[0] + n[0], pt[1] + n[1])]
                if q["ada"]:
                    q["std_at_" + qend] = True             # its wing meets this run's inner edge
                    has = q.get("wing_l", True) if qend == "a" else q.get("wing_r", True)
                    if not has or q.get("ada_width") == "one":
                        cut = square
                    else:                                  # 45 degrees out from the ADA ramp's wall corner
                        sgn = -1 if qend == "a" else 1
                        cut = [pt, (pt[0] + qu[0] * sgn * W + qn[0] * W, pt[1] + qu[1] * sgn * W + qn[1] * W)]
                elif abs(u[0] * qu[1] - u[1] * qu[0]) < 0.05:
                    cut = square                           # the edge carries straight on
                elif abs(u[0] * qu[1] - u[1] * qu[0]) > 0.966:   # a square corner (75-105 degrees): mitre
                    X = _cross_lines((ra[0] + n[0] * W, ra[1] + n[1] * W), u,
                                     (qa[0] + qn[0] * W, qa[1] + qn[1] * W), qu)
                    cut = [pt, X] if X and math.dist(X, pt) > 1e-6 else square
                else:                                      # a slanted wall: the straighter run ends square
                    straight = min(abs(u[0]), abs(u[1])) <= min(abs(qu[0]), abs(qu[1]))
                    m = n if straight else qn
                    cut = [pt, (pt[0] + m[0], pt[1] + m[1])]
                r["cut_" + end] = cut
                break
            else:
                ring = rooms[r["room_index"]].get("outline_pt") if rooms and 0 <= r["room_index"] < len(rooms) else None
                if not ring:
                    continue
                w = _wall_turning_in(ring, pt, u, n, ft * 0.75)
                if w:
                    # right in an inside corner: butt into the wall (Lifetime Sky; Cottages Cardio)
                    r["cut_" + end] = [pt, (pt[0] + w[0], pt[1] + w[1])]
                elif not _edge_carries_on(ring, pt, (-u[0], -u[1]) if end == "a" else u, n, ft):
                    # the room's edge turns within a foot past the end (a corner, or round the end
                    # of a wall pier): cut back with its triangle, but no piece is ordered for it
                    # (Cottages Cardio, Maroon Creek: however the outline is traced there)
                    r["corner_" + end] = True


def _edge_carries_on(ring, pt, out, n, ft):
    """A door jamb: past the ramp's end the room's edge carries straight on for at least a foot,
    floor on the room side (n) and none on the other. False when the outline turns within that
    foot, into the room or round the end of a wall."""
    for d in (2, 4, 6, 8, 10, 12):
        x, y = pt[0] + out[0] * d * ft, pt[1] + out[1] * d * ft
        # 3" either side, so an opening drawn an inch off the room's edge still reads right
        if not point_in_polygon(x + n[0] * 3 * ft, y + n[1] * 3 * ft, ring) or point_in_polygon(x - n[0] * 3 * ft, y - n[1] * 3 * ft, ring):
            return False
    return True


def _wall_turning_in(ring, pt, u, n, tol):
    """At an end of a ramp's edge: is it up against a wall? True when the room's outline has a
    corner within 3" of the end and the wall from there heads into the room (an inside corner,
    or the side of a door recess the ramp sits in). Returns that wall's direction, else None.
    tol is 3/4" (ft in _joins is points per inch)."""
    k = min(range(len(ring)), key=lambda i: math.dist(ring[i], pt))
    if math.dist(ring[k], pt) > tol * 4:              # an opening drawn an inch or two off still counts
        return None
    leg = tol * 4                                     # judge the wall over 3", so a 1" jog in a traced
                                                      # outline isn't a wall,
    for step in (-1, 1):                              # so a 1" jog in a traced outline isn't a corner
        j, q = k, None
        for _ in range(len(ring)):
            j = (j + step) % len(ring)
            if math.dist(ring[j], pt) >= leg:
                q = ring[j]
                break
        if q is None:
            continue
        d = math.dist(q, pt)
        w = ((q[0] - pt[0]) / d, (q[1] - pt[1]) / d)
        if abs(w[0] * u[1] - w[1] * u[0]) > 0.26 and w[0] * n[0] + w[1] * n[1] > 0.26:
            return w
    return None


def _cross_lines(p, d, q, e):
    """Where the line p + s*d crosses q + t*e (None if parallel)."""
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-9:
        return None
    s = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return (p[0] + d[0] * s, p[1] + d[1] * s)


def _shared_pieces(edges, max_in, max_per):
    """Standard pieces needed for short edges: up to `max_per` edges can be cut from one 48"
    piece as long as together they use no more than `max_in` (87.5% of the piece = 42").
    First fit, longest first. FIT30 ADA side wings (13", 18" rounded) go two to a piece;
    FIT70 ones (32.5", 36" rounded) need a piece each."""
    bins = []                                    # [used inches, edge count]
    for e in sorted(edges, reverse=True):
        for b in bins:
            if b[0] + e <= max_in + 1e-6 and b[1] < max_per:
                b[0] += e
                b[1] += 1
                break
        else:
            bins.append([e, 1])
    return len(bins)


def plan(cat, rooms, openings, settings, pt_per_ft=None):
    """rooms: engine rooms (with "_calc", optional outline_pt). openings:
    [{a, b} in PDF points, or {length_in, room: index, other: index}; type "ada"|"std";
    sides (ADA sides needing a wing: 2, 1 or 0); landing (door swings onto the buildup)}].
    settings: {recess_mm, no_ada}.
    Returns {lines: {(series, ada): LF}, pieces, adhesive (FAS1 pails), landing_sf,
    fire: [(note id, fill)], warnings, ramps: [per-opening drawing info], labels}."""
    R = cat["rules"]["ramps"]
    P = cat["products"]
    recess_mm = float(settings.get("recess_mm") or 0)
    heights, missing = [], []
    for r in rooms:
        h, miss = buildup_mm(cat, r["_calc"])
        heights.append(h)
        missing += miss
    out = {"lines": {}, "pieces": {}, "adhesive": 0.0, "landing_sf": 0.0, "fire": [], "warnings": [],
           "ramps": [], "labels": []}
    ada_n, std_long, std_short = {}, {}, {}
    used = set()

    def fire(nid, fill=None):
        if (nid, fill) not in out["fire"]:
            out["fire"].append((nid, fill))

    for k, o in enumerate(openings):
        if "length_in" in o:
            L_in = float(o["length_in"])
            hi, lo, normal = o.get("room"), o.get("other"), (0, 1)
        else:
            ppf = pt_per_ft or 1.0
            L_in = math.dist(o["a"], o["b"]) / ppf * 12
            hi, lo, normal = sides(o, rooms, ppf)
        if hi is None:
            out["warnings"].append(f"Access point {k + 1} isn't on a takeoff room's edge, so it has no ramp.")
            continue
        if lo is not None and heights[lo] > heights[hi]:
            hi, lo, normal = lo, hi, (-normal[0], -normal[1])
        h = heights[hi] - (heights[lo] if lo is not None else 0) - recess_mm
        if h <= 0.5:
            out["warnings"].append(f"Access point {k + 1} ({rooms[hi].get('name')}): no height difference, so no ramp.")
            continue
        series, tiers, trimmed, blocking = ramp_for(h)
        # Between two buildups the ramp runs off the higher one onto the lower room's floor, so
        # it is drawn (and its SF taken) on the lower room's side, sitting on the higher room's
        # underlayment carried on under it (Cottages Cardio: "FIT30 Standard Transition over
        # GenieMat FFNC25NBN"). Onto the slab (no room beyond) it sits inside the room.
        sit, nrm, over, extra = hi, normal, None, []
        if lo is not None:
            sit, nrm = lo, (-normal[0], -normal[1])
            lo_codes = {l["code"] for l in rooms[lo]["_calc"]}
            extra = [l["code"] for l in rooms[hi]["_calc"][1:] if l["code"] not in lo_codes]
            if extra:
                over = " and ".join(P[c].get("display") or c for c in extra if c in P) or None
        outside = False
        if o.get("flip"):
            # the drafter's choice: the ramp on the other side of the opening (e.g. a closet
            # door would swing into it on the lower side)
            if lo is not None:
                sit, nrm, over, extra = hi, normal, None, []     # on the higher room's floor
            else:
                nrm, outside = (-normal[0], -normal[1]), True     # out onto the slab
        sfx = "-over-" + over if over else ""
        scale = tier_scale(h, tiers, 30 if series == "FIT30" else 70)
        ada = o.get("type") == "ada" and not settings.get("no_ada")
        depth = R["depth_in"][series]
        edges, wings = [], []                         # edges: one per tier; wings: listed per tier
        n_sides = int(o.get("sides", 2))
        full = o.get("ada_width", "full") != "one"
        n_ada = max(1, math.ceil((L_in - ADA_TOL_IN) / ADA_IN - 1e-6)) if full else 1
        land_in = L_in + R.get("landing_extra_in", 2) if o.get("landing") else 0.0
        # which sides of an ADA ramp get a wing: the drafter's choice, less any side where the
        # ramp ends at a corner of the room (a wing there would hang outside the room)
        wing_l, wing_r = n_sides == 2, n_sides >= 1
        if ada and "a" in o and rooms[sit].get("outline_pt") and not o.get("landing") and not outside:
            ppf = pt_per_ft or 1.0
            (ax, ay), (bx, by) = o["a"], o["b"]
            Lp = math.hypot(bx - ax, by - ay) or 1.0
            ux, uy = (bx - ax) / Lp, (by - ay) / Lp
            ix, iy = -nrm[0], -nrm[1]
            s = 4 / 12 * ppf                          # 4" past the end, 4" into the room
            ring = rooms[sit]["outline_pt"]
            if wing_l and not point_in_polygon(ax - ux * s + ix * s, ay - uy * s + iy * s, ring):
                wing_l = False
            if wing_r and not point_in_polygon(bx + ux * s + ix * s, by + uy * s + iy * s, ring):
                wing_r = False
            n_sides = int(wing_l) + int(wing_r)
        if ada:
            ada_n[series] = ada_n.get(series, 0) + n_ada * tiers
            # each exposed side of each tier's ADA ramp gets a wing as long as the ADA ramps
            # up to it (one tier: as long as the ADA ramp is deep)
            run = 0.0
            for f in scale:
                run += depth["ada"] * f
                wings += [run] * n_sides
            rem = L_in - n_ada * ADA_IN
            if rem > ADA_TOL_IN:
                parts = 2 if n_sides == 2 else 1
                edges += [rem / parts] * parts
        else:
            edges.append(L_in)
        if land_in:                                   # standard ramps down the sides of the landing
            edges += [land_in] * n_sides
        # straight edges round up to the next 6"; the triangular wings beside an ADA ramp are cut
        # to their own length, so two short ones (a trimmed ramp's) can come from one piece
        for e, rounded in [(e, True) for e in edges * tiers] + [(w, False) for w in wings]:
            if rounded or e > R["standard_max_share_in"]:
                e = math.ceil(e / ROUND_IN - 1e-6) * ROUND_IN
            if e <= R["standard_max_share_in"]:
                std_short.setdefault(series, []).append(e)
            else:
                std_long[series] = std_long.get(series, 0) + math.ceil(e / R["standard_piece_in"] - 1e-6)
        if trimmed:
            fire("shimmed")
        if blocking:
            fire("blocking_tiered")
            codes = {l["code"] for l in rooms[hi]["_calc"]}
            if codes & {"FFNC25", "FFNC25NBN", "FFNC50NBN", "FFNC75NBN"}:
                out["warnings"].append(f"{rooms[hi].get('name')} has FFNC25 in its buildup: if the tiered ramp sits on "
                                       "FFNC25 instead of wood blocking, tick the FFNC25NBN tiered-ramping note.")
        if ada:
            fire("ada_consultant")
        fams = [P[l["code"]]["family"] for l in rooms[hi]["_calc"]]
        if any(c.startswith("FFNC") and c.endswith("NBN") for c in extra):
            # the FFNC under the higher room carries on under the ramp and holds up the tile
            # edge, so no wood support (Cottages Cardio)
            fire("ffnc_tiered")
        else:
            for i in range(len(fams) - 1):
                if fams[i] in ("fit_tile", "xfit") and fams[i + 1] in ("ff", "ff_panel", "plywood"):
                    fire("tile_edge_support")
        landing = 0.0
        if land_in:
            landing = L_in * land_in / 144
            out["landing_sf"] += landing
            fire("doors_trimmed")                    # the door swings over GenieMat (Parkview PTC-1)
        for kind in ([True] if ada else []) + ([False] if (not ada or L_in > n_ada * ADA_IN + 0.5 or n_sides) else []):
            # the first tier sits on the floor; the rest are on wood blocking
            for key, n, wb in [(f"{series}-{'ada' if kind else 'std'}-1{sfx}", 1, False)] + \
                    ([(f"{series}-{'ada' if kind else 'std'}-{tiers - 1}-wb", tiers - 1, True)] if tiers > 1 else []):
                if key not in used:
                    used.add(key)
                    out["labels"].append({"key": key, "series": series, "ada": kind, "tiers": n,
                                          "text": label(series, kind, n, wb, None if wb else over)})
        out["ramps"].append({"index": k, "room_index": sit, "a": o.get("a"), "b": o.get("b"), "normal": list(nrm),
                             "series": series, "tiers": tiers, "blocking": blocking, "ada": ada, "sides": n_sides, "wing_l": wing_l, "wing_r": wing_r, "height_mm": round(h, 1),
                             "over_index": hi if extra else None, "over_codes": extra,
                             "length_in": round(L_in, 1), "room": rooms[hi].get("name"),
                             "other": rooms[lo].get("name") if lo is not None else None, "trimmed": trimmed,
                             "landing_sf": round(landing, 1), "landing_in": land_in, "n_ada": n_ada if ada else 0, "ada_width": "full" if full else "one",
                             "depth_in": depth, "tier_scale": scale,
                             "label_std": f"{series}-std-1{sfx}", "label_ada": f"{series}-ada-1{sfx}",
                             "label_std_wb": f"{series}-std-{tiers - 1}-wb", "label_ada_wb": f"{series}-ada-{tiers - 1}-wb",
                             "flip": bool(o.get("flip")), "outside": outside})

    _joins(out["ramps"], pt_per_ft or 1.0, rooms)
    # a standard ramp ending at a door jamb (no wall or other ramp to stop against) is cut back
    # at 45 degrees and finished with a triangle; a run's triangles (one or both ends) are cut
    # from one extra piece per tier (Cottages Cardio: 6 runs + 2 pieces of ADA wings + 2 runs
    # with triangles + 1 spare = 11; Modera Mystic: 6 + 2 door ramps with 2 each + 1 = 9).
    # A triangle drawn only because the run stops just short of a corner isn't ordered
    # (Maroon Creek: 6 + 3 wings + 1 run at a jamb + 1 spare = 11)
    for x in out["ramps"]:
        if not x["ada"] and x.get("a") and not x.get("landing_in"):
            if any(x.get("cut_" + e) is None and not x.get("corner_" + e) for e in ("a", "b")):
                std_long[x["series"]] = std_long.get(x["series"], 0) + x["tiers"]
    if any(x["landing_in"] for x in out["ramps"]):
        out["labels"].append({"key": "landing", "series": None, "ada": False, "tiers": 1, "text": "GenieMat FIT08"})
    extra_tiers = cat["rules"].get("transition_overage") or []
    for series in ("FIT70", "FIT30"):
        n = std_long.get(series, 0) + _shared_pieces(std_short.get(series, []), R["standard_max_share_in"],
                                                     R["standard_max_per_piece"])
        if n:
            extra = next((t["extra"] for t in extra_tiers if t["max"] is None or n <= t["max"]), 0)
            n += extra
            out["pieces"][(series, False)] = n
            out["lines"][(series, False)] = n * R["standard_piece_in"] / 12
        if ada_n.get(series):
            out["pieces"][(series, True)] = ada_n[series]
            out["lines"][(series, True)] = ada_n[series] * R["ada_piece_in"] / 12
    ad = R["adhesive_depth_in"]
    for (series, ada), lf in out["lines"].items():
        sf = lf * ad[series]["ada" if ada else "standard"] / 12
        out["adhesive"] += sf / R["adhesive_fas1_sf_per_pail"]
    if out["ramps"] and missing:
        out["warnings"].append("No thickness on file for " + ", ".join(sorted(set(missing))) +
                               ": the ramp heights leave it out.")
    return out
