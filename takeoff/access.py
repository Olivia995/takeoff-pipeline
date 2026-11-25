"""Access points: where people walk onto the buildup.

On a slab that isn't recessed, the buildup sits above the surrounding floor, so every
opening into a takeoff area needs a transition (ramp) and at least one needs to be an
ADA ramp. This finds the openings: stretches of an area's edge with no wall behind them
(doorways, wide openings to corridors), skipping edges shared with another included area.
"""
import math

import numpy as np

from .detect import (clear_cache, point_in_polygon, polygon_area, render_page, wall_mask,
                     wall_mask_vector)


def _outward(poly):
    """+1 if the polygon is clockwise on screen (y down), so (dy, -dx) points outward."""
    a = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i - 1]
        x1, y1 = poly[i]
        a += x0 * y1 - x1 * y0
    return 1 if a > 0 else -1


def find_access_points(page, rooms, pt_ft, min_open_ft=2.5, band_ft=(0.15, 2.0), step_ft=0.25, all_rooms=None):
    """rooms: included areas with outline_pt (PDF points).
    Returns [{a, b, length_ft, room}] in PDF points, widest first."""
    clear_cache()
    img, s = render_page(page)
    # thin lines count (glazing is a wall); door swing arcs don't (they cross doorways)
    walls = wall_mask_vector(page, s, min_stroke_pt=0.2, curves=False)
    if walls is None:
        walls = wall_mask(img)
    H, W = walls.shape
    others = [r["outline_pt"] for r in (all_rooms or rooms)]   # edges onto any included area are skipped

    def walled(x, y, nx, ny):
        d0, d1 = band_ft[0] * pt_ft, band_ft[1] * pt_ft
        d = d0
        while d <= d1:
            px, py = int((x + nx * d) * s), int((y + ny * d) * s)
            if 0 <= px < W and 0 <= py < H:
                if walls[py, px]:
                    return True
            else:
                return True                       # off the sheet: treat as closed
            d += 0.6 / s                          # about every pixel
        return False

    out = []
    for r in rooms:
        poly = r["outline_pt"]
        sign = _outward(poly)
        n = len(poly)
        for i in range(n):
            (ax, ay), (bx, by) = poly[i], poly[(i + 1) % n]
            L = math.hypot(bx - ax, by - ay)
            if L < min_open_ft * pt_ft:
                continue
            ux, uy = (bx - ax) / L, (by - ay) / L
            nx, ny = sign * uy, -sign * ux
            # outward check: a point just outside should not be inside the room
            mx, my = (ax + bx) / 2 + nx * 2, (ay + by) / 2 + ny * 2
            if point_in_polygon(mx, my, poly):
                nx, ny = -nx, -ny
            k = max(2, int(L / (step_ft * pt_ft)))
            state = []
            for j in range(k + 1):
                t = j / k
                x, y = ax + ux * L * t, ay + uy * L * t
                ox, oy = x + nx * 0.5 * pt_ft, y + ny * 0.5 * pt_ft
                shared = any(o is not poly and point_in_polygon(ox, oy, o) for o in others)
                state.append(not shared and not walled(x, y, nx, ny))
            j = 0
            while j <= k:
                if not state[j]:
                    j += 1
                    continue
                j0 = j
                while j <= k and state[j]:
                    j += 1
                t0, t1 = j0 / k, (j - 1) / k
                length = (t1 - t0) * L
                if length >= min_open_ft * pt_ft:
                    a = [round(ax + ux * L * t0, 2), round(ay + uy * L * t0, 2)]
                    b = [round(ax + ux * L * t1, 2), round(ay + uy * L * t1, 2)]
                    out.append({"a": a, "b": b, "length_ft": round(length / pt_ft, 2), "room": r.get("name")})
    clear_cache()
    out.sort(key=lambda t: -t["length_ft"])
    return out
