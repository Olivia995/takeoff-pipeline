"""Find and parse drawing scales such as 1/8" = 1'-0"."""
import re
from .pdftext import text_dict, text_words

_SCALE_RE = re.compile(
    r"""(?P<whole>\d+(?:\.\d+)?)?\s*(?:(?P<num>\d+)\s*/\s*(?P<den>\d+))?\s*["”″]\s*(?:=|\s)\s*1\s*['’′]\s*-?\s*0\s*["”″]?"""
)


_RATIO_RE = re.compile(r"^\s*(?:SCALE\s*:?\s*)?1\s*:\s*(\d{1,4})\s*$", re.I)
METRIC_RATIOS = {10, 20, 25, 30, 50, 75, 100, 125, 150, 200}   # 1:250+ are key and site plans


def parse_scale(text):
    """Return paper inches per real foot for a scale string, or None.

    Accepts '1/8" = 1\'-0"', '1 1/2" = 1\'-0"', '3/16', '0.125', '1/8'.
    """
    t = text.strip()
    m = _SCALE_RE.search(t)
    if m and (m.group("whole") or m.group("num")):
        val = float(m.group("whole") or 0)
        if m.group("num"):
            val += int(m.group("num")) / int(m.group("den"))
        return val or None
    m = re.fullmatch(r"(?:(\d+)\s+)?(\d+)\s*/\s*(\d+)", t)
    if m:
        return (int(m.group(1) or 0)) + int(m.group(2)) / int(m.group(3))
    try:
        v = float(t)
        return v if v > 0 else None
    except ValueError:
        return None


def pt_per_ft(in_per_ft):
    """PDF points per real foot (72 pt = 1 paper inch)."""
    return 72.0 * in_per_ft


def find_plan_scale(page):
    """Guess the scale of the main plan on a page from its text layer.

    Prefers a scale string that sits next to a view title containing 'PLAN'.
    Returns (in_per_ft, label) or (None, None).
    """
    lines = []
    for b in text_dict(page)["blocks"]:
        for l in b.get("lines", []):
            txt = "".join(s["text"] for s in l["spans"]).strip()
            if txt:
                size = max(s["size"] for s in l["spans"])
                lines.append((l["bbox"], txt, size))
    scales = []
    for bbox, txt, _ in lines:
        v = parse_scale(txt) if ("=" in txt or _SCALE_RE.search(txt)) else None
        if not v:
            m = _RATIO_RE.match(txt)          # metric sheets: 1:50, SCALE: 1 : 100
            if m and int(m.group(1)) in METRIC_RATIOS:
                v, txt = 12 / int(m.group(1)), f"1:{int(m.group(1))}"
        if v:
            scales.append((bbox, txt, v))
    if not scales:
        return None, None

    def near_plan_title(bbox):
        best = 0.0
        for b2, t2, size in lines:
            if "PLAN" not in t2.upper() or "=" in t2:
                continue
            dy = min(abs(b2[1] - bbox[3]), abs(bbox[1] - b2[3]))
            overlap_x = min(bbox[2], b2[2]) - max(bbox[0], b2[0])
            if dy < 60 and overlap_x > -20:
                best = max(best, size)
        return best

    scored = sorted(scales, key=lambda s: near_plan_title(s[0]), reverse=True)
    bbox, txt, v = scored[0]
    if near_plan_title(bbox) == 0 and len({s[2] for s in scales}) > 1:
        return None, None  # several scales and none tied to a plan title: ask the user
    return v, txt.strip()


STANDARD_SCALES = [  # paper inches per foot
    (1 / 16, '1/16" = 1\'-0"'), (3 / 32, '3/32" = 1\'-0"'), (1 / 8, '1/8" = 1\'-0"'),
    (3 / 16, '3/16" = 1\'-0"'), (1 / 4, '1/4" = 1\'-0"'), (3 / 8, '3/8" = 1\'-0"'),
    (1 / 2, '1/2" = 1\'-0"'), (3 / 4, '3/4" = 1\'-0"'), (1.0, '1" = 1\'-0"'),
    (1.5, '1 1/2" = 1\'-0"'), (3.0, '3" = 1\'-0"'),
]
_DIM = re.compile(r"""^(\d+)\s*['’′]\s*-?\s*(\d+(?:\s+\d+/\d+)?|\d+/\d+)?\s*["”″]?$""")


def _dim_feet(text):
    m = _DIM.match(text.strip())
    if not m:
        return None
    inch = 0.0
    for q in (m.group(2) or "").split():
        if "/" in q:
            a, b = q.split("/")
            inch += int(a) / int(b)
        else:
            inch += int(q)
    return int(m.group(1)) + inch / 12


def scale_from_dimensions(page):
    """Estimate the plan scale from dimension strings (e.g. 7'-0") and the dimension lines
    they sit on, for sheets whose scale note isn't stored as text.
    Returns (in_per_ft, label) snapped to a standard architectural scale, or (None, None)."""
    import math
    import statistics
    segs = [(it[1], it[2]) for d in page.get_drawings() for it in d["items"] if it[0] == "l"]
    est = []
    for b in text_dict(page)["blocks"]:
        for l in b.get("lines", []):
            ft = _dim_feet("".join(s["text"] for s in l["spans"]))
            if not ft or ft < 1:
                continue
            dx, dy = l["dir"]
            x0, y0, x1, y1 = l["bbox"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            h = abs((x1 - x0) * dy) + abs((y1 - y0) * dx) or 8
            best = None
            for a, c in segs:
                vx, vy = c.x - a.x, c.y - a.y
                L = math.hypot(vx, vy)
                if L < 5 or abs(vx * dy - vy * dx) > 0.02 * L:
                    continue                      # dimension line runs along the text
                t = ((cx - a.x) * vx + (cy - a.y) * vy) / L ** 2
                off = abs((cx - a.x) * vy - (cy - a.y) * vx) / L
                if 0.2 < t < 0.8 and off < 2 * h and (best is None or off < best[0]):
                    best = (off, L)
            if best:
                est.append(best[1] / ft)          # PDF points per foot
    if len(est) < 3:
        return None, None
    med = statistics.median(est)
    if sum(abs(e / med - 1) < 0.2 for e in est) < 0.6 * len(est):
        return None, None                         # dimensions disagree: don't guess
    ipf, label = min(STANDARD_SCALES, key=lambda s: abs(math.log(72 * s[0] / med)))
    if abs(math.log(72 * ipf / med)) > 0.25:
        return None, None
    return ipf, label
