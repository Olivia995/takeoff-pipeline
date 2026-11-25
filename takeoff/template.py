"""The takeoff sheet's template: border, title block, logo and fixed text, taken from
Pliteq's AutoCAD template (DXF) so generated sheets match the drafted ones.

    python -m takeoff.template "Template Takeoff 2026.dxf"     # writes config/sheet_template.json

The JSON holds the template's paper-space line work (in PDF points on an 11 x 8.5 in
page), the logo fills, the fixed text, and where each filled-in field goes (project,
drawn by, date, DWG no., reviewed by, the BOM, the notes, the drawing area). render_sheet
draws it with draw(); ezdxf is only needed to convert a new template.
"""
import json
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_JSON = ROOT / "config" / "sheet_template.json"
CAP = 0.70             # Georgia's capital height as a share of the font size (AutoCAD heights are cap heights)
LINE_PITCH = 5 / 3     # AutoCAD MTEXT line spacing, x text height
GEORGIA = [Path("/System/Library/Fonts/Supplemental/Georgia.ttf"), Path("/Library/Fonts/Georgia.ttf"),
           Path("C:/Windows/Fonts/georgia.ttf"), ROOT / "config" / "fonts" / "Georgia.ttf"]

# placeholder text in the template -> the field that replaces it
FIELDS = {"Project Name": "project", "ET": "drawn_by", "2026-01-01": "date", "XX - 1": "dwg_no", "XX": "reviewed_by"}


# ---------------------------------------------------------------- convert (needs ezdxf)
def convert(dxf_path, out=TEMPLATE_JSON, layout="Layout"):
    import ezdxf
    from ezdxf import path as ezpath

    doc = ezdxf.readfile(str(dxf_path))
    lay = doc.layouts.get(layout)
    lo, hi = lay.dxf.limmin, lay.dxf.limmax
    page_w, page_h = (hi[0] - lo[0]) * 72, (hi[1] - lo[1]) * 72

    def P(x, y):
        return [round((x - lo[0]) * 72, 2), round((hi[1] - y) * 72, 2)]

    def lw(e):
        w = e.dxf.get("lineweight", -1)
        if w is None or w < 0:
            layer = doc.layers.get(e.dxf.layer)
            w = layer.dxf.get("lineweight", 25) if layer else 25
        if w is None or w < 0:
            w = 25
        return round(w / 100 / 25.4 * 72, 2)            # 1/100 mm -> pt

    strokes, fills, texts, frame, draw_area = [], [], [], None, None
    callout_size = None
    for e in lay:
        t = e.dxftype()
        if t == "LINE":
            strokes.append({"pts": [P(e.dxf.start.x, e.dxf.start.y), P(e.dxf.end.x, e.dxf.end.y)], "w": lw(e)})
        elif t in ("ARC", "LWPOLYLINE"):
            p = ezpath.make_path(e)
            pts = [P(v.x, v.y) for v in p.flattening(0.002)]
            item = {"pts": pts, "w": lw(e), "closed": bool(getattr(e, "closed", False))}
            strokes.append(item)
            if t == "LWPOLYLINE" and lw(e) > 1.5:
                xs, ys = [q[0] for q in pts], [q[1] for q in pts]
                frame = [min(xs), min(ys), max(xs), max(ys)]
        elif t == "HATCH":
            rgb = e.rgb or (0, 0, 0)
            rings = []
            for p in ezpath.from_hatch(e):
                rings.append([P(v.x, v.y) for v in p.flattening(0.002)])
            fills.append({"rings": rings, "rgb": [round(c / 255, 3) for c in rgb]})
        elif t == "VIEWPORT" and e.dxf.get("id", 0) != 1 and e.dxf.width < 20:
            cx, cy, w, h = e.dxf.center[0], e.dxf.center[1], e.dxf.width, e.dxf.height
            a, b = P(cx - w / 2, cy + h / 2), P(cx + w / 2, cy - h / 2)
            draw_area = [a[0], a[1], b[0], b[1]]
        elif t == "MULTILEADER" and callout_size is None:
            callout_size = round(e.context.char_height * 72 / CAP, 2)      # the buildup callouts
        elif t == "MTEXT":
            raw = e.text
            plain = e.plain_text()
            for sup in _superscripts(raw):
                plain = plain.replace(sup + "^", sup)
            m = re.match(r"\{?\\H([\d.]+)x;", raw)
            factor = float(m.group(1)) if m else 1.0
            size = e.dxf.char_height * factor * 72 / CAP
            x, y = P(e.dxf.insert.x, e.dxf.insert.y)
            item = {"text": plain, "x": x, "y": y, "size": round(size, 2), "width": round(e.dxf.get("width", 0) * 72, 1),
                    "attach": e.dxf.get("attachment_point", 1), "sup": _superscripts(raw)}
            texts.append(item)

    # which text is which: placeholders become fields; the BOM and notes samples become areas
    fields, static = {}, []
    for t in texts:
        s = t["text"].strip()
        if s in FIELDS:
            fields[FIELDS[s]] = t
        elif re.match(r"^[\d.]+\s+(SF|LF)\b", s):
            fields["bom"] = t
        elif s.startswith("ADA transition") or "transition placements" in s:
            fields["notes"] = t
        elif s == "®" and t["size"] < 4:
            static.append(dict(t, logo_mark=True))
        else:
            static.append(t)
    tpl = {"source": Path(dxf_path).name, "page": [round(page_w, 2), round(page_h, 2)], "frame": frame,
           "draw_area": draw_area, "callout_size": callout_size, "strokes": strokes, "fills": fills, "texts": static, "fields": fields}
    Path(out).write_text(json.dumps(tpl, indent=1))
    return tpl


def _superscripts(raw):
    """Characters AutoCAD stacks as superscript (\\S®^ ;): drawn small and raised."""
    return re.findall(r"\\S([^;^]+)\^ ?;", raw)


# ---------------------------------------------------------------- draw
def load(path=TEMPLATE_JSON):
    return json.loads(Path(path).read_text())


class Fonts:
    """Georgia when the machine has it (it's what the template uses), else Times."""

    def __init__(self, page):
        self.page = page
        self.file = next((str(p) for p in GEORGIA if p.exists()), None)
        if self.file:
            page.insert_font(fontname="georgia", fontfile=self.file)
            self.name, self.font = "georgia", pymupdf.Font(fontfile=self.file)
        else:
            self.name, self.font = "tiro", pymupdf.Font("tiro")

    def width(self, text, size):
        return self.font.text_length(text, fontsize=size)

    def wrap(self, text, size, width):
        out = []
        for para in text.split("\n"):
            words, line = para.split(" "), ""
            for w in words:
                cand = (line + " " + w).strip() if line else w
                if width and self.width(cand, size) > width and line:
                    out.append(line)
                    line = w
                else:
                    line = cand
            out.append(line)
        return out


def text_block(page, fonts, t, text=None, size=None, width=None, max_bottom=None):
    """Draw MTEXT-style text: top-left at (x, y), cap height = AutoCAD height, wrapped to width.
    Returns the y of the last baseline."""
    size = size or t["size"]
    text = t["text"] if text is None else text
    width = t.get("width") if width is None else width
    lines = fonts.wrap(text, size, width) if width else text.split("\n")
    pitch = size * CAP * LINE_PITCH
    y = t["y"] + size * CAP
    sups = set(t.get("sup") or [])
    for line in lines:
        x = t["x"]
        if t.get("attach") in (2, 5, 8):
            x -= fonts.width(line, size) / 2
        elif t.get("attach") in (3, 6, 9):
            x -= fonts.width(line, size)
        _line(page, fonts, x, y, line, size, sups)
        y += pitch
    return y - pitch


def _line(page, fonts, x, y, line, size, sups):
    if not sups or not any(s in line for s in sups):
        page.insert_text((x, y), line, fontname=fonts.name, fontsize=size)
        return
    pat = "(" + "|".join(re.escape(s) for s in sups) + ")"
    for part in re.split(pat, line):
        if not part:
            continue
        if part in sups:
            page.insert_text((x, y - size * 0.4), part, fontname=fonts.name, fontsize=size * 0.6)
            x += fonts.width(part, size * 0.6)
        else:
            page.insert_text((x, y), part, fontname=fonts.name, fontsize=size)
            x += fonts.width(part, size)


def draw(page, tpl, values, fonts=None):
    """The template's border, title block, logo, fixed text and the filled-in fields."""
    fonts = fonts or Fonts(page)
    for f in tpl["fills"]:
        sh = page.new_shape()
        for ring in f["rings"]:
            if len(ring) >= 3:
                sh.draw_polyline([pymupdf.Point(*p) for p in ring] + [pymupdf.Point(*ring[0])])
        sh.finish(color=None, fill=tuple(f["rgb"]), even_odd=True, closePath=True)
        sh.commit()
    for s in tpl["strokes"]:
        pts = [pymupdf.Point(*p) for p in s["pts"]]
        if s.get("closed"):
            pts.append(pts[0])
        page.draw_polyline(pts, color=(0, 0, 0), width=s["w"], lineCap=1, lineJoin=1)
    for t in tpl["texts"]:
        if t.get("logo_mark"):
            page.insert_text((t["x"] - 1, t["y"] + 1), t["text"], fontname="helv", fontsize=3.2,
                             color=(0.48, 0.49, 0.5))
            continue
        text_block(page, fonts, t)
    f = tpl["fields"]
    for key in ("project", "drawn_by", "date", "dwg_no", "reviewed_by"):
        if key in f and values.get(key) is not None:
            text_block(page, fonts, f[key], text=str(values[key]), width=0)
    return fonts


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    tpl = convert(argv[0], argv[1] if len(argv) > 1 else TEMPLATE_JSON)
    print(f"Template saved: {len(tpl['strokes'])} lines, {len(tpl['fills'])} fills, {len(tpl['texts'])} texts, "
          f"fields {sorted(tpl['fields'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
