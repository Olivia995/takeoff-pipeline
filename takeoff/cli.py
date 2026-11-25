"""Command line: detect rooms -> choose rooms -> build the takeoff sheet.

    python -m takeoff detect PLAN.pdf -o work/
    python -m takeoff select work/rooms.json --names FITNESS,YOGA
    python -m takeoff build work/rooms.json --system FIT70 --project "..." --dwg ASH-1 -o ASH-1.pdf
    python -m takeoff run PLAN.pdf --names FITNESS --system FIT70 --project "..." --dwg ASH-1
"""
import argparse
import math
import csv
import json
import os
import sys
from pathlib import Path

import pymupdf
import yaml

from .detect import polygon_area, room_area_pt
from .rooms import find_rooms
from . import engine
from .scale import find_plan_scale, parse_scale, pt_per_ft, scale_from_dimensions
from .sheet import render_preview, render_sheet

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = ROOT / "config" / "products.yaml"
DEFAULT_COMPANY = ROOT / "config" / "company.yaml"


# ---------------------------------------------------------------- helpers
def _open_page(pdf, page_no):
    doc = pymupdf.open(pdf)
    if not 1 <= page_no <= len(doc):
        sys.exit(f"{pdf} has {len(doc)} page(s); --page {page_no} is out of range.")
    page = doc[page_no - 1]
    if page.rotation:
        page.set_rotation(0)   # work in unrotated page space throughout
    return doc, page


def _save(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _print_rooms(data):
    print(f"\n  {'#':>3}  {'incl':4}  {'area SF':>8}  name")
    for r in data["rooms"]:
        area = f"{r['area_sf']:.0f}" if r.get("area_sf") is not None else "-"
        print(f"  {r['id']:>3}  {'yes' if r.get('included') else '':4}  {area:>8}  {r['name']}")
    inc = [r for r in data["rooms"] if r.get("included")]
    if inc and all(r.get("area_sf") is not None for r in inc):
        print(f"\n  {len(inc)} included, {sum(r['area_sf'] for r in inc):.0f} SF net")


def _barriers(path):
    if not path:
        return []
    data = _load(path)
    return data.get("barriers_pt", []) if isinstance(data, dict) else data


# ---------------------------------------------------------------- commands
def cmd_detect(a):
    doc, page = _open_page(a.plan, a.page)
    if a.scale:
        ipf, label = parse_scale(a.scale), a.scale
        if not ipf:
            sys.exit(f"Couldn't read the scale '{a.scale}'. Try 1/8 or '1/8\" = 1'-0\"'.")
    else:
        ipf, label = find_plan_scale(page)
        if not ipf:
            ipf, label = scale_from_dimensions(page)
            if ipf:
                label += "  (estimated from dimension strings - check it)"
        if not ipf:
            sys.exit("Couldn't find the plan scale on the sheet. Pass it with --scale, e.g. --scale 1/8")
    ppf = pt_per_ft(ipf)
    print(f"Scale: {label}  ({ppf:.2f} pt per ft)")

    rooms = find_rooms(page, ppf, threshold=a.threshold, thin=a.thin, gap_ft=a.gap_ft,
                       min_sf=a.min_sf, barriers=_barriers(a.barriers))
    for i, r in enumerate(rooms, 1):
        r["id"] = i
        r["included"] = False
    data = {"source": str(Path(a.plan).resolve()), "page": a.page,
            "scale": {"label": label, "in_per_ft": ipf, "pt_per_ft": ppf},
            "rooms": [{k: r.get(k) for k in ("id", "name", "number", "included", "area_sf", "outline_pt", "holes_pt", "system")} for r in rooms]}
    out = Path(a.out)
    rooms_json = out / "rooms.json" if out.suffix != ".json" else out
    _save(data, rooms_json)
    preview = rooms_json.with_name(rooms_json.stem + "_preview.pdf")
    render_preview(str(preview), doc, a.page, data["rooms"])
    _print_rooms(data)
    print(f"\nWrote {rooms_json}\nWrote {preview}  (open it to see which number is which room)")


def cmd_select(a):
    data = _load(a.rooms)
    _apply_selection(data, a)
    _save(data, a.rooms)
    _print_rooms(data)


def _apply_selection(data, a):
    rooms = data["rooms"]
    if getattr(a, "none", False):
        for r in rooms:
            r["included"] = False
    if getattr(a, "all", False):
        for r in rooms:
            r["included"] = True
    if getattr(a, "ids", None):
        want = {int(x) for x in a.ids.split(",") if x.strip()}
        for r in rooms:
            r["included"] = r["id"] in want
    if getattr(a, "names", None):
        keys = [k.strip().upper() for k in a.names.split(",") if k.strip()]
        for r in rooms:
            r["included"] = any(k in (r["name"] or "").upper() for k in keys)
    if getattr(a, "interactive", False):
        _print_rooms(data)
        ans = input("\nRoom numbers to include (e.g. 3,5,10): ").strip()
        want = {int(x) for x in ans.replace(" ", "").split(",") if x}
        for r in rooms:
            r["included"] = r["id"] in want


def cmd_build(a):
    data = _load(a.rooms)
    _apply_selection(data, a)
    inc = [r for r in data["rooms"] if r.get("included")]
    if not inc:
        sys.exit("No rooms are included. Use `select`, --ids, --names, or the review page first.")
    ppf = data["scale"]["pt_per_ft"]
    default = [{"code": c.strip()} for c in (a.buildup or "").split("/") if c.strip()]
    for r in inc:   # recompute from the outline in case it was edited in the review page
        rings = [r["outline_pt"]] + list(r.get("holes_pt") or [])
        r["area_sf"] = room_area_pt(r) / ppf ** 2
        r["perimeter_lf"] = sum(math.dist(g[i - 1], g[i]) for g in rings for i in range(len(g))) / ppf
        r["_ppf"] = ppf
        if not r.get("layers"):
            if not default:
                sys.exit(f"{r.get('name')} has no buildup. Pass --buildup, e.g. --buildup FIT70/FFNC50NBN (top first).")
            r["layers"] = default
    net = sum(r["area_sf"] for r in inc)

    catalog = engine.load()
    company = yaml.safe_load(Path(a.company).read_text(encoding="utf-8"))
    try:
        t = engine.compute(catalog, inc, a.note or [])
    except engine.TakeoffError as e:
        sys.exit(str(e))

    print(f"\n{a.project}  ({len(inc)} rooms, {net:.0f} SF net)\n")
    for ln in t.lines:
        print("  " + ln.text())
    for w in t.warnings:
        print("  ! " + w)

    out = Path(a.out or f"{a.dwg or 'takeoff'}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    source = a.plan or data["source"]
    if not os.path.exists(source):
        sys.exit(f"Plan PDF not found at {source}. Pass it with --plan.")
    plan_doc = pymupdf.open(source)
    if plan_doc[data["page"] - 1].rotation:
        plan_doc[data["page"] - 1].set_rotation(0)
    meta = {"project": a.project, "dwg_no": a.dwg or "", "drawn_by": a.drawn_by or "",
            "date": a.date, "pt_per_ft": ppf}
    render_sheet(str(out), plan_doc, data["page"], inc, t, company, meta, catalog)

    bom_csv = out.with_suffix(".csv")
    write_bom_csv(bom_csv, t, inc)
    print(f"\nWrote {out}\nWrote {bom_csv}")


def write_bom_csv(path, t, rooms):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "item", "quantity", "measure", "units", "unit", "overage_pct", "line"])
        for ln in t.lines:
            w.writerow([ln.kind, ln.item, ln.quantity, "pails" if ln.kind == "adhesive" else ln.measure,
                        ln.units or "", ln.unit, ln.overage_pct if ln.kind == "product" else "", ln.text()])
        w.writerow([])
        w.writerow(["room", "buildup (top to bottom)", "area_sf", "perimeter_lf", "columns", "columns_sf", "transitions_sf"])
        for r in rooms:
            holes = r.get("holes_pt") or []
            col_sf = sum(polygon_area(h) for h in holes) / r["_ppf"] ** 2 if r.get("_ppf") else ""
            w.writerow([r["name"], " over ".join(l["code"] for l in r.get("layers") or []), round(r["area_sf"], 1),
                        round(r.get("perimeter_lf", 0), 1), len(holes), round(col_sf, 1) if col_sf != "" else "",
                        round(r.get("ramps_sf", 0), 1)])
        if getattr(t, "warnings", None):
            w.writerow([])
            w.writerow(["warning"])
            for x in t.warnings:
                w.writerow([x])


def cmd_run(a):
    work = Path(a.work)
    a.out_detect = work
    det = argparse.Namespace(plan=a.plan, page=a.page, scale=a.scale, threshold=a.threshold, thin=a.thin,
                             gap_ft=a.gap_ft, min_sf=a.min_sf, barriers=a.barriers, out=str(work))
    cmd_detect(det)
    a.rooms = str(work / "rooms.json")
    if not (a.ids or a.names or a.all or a.interactive):
        a.interactive = True
    cmd_build(a)


def cmd_products(a):
    for p in engine.selectable(engine.load()):
        size = f"{p['unit_size']:g} {'LF' if p['unit'] == 'lf_roll' else 'SF'}/{p['unit']}" if p["unit_size"] else "not supplied"
        print(f"  {p['code']:20} {p['display']:28} {size}")


# ---------------------------------------------------------------- parser
def main(argv=None):
    p = argparse.ArgumentParser(prog="takeoff", description="Plan room detection and GenieMat takeoffs")
    sub = p.add_subparsers(dest="cmd", required=True)

    def detect_opts(sp):
        sp.add_argument("--page", type=int, default=1)
        sp.add_argument("--scale", help="e.g. 1/8 or '1/8\" = 1'-0\"' (read from the sheet if omitted)")
        sp.add_argument("--threshold", type=int, default=200, help="wall line darkness, 0-255 (default 200)")
        sp.add_argument("--thin", type=int, default=1, help="ignore lines thinner than 2*thin+1 px (default 1)")
        sp.add_argument("--gap-ft", type=float, default=3.75, help="bridge openings up to this width in feet")
        sp.add_argument("--min-sf", type=float, default=30, help="smallest room to keep (SF)")
        sp.add_argument("--barriers", help="JSON list of [[x0,y0],[x1,y1]] segments (PDF points) to close doorways")

    def select_opts(sp):
        sp.add_argument("--ids", help="comma-separated room numbers to include")
        sp.add_argument("--names", help="include rooms whose name contains any of these, e.g. FITNESS,YOGA")
        sp.add_argument("--all", action="store_true")
        sp.add_argument("--none", action="store_true")
        sp.add_argument("--interactive", "-i", action="store_true", help="choose rooms at a prompt")

    def build_opts(sp):
        sp.add_argument("--buildup", help="layers for rooms without one, TOP FIRST, e.g. FIT70/FFNC50NBN (see `products`)")
        sp.add_argument("--note", action="append", help="add a manual note by id from notes.yaml (repeatable)")
        sp.add_argument("--project", required=True)
        sp.add_argument("--dwg", help="drawing number, e.g. ASH-1")
        sp.add_argument("--drawn-by")
        sp.add_argument("--date", help="YYYY-MM-DD (default today)")
        sp.add_argument("--standard-lf", type=float, default=0, help="LF of standard transition")
        sp.add_argument("--ada-lf", type=float, default=0, help="LF of ADA transition")
        sp.add_argument("--plan", help="plan PDF (default: the one recorded in rooms.json)")
        sp.add_argument("--company", default=str(DEFAULT_COMPANY))
        sp.add_argument("-o", "--out", help="output PDF (a .csv BOM is written beside it)")

    sp = sub.add_parser("detect", help="find rooms on a plan and write rooms.json + a numbered preview")
    sp.add_argument("plan")
    detect_opts(sp)
    sp.add_argument("-o", "--out", default="work", help="output folder or .json path")
    sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("select", help="choose which rooms go into the takeoff")
    sp.add_argument("rooms")
    select_opts(sp)
    sp.set_defaults(func=cmd_select)

    sp = sub.add_parser("build", help="build the takeoff sheet PDF and BOM CSV")
    sp.add_argument("rooms")
    select_opts(sp)
    build_opts(sp)
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("run", help="detect, select and build in one go")
    sp.add_argument("plan")
    detect_opts(sp)
    select_opts(sp)
    build_opts(sp)
    sp.add_argument("--work", default="work", help="folder for rooms.json and the preview")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("serve", help="open the takeoff app in your browser")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--no-browser", action="store_true")
    sp.set_defaults(func=lambda a: __import__("takeoff.server", fromlist=["serve"]).serve(a.port, not a.no_browser))

    sp = sub.add_parser("products", help="list the products buildups can use")
    sp.set_defaults(func=cmd_products)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
