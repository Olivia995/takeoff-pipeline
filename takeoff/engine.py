"""Takeoff engine: rooms with layer-by-layer buildups -> BOM lines, adhesive and notes.

Driven entirely by config/products.yaml, config/rules.yaml and config/notes.yaml.

A room's buildup is a list of layers, TOP FIRST: the first layer is the walking
surface, the last sits on the slab. Each layer is {"code": "FIT70"} with an optional
"bond": "glued" | "loose" that overrides the product's usual bond.

Agreed with Pliteq (Nov 2026):
- Overage comes from each product's TOTAL quantity across the whole takeoff
  (rules.yaml overage_tiers; a value exactly on a boundary gets the higher overage).
- Adhesive is worked out per room and layer on the ORDERED area (net x overage):
  FAS3 for plank-class products; FASHM for a rolled layer glued onto FIT tiles;
  FAS1 for everything else (FAS1 pails = FASHM pails x 0.3). Pails are summed per
  adhesive across the takeoff, then rounded up.
- PMI12PF (perimeter isolation) is the room perimeter in LF.
- Transitions (ramps) when the slab isn't recessed: see ramps.py.
"""
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import ramps as ramps_mod
from .rules import BomLine

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"

ADHESIVES = {"FAS1": "Pliteq FAS1 Adhesive", "FASHM": "Pliteq FASHM Adhesive", "FAS3": "Pliteq FAS3 Adhesive"}
TILE_FAMILIES = ("fit_tile", "xfit")
FF_FAMILIES = ("ff", "ff_panel")


PMI_CODE = "PMI12PF"
FF70_CODES = ("FF70", "FF70LDM", "FF70-LF")


def expand(cat, layers):
    """A buildup's layers as installed: systems (FIT62, FIT55, PeakForce...) become their
    component layers (products.yaml "components", top first), and perimeter isolation is
    dropped (it's added automatically, see needs_pmi)."""
    P = cat["products"]
    out = []
    for l in layers:
        p = P[l["code"]]
        if p["family"] == "pmi":
            continue
        if p.get("components"):
            out.extend(dict({"code": c, "_sys": p["code"]}, **({"colour": l["colour"]} if l.get("colour") else {}))
                       for c in p["components"])
        else:
            out.append(l)
    return out


def needs_pmi(cat, layers):
    """Perimeter isolation goes around rooms whose buildup has plywood, FF under concrete,
    or FF70 (modular panels)."""
    P = cat["products"]
    fams = [P[l["code"]]["family"] for l in layers]
    if "plywood" in fams or any(l["code"] in FF70_CODES for l in layers):
        return True
    return any(f == "concrete" and "ff" in fams[i + 1:] for i, f in enumerate(fams))


def _c1(p, i):
    """A FIT tile with another layer on top of it is ordered as the -C1 tile (FIT70-C1)."""
    return i > 0 and p["family"] == "fit_tile"


def _colour(l):
    c = (l or {}).get("colour")
    return str(c).strip() if c else ""


def _key(p, i, l=None):
    """BOM key: one line per product, -C1 tile and colour (FIT70 Colour A and B are ordered
    separately, each with its own overage: 215 Lakeshore, 49 Ontario, Hillsboro)."""
    return p["code"] + ("-C1" if _c1(p, i) else "") + (f"|{_colour(l)}" if _colour(l) else "")


def _name(p, i, l=None):
    return p["display"] + ("-C1" if _c1(p, i) else "") + (f" (Colour {_colour(l)})" if _colour(l) else "")


def turf_order(rules, room, pt_per_ft=None):
    """VersaTURF ordered SF for one room (Pliteq: "the length, multiplied by the nearest 15 feet
    width"): strips run along the room's length, the width rounds up to a multiple of 15',
    and the length rounds up to the next foot (rolls are at most 100' long; a longer strip is two rolls
    end to end, same SF). Uses the room's minimum bounding rectangle; without an outline,
    a square of the same area. Returns (ordered SF, had_outline)."""
    T = rules.get("turf") or {"roll_width_ft": 15}
    wft = T["roll_width_ft"]
    ppf = room.get("_ppf") or pt_per_ft
    dims = None
    if room.get("outline_pt") and ppf:
        import cv2
        import numpy as np
        pts = np.array([[x / ppf, y / ppf] for x, y in room["outline_pt"]], np.float32)
        (_, _), (a, b), _ = cv2.minAreaRect(pts)
        dims = (a, b)
    if not dims:
        side = math.sqrt(room["area_sf"])
        dims = (side, side)
    W, L = sorted(dims)                          # strips run along the room's length
    return wft * math.ceil(W / wft - 1e-6) * math.ceil(L - 1e-6), bool(room.get("outline_pt"))


def _pails(v):
    """Pails quoted in notes: rounded to the nearest tenth (the BOM rounds up to whole pails)."""
    return f"{math.floor(v * 10 + 0.5 + 1e-9) / 10:.1f}"


def _glueless(below):
    """FIT tiles sit on GenieMat FFxxNBN without glue; on FFNCxxNBN they are still glued."""
    return below is not None and below.get("nbn") and not below["code"].startswith("FFNC")


def _recess_mm(T):
    """Recess depth in mm (the sheets give it in mm; older saves have recess_in)."""
    if T.get("recess_mm"):
        return float(T["recess_mm"])
    return float(T.get("recess_in") or 0) * 25.4


class TakeoffError(Exception):
    pass


@dataclass
class Result:
    lines: list = field(default_factory=list)        # BomLine
    notes: list = field(default_factory=list)        # note text, in notes.yaml order
    note_ids: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    groups: list = field(default_factory=list)       # one per distinct buildup
    pails_raw: dict = field(default_factory=dict)
    net_sf: float = 0.0
    ramps: dict = None                               # ramps.plan() result, when there are transitions


def load(config_dir=CONFIG):
    d = Path(config_dir)
    products = yaml.safe_load((d / "products.yaml").read_text(encoding="utf-8"))["products"]
    rules = yaml.safe_load((d / "rules.yaml").read_text(encoding="utf-8"))
    notes = yaml.safe_load((d / "notes.yaml").read_text(encoding="utf-8"))["notes"]
    by_code = {}
    for i, p in enumerate(products):
        p = dict(p)
        p["_order"] = i
        p.setdefault("in_bom", True)
        p.setdefault("nbn", False)
        by_code[p["code"]] = p
    return {"products": by_code, "rules": rules, "notes": notes}


FAMILY_LABELS = {"xfit": "xFIT", "fit_tile": "FIT tiles", "fit_roll": "FIT rolls", "flooring": "Flooring",
                 "rst": "RST / interstitial", "dh": "DH", "ff": "FF (rolled)", "ff_panel": "FF (panels)",
                 "other": "Other", "pmi": "Perimeter isolation", "plywood": "Not supplied by Pliteq",
                 "system": "Systems", "concrete": "Not supplied by Pliteq",
                 "finished_floor": "Not supplied by Pliteq"}


def _ready(cat, p):
    if p.get("components"):
        return all(_ready(cat, cat["products"][c]) for c in p["components"])
    return p.get("unit_size") is not None or not p.get("in_bom", True) or p.get("count") in ("sf", "turf")


def selectable(cat):
    """Products offered in the buildup dropdowns, in catalog order: everything with a known
    unit size (systems: all their components), plus plywood, concrete and finished floor
    (never quantified). Perimeter isolation isn't offered: it's added automatically."""
    out = []
    for p in sorted(cat["products"].values(), key=lambda p: p["_order"]):
        if p["family"] == "pmi" or not _ready(cat, p):
            continue
        out.append({"code": p["code"], "display": p["display"], "family": p["family"],
                    "group": FAMILY_LABELS.get(p["family"], p["family"]), "unit": p["unit"],
                    "unit_size": p.get("unit_size"), "thickness_mm": p.get("thickness_mm"),
                    "bond": p.get("bond", "glued"), "in_bom": p.get("in_bom", True),
                    "components": p.get("components")})
    return out


def overage_pct(rules, total):
    for t in rules["overage_tiers"]:
        if t["max"] is None or total <= t["max"] + 1e-9:
            return t["pct"]
    return rules["overage_tiers"][-1]["pct"]


def _num(v):
    return round(v, 1) if abs(v - round(v)) > 0.05 else int(round(v))


def callout(cat, layers):
    """Buildup label, top first: "A over\nB over\nC". Perimeter isolation runs around the
    edge rather than under the floor, so it's left out; repeated layers are written "2 x ..."."""
    names = []
    for i, l in enumerate(layers):
        p = cat["products"][l["code"]]
        if p["family"] == "pmi":
            continue
        if p["family"] == "finished_floor":
            n = p["display"] + " (supplied by others)"
        elif p["family"] in ("plywood", "concrete"):
            n = p["display"] + " (supplied by others)"
        else:
            n = _name(p, i, l)
        if names and names[-1][1] == n:
            names[-1][0] += 1
        else:
            names.append([1, n])
    names = [(f"{k}x {n}" if k > 1 else n) for k, n in names]
    return "\n".join(n + (" over" if i < len(names) - 1 else "") for i, n in enumerate(names))


def _poly_area(p):
    return abs(sum(p[i - 1][0] * p[i][1] - p[i][0] * p[i - 1][1] for i in range(len(p)))) / 2


def compute(cat, rooms, manual_notes=(), transitions=None, pt_per_ft=None):
    """rooms: [{name, area_sf, perimeter_lf, layers:[{code, bond?}], finish_adhesive_by_pliteq?, outline_pt?}]
    transitions: {slab: "recessed" (default, no transitions) | "raised" | "partial" (recessed
    by recess_mm but ramps still needed), recess_mm (recess_in still read), no_transitions_date, no_ada_date,
    openings: [see ramps.plan]}; pt_per_ft converts the openings' PDF points to feet."""
    P, R = cat["products"], cat["rules"]
    res = Result()
    for r in rooms:
        if not r.get("layers"):
            raise TakeoffError(f"{r.get('name') or 'A room'} has no buildup yet.")
        for l in r["layers"]:
            if l.get("code") not in P:
                raise TakeoffError(f"Unknown product '{l.get('code')}' in {r.get('name')}.")
    res.net_sf = sum(r["area_sf"] for r in rooms)
    for r in rooms:
        r["_calc"] = expand(cat, r["layers"])
        if not r["_calc"]:
            raise TakeoffError(f"{r.get('name') or 'A room'} has no buildup yet.")
    T = transitions or {}
    slab = T.get("slab") or "recessed"
    ramped = slab in ("raised", "partial") and not T.get("no_transitions_date")
    if ramped:
        res.ramps = ramps_mod.plan(cat, rooms, T.get("openings") or [],
                                   {"recess_mm": _recess_mm(T) if slab == "partial" else 0,
                                    "no_ada": bool(T.get("no_ada_date"))}, pt_per_ft)
        res.warnings += res.ramps["warnings"]
        # The ramps (and any FIT08 landing) sit inside a room drawn to its walls, where the
        # buildup isn't, so their footprint comes off the room's SF (Lifetime Sky LFS-3:
        # 3716 SF = the rooms less 227 SF of tiered ramps, at 9% overage)
        if pt_per_ft:
            from .sheet import ramp_shapes
            for x in res.ramps["ramps"]:
                i = x.get("room_index", -1)
                if x.get("a") and 0 <= i < len(rooms) and rooms[i].get("outline_pt"):
                    sf = sum(_poly_area(p) for _k, p in ramp_shapes(x, pt_per_ft)) / pt_per_ft ** 2
                    sf = min(sf, rooms[i]["area_sf"])
                    rooms[i]["ramps_sf"] = rooms[i].get("ramps_sf", 0) + sf
                    rooms[i]["area_sf"] -= sf
                    # the higher room's underlayment carries on under a ramp between two buildups
                    # ("FIT30 Standard Transition over GenieMat FFNC25NBN"): its SF goes up by the
                    # ramp's footprint (Cottages Cardio: 1.7 pails for the FFNC25NBN, not 1.6)
                    j = x.get("over_index")
                    if j is not None and 0 <= j < len(rooms):
                        u = rooms[j].setdefault("under_ramps_sf", {})
                        for c in x.get("over_codes") or []:
                            u[c] = u.get(c, 0) + sf
            res.net_sf = sum(r["area_sf"] for r in rooms)

    # ---- quantities: each product's total across the takeoff decides its overage
    qty, names, turf_sf = {}, {}, {}
    for r in rooms:
        seen = set()
        for i, l in enumerate(r["_calc"]):
            p = P[l["code"]]
            sys_p = P[l["_sys"]] if l.get("_sys") else None
            if sys_p is not None and sys_p.get("bom") == "system":
                # listed as the system itself (PeakForce), once per room
                if l["_sys"] in seen:
                    continue
                seen.add(l["_sys"])
                p, k, name = sys_p, _key(sys_p, 0, l), _name(sys_p, 0, l)
            else:
                k, name = _key(p, i, l), _name(p, i, l)
            qty[k] = qty.get(k, 0) + r["area_sf"] + (r.get("under_ramps_sf") or {}).get(l["code"], 0)
            names[k] = (p, name)
            if p.get("count") == "turf":             # VersaTURF: ordered in 15' strips, room by room
                o, had = turf_order(R, r, pt_per_ft)
                turf_sf[k] = turf_sf.get(k, 0) + o
                if not had:
                    res.warnings.append(f"{r.get('name')}: no outline, so VersaTURF strips are estimated from a square room.")
        # perimeter isolation: the room's perimeter (outline + columns), in LF
        if needs_pmi(cat, r["_calc"]) and PMI_CODE in P:
            qty[PMI_CODE] = qty.get(PMI_CODE, 0) + (r.get("perimeter_lf") or 0)
            names[PMI_CODE] = (P[PMI_CODE], P[PMI_CODE]["display"])
    landing_sf = res.ramps["landing_sf"] if res.ramps else 0
    if landing_sf and "FIT08" in P:                 # FIT08 landings at doors that swing onto the buildup
        qty["FIT08"] = qty.get("FIT08", 0) + landing_sf
        names["FIT08"] = (P["FIT08"], P["FIT08"]["display"])
    pct = {k: overage_pct(R, q) for k, q in qty.items()}
    for k in sorted(qty, key=lambda k: (names[k][0]["_order"], k)):
        p, name = names[k]
        if not p.get("in_bom", True):
            continue
        if p.get("count") == "turf":                # ordered SF from the strips; overage = ordered / net - 1
            o = turf_sf.get(k, 0)
            res.lines.append(BomLine("product", name, _num(o), None, p["unit"],
                                     int(round((o / qty[k] - 1) * 100)) if qty[k] else 0, "SF"))
            continue
        if p.get("count") == "sf":                  # square feet only, no roll count
            sf = qty[k] * (1 + pct[k] / 100)
            if p.get("sf_multiple"):                 # Envy / PeakForce: ROUNDDOWN(CEILING(SF, 8*(8/12)*(59.8/12)), 1)
                m = 8 * (8 / 12) * (59.8 / 12)
                sf = math.floor(math.ceil(sf / m - 1e-9) * m * 10 + 1e-6) / 10
            else:
                sf = math.ceil(sf - 1e-9)
            res.lines.append(BomLine("product", name, round(sf, 1), None, p["unit"], pct[k], "SF"))
            continue
        if not p.get("unit_size"):
            raise TakeoffError(f"{p['display']} has no unit size in products.yaml yet.")
        units = math.ceil(qty[k] * (1 + pct[k] / 100) / p["unit_size"] - 1e-9)
        unit = "roll" if p["unit"] == "lf_roll" else p["unit"]
        res.lines.append(BomLine("product", name, _num(units * p["unit_size"]), units, unit,
                                 pct[k], "LF" if p["unit"] == "lf_roll" else "SF"))

    # ---- transitions: standard then ADA, FIT70 then FIT30, every tier counted
    if res.ramps:
        for series in ("FIT70", "FIT30"):
            for ada in (False, True):
                lf = res.ramps["lines"].get((series, ada))
                if lf:
                    res.lines.append(BomLine("transition", ramps_mod.name(series, ada), _num(lf), measure="LF"))

    # ---- adhesive: per room, per layer, onto the layer below (or the slab)
    A = R["adhesive"]
    pails = {}
    if res.ramps:
        pails["FAS1"] = res.ramps["adhesive"]
        if landing_sf:
            # FIT08 landing glued to the slab (rolled rate, FAS1), on the landings' share of the FIT08
            # actually ordered (whole rolls), as on the drafted sheets: 100 SF roll -> 0.2 pails (Parkview)
            ordered = next((ln.quantity for ln in res.lines if ln.item == P["FIT08"]["display"]), 0)
            share = landing_sf / qty["FIT08"] if qty.get("FIT08") else 0
            pails["FAS1"] += ordered * share / A["rates_sf_per_pail"]["rolled"] * A["fas1_factor"]
    fired = {}            # note id -> list of fill-ins (parts / products)

    def fire(nid, part=None):
        fired.setdefault(nid, [])
        if part and part not in fired[nid]:
            fired[nid].append(part)

    for r in rooms:
        layers = [dict(l, p=P[l["code"]]) for l in r["_calc"]]
        for i, l in enumerate(layers):
            p = l["p"]
            below = layers[i + 1]["p"] if i + 1 < len(layers) else None
            bond = l.get("bond") or p.get("bond", "glued")
            target = _name(below, i + 1, layers[i + 1]) if below else "slab"
            if p["unit"] == "lf_roll" or bond == "loose" or p.get("adhesive") == "none" and p["family"] != "plywood":
                continue
            sys_code = l.get("_sys")
            okey = _key(P[sys_code], 0, l) if sys_code and P[sys_code].get("bom") == "system" else _key(p, i, l)
            area = (r["area_sf"] + (r.get("under_ramps_sf") or {}).get(l["code"], 0)) * (1 + pct.get(okey, 0) / 100)
            if p["family"] == "fit_roll" and qty.get(okey):
                # a small FIT08 pad that takes a single roll is glued on the whole roll, as with a
                # landing (Parkview: 100 SF roll -> 0.2 pails; Modera Mystic's pad: 0.2, 2 pails in
                # all). Larger areas stay on net SF + overage (FIT08 on plywood: 24.3 pails)
                ordered = next((ln.quantity for ln in res.lines if ln.item == _name(p, i, l) and ln.units == 1), None)
                if ordered:
                    area = ordered * (r["area_sf"] + (r.get("under_ramps_sf") or {}).get(l["code"], 0)) / qty[okey]
            onto_ff = below is not None and below["family"] in FF_FAMILIES
            onto_tile = below is not None and below["family"] in TILE_FAMILIES

            if p["family"] == "finished_floor":
                if not r.get("finish_adhesive_by_pliteq"):
                    fire("finished_by_manufacturer")
                    continue
            if _glueless(below) and p["family"] in TILE_FAMILIES:
                fire("glueless")
                continue
            if p["family"] == "plywood":               # only here when the buildup says it's glued
                rate = A["plywood_glued"]["onto_fit_tiles"] if onto_tile else (
                    A["plywood_glued"]["onto_ff"] if onto_ff else A["plywood_glued"]["other"])
                adh, need = "FAS1", (area * (A["onto_ff_multiplier"] if onto_ff else 1)) / rate
                if below is not None and below.get("nbn"):
                    fire("plywood_not_ff", below["display"])
            elif p.get("adhesive") == "pending":
                res.warnings.append(f"No adhesive calculator yet for {p['display']}: its adhesive isn't included.")
                continue
            elif p.get("adhesive") == "turf":       # on the NET area: the ordered SF includes strip waste
                need = r["area_sf"] / A["turf_fas1_sf_per_pail"]
                if onto_tile:
                    adh, need = "FASHM", need / A["fas1_factor"]
                    fire("fashm_exclusive", p["display"])
                else:
                    adh = "FAS1"
            elif p.get("adhesive") == "plank":
                adh, need = "FAS3", area * (A["onto_ff_multiplier"] if onto_ff else 1) / A["fas3_sf_per_pail"]
            else:
                rate = A["rates_sf_per_pail"].get(p.get("adhesive"))
                if not rate:
                    res.warnings.append(f"No adhesive rate for {p['display']} ({p.get('adhesive')}).")
                    continue
                fashm = area * (A["onto_ff_multiplier"] if onto_ff else 1) / rate
                if below is not None and below["family"] == "plywood":
                    fashm += area / A["onto_plywood_extra_sf_per_pail"]
                if p.get("adhesive") in ("rolled", "turf") and onto_tile:
                    adh, need = "FASHM", fashm
                    fire("fashm_exclusive", p["display"])
                else:
                    adh, need = "FAS1", fashm * A["fas1_factor"]
            # a product made of several plies (FFNC50NBN = 2 x FFNC25NBN) is also glued ply to ply,
            # at the same rate; only the bottom ply's adhesive is quoted in the notes
            pails[adh] = pails.get(adh, 0) + need * p.get("plies", 1)
            part = (adh, _name(p, i, l), target)
            if below is not None and p["family"] not in TILE_FAMILIES:
                fired.setdefault("_interface", {}).setdefault(part, 0)
                fired["_interface"][part] += need
            elif below is None and p["unit"] in ("roll", "sheet"):
                # rolled goods glued to the slab get the adhesive note, even on their own (VersaTREAD09, FIT08, VersaTURF)
                fired.setdefault("_to_slab", {}).setdefault(part, 0)
                fired["_to_slab"][part] += need

        # buildup-shape notes
        fams = [l["p"]["family"] for l in layers]
        codes = [l["p"]["code"] for l in layers] + [l["code"] for l in r["layers"]]
        for i in range(len(layers) - 1):
            if fams[i] == "finished_floor" and fams[i + 1] in TILE_FAMILIES:
                fire("interstitial_recommended")
            if fams[i] == "finished_floor" and i + 2 < len(layers) and fams[i + 1] == "rst" and fams[i + 2] in TILE_FAMILIES:
                fire("finished_floor_approval")
            if fams[i] == "plywood" and fams[i + 1] == "plywood" and i + 2 < len(layers) and fams[i + 2] == "ff":
                fire("second_plywood")
        if "FFNC50NBN" in codes:
            fire("ffnc50_two_layers")
        if "FFNC75NBN" in codes:
            fire("ffnc75_three_layers")
        if "PeakForce" in codes or "PeakForceELITE" in codes:
            fire("peakforce_fas3")
        if any(c in ("PeakForce", "PeakForceELITE", "LDM5-25") for c in codes):
            fire("load_plates")

    # an amount quoted in a note is added to the total as quoted, to the tenth (Cottages Cardio:
    # 1.7 for the FFNC25NBN + 1.19 tiles + 0.12 ramps = 3.01 -> 4 pails, where 1.67 would give 3)
    quoted = dict(pails)
    for d in (fired.get("_interface") or {}, fired.get("_to_slab") or {}):
        for (adh, _n, _t), need in d.items():
            quoted[adh] += float(_pails(need)) - need
    for adh in ("FAS1", "FASHM", "FAS3"):
        if pails.get(adh):
            # BOM: whole pails, rounded UP (Pliteq, Nov 2026: 1.18 pails -> 2; every drafted sheet does this)
            res.lines.append(BomLine("adhesive", ADHESIVES[adh], max(1, math.ceil(quoted[adh] - 1e-9))))
    res.pails_raw = pails

    def parts_text(d):
        return ", ".join(f"{_pails(need)} pails of {ADHESIVES[adh].replace('Adhesive', 'adhesive')} for {name} to {target}"
                         for (adh, name, target), need in d.items())
    if fired.get("_interface"):
        fire("interface_adhesive", parts_text(fired.pop("_interface")))
    if fired.get("_to_slab"):
        fire("rolled_to_slab", parts_text(fired.pop("_to_slab")))

    all_codes = {l["code"] for r in rooms for l in r["layers"] + r["_calc"]}
    if {"FIT30", "FIT70"} <= all_codes:
        fire("fit30_interlock_support")
    if "xFIT70" in all_codes and ("PeakForce" in all_codes or "PeakForceELITE" in all_codes):
        fire("xfit_peakforce_glued")
    if T.get("no_transitions_date"):
        fire("client_no_transitions", T["no_transitions_date"])
    elif slab == "recessed":
        if _recess_mm(T):                          # "No transitions - 85mm recessed slab assumed."
            fire("recessed_depth_no_transitions", _num(_recess_mm(T)))
        else:
            fire("recessed_no_transitions")
    else:
        if slab == "partial" and _recess_mm(T):
            fire("recess_depth", _num(_recess_mm(T)))
        if T.get("no_ada_date"):
            fire("client_no_ada", T["no_ada_date"])
        for nid, fill in res.ramps["fire"]:
            fire(nid, fill)
    for nid in manual_notes or ():
        fire(nid)

    for n in cat["notes"]:
        if n["id"] not in fired:
            continue
        fills = fired[n["id"]]
        text = n["text"]
        if "{parts}" in text:
            text = text.replace("{parts}", fills[0] if fills else "")
        for ph in ("{date}", "{depth}"):
            if ph in text:
                text = text.replace(ph, str(fills[0]) if fills else "____")
        if "{product}" in text:
            text = text.replace("{product}", " and ".join(fills) if fills else "the product")
        res.notes.append(text)
        res.note_ids.append(n["id"])

    # ---- one group per distinct buildup (for callouts, fills and the tile grid)
    groups = {}
    for r in rooms:
        key = "/".join(f"{l['code']}{'' if not l.get('bond') or l['bond'] == P[l['code']].get('bond', 'glued') else ':' + l['bond']}"
                       f"{'|' + _colour(l) if _colour(l) else ''}" for l in r["layers"])
        r["_key"] = key
        g = groups.setdefault(key, {"key": key, "layers": [l["code"] for l in r["layers"]],
                                    "callout": callout(cat, r["layers"]), "rooms": [], "net_sf": 0.0})
        g["rooms"].append(r.get("name"))
        g["net_sf"] += r["area_sf"]
    for g in groups.values():
        th = [P[c].get("thickness_mm") for c in g["layers"]]
        g["thickness_mm"] = None if any(t is None for t in th) else round(sum(th), 2)
    res.groups = sorted(groups.values(), key=lambda g: -g["net_sf"])
    return res


def manual_notes(cat):
    return [{"id": n["id"], "text": n["text"], "when": n.get("when", "")} for n in cat["notes"] if not n.get("auto")]
