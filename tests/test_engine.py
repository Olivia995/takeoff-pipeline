"""Check the takeoff engine against quantities from real Pliteq takeoff sheets.

Run:  python tests/test_engine.py
Net areas are back-solved from each sheet's unit count. Layers are listed TOP FIRST.
Add a case whenever you finish a job.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from takeoff.engine import TakeoffError, compute, load, overage_pct  # noqa: E402

CAT = load()


def room(area, *codes, **kw):
    return dict({"name": "Room", "area_sf": area, "perimeter_lf": kw.pop("perimeter", 0),
                 "layers": [{"code": c} for c in codes]}, **kw)


CASES = [
    # (sheet, rooms, lines that must appear, notes that must appear)
    # Sheets marked "ramps" also had transitions: only the product lines and notes that
    # don't depend on ramps are checked (their FAS1 totals include ramp adhesive).
    ("Alyiah ASH-1", [room(2579, "FIT70")],
     ["2812 SF GenieMat FIT70 (703 tiles) (9% overage)", "2 pails of Pliteq FAS1 Adhesive"],
     ["No transitions - recessed slab assumed."]),
    ("175 3rd St STB-1", [room(3430, "FIT30")],
     ["3740 SF GenieMat FIT30 (935 tiles) (9% overage)", "3 pails of Pliteq FAS1 Adhesive"], []),
    # 1.14 pails -> 2 (BOM rounds up)
    ("23-07 43rd Ave 43A-1", [room(1396, "FIT30")],
     ["1524 SF GenieMat FIT30 (381 tiles) (9% overage)", "2 pails of Pliteq FAS1 Adhesive"], []),
    ("797 Richmond Road RRC-1", [room(335, "FIT70")],
     ["380 SF GenieMat FIT70 (95 tiles) (13% overage)", "1 pail of Pliteq FAS1 Adhesive"], []),
    ("OPP Red Lake RLM-1", [room(661, "FIT30")],
     ["736 SF GenieMat FIT30 (184 tiles) (11% overage)", "1 pail of Pliteq FAS1 Adhesive"], []),
    ("OPP Sioux Lookout SLM-1", [room(650, "FIT30")],
     ["724 SF GenieMat FIT30 (181 tiles) (11% overage)", "1 pail of Pliteq FAS1 Adhesive"], []),
    ("Reserve at Falls Church RFC-2", [room(802, "FIT30")],
     ["876 SF GenieMat FIT30 (219 tiles) (9% overage)", "1 pail of Pliteq FAS1 Adhesive"], []),
    ("2500 Westchester WA-1 (ramps)", [room(824, "FIT70")],
     ["900 SF GenieMat FIT70 (225 tiles) (9% overage)"], []),
    ("Cora Green Hills CGH-1 (ramps)", [room(1921, "FIT70")],
     ["2096 SF GenieMat FIT70 (524 tiles) (9% overage)"], []),
    ("Westerly WFCS-1 (ramps)", [room(2190, "FIT70", "FFNC50NBN")],
     ["2388 SF GenieMat FIT70 (597 tiles) (9% overage)", "2394 SF GenieMat FFNC50NBN (114 rolls) (9% overage)"],
     ["4.8 pails of Pliteq FAS1 adhesive for GenieMat FFNC50NBN to slab included in the BOM.",
      "GenieMat FFNC50NBN is two layers of GenieMat FFNC25NBN."]),
    ("Symphony Park SPC-1", [room(2187, "FINISH", "RST10", "FIT70")],
     ["2384 SF GenieMat FIT70-C1 (596 tiles) (9% overage)", "2400 SF GenieMat RST10 (40 rolls) (9% overage)",
      "16 pails of Pliteq FASHM Adhesive", "2 pails of Pliteq FAS1 Adhesive"],   # 1.43 rounded up
     ["15.9 pails of Pliteq FASHM adhesive for GenieMat RST10 to GenieMat FIT70-C1 included in the BOM.",
      "Pliteq FASHM is exclusively intended for the adhesion of GenieMat RST10 to GenieMat FIT tiles.",
      "Finished layer adhesive to be supplied by manufacturer."]),
    ("Lifetime Sky LFS-3 (ramps)", [room(3407, "FINISH", "RST10", "FIT70", finish_adhesive_by_pliteq=True)],
     ["3716 SF GenieMat FIT70-C1 (929 tiles) (9% overage)", "3720 SF GenieMat RST10 (62 rolls) (9% overage)",
      "25 pails of Pliteq FASHM Adhesive"],
     # sheet: 7.5 (7.44 rounded up); notes now round to the nearest tenth
     ["7.4 pails of Pliteq FAS1 adhesive for Finished Floor to GenieMat RST10, 24.8 pails of Pliteq FASHM "
      "adhesive for GenieMat RST10 to GenieMat FIT70-C1 included in the BOM."]),
    ("Park Hyatt PHV-1 (ramps)", [room(486, "VersaRUBBERELITE6", "RST10", "FIT70")],
     ["552 SF GenieMat FIT70-C1 (138 tiles) (13% overage)", "550.4 SF VersaRUBBER ELITE6 (172 planks) (13% overage)",
      "600 SF GenieMat RST10 (10 rolls) (13% overage)", "4 pails of Pliteq FASHM Adhesive",
      "1 pail of Pliteq FAS3 Adhesive"], []),
    ("Visa Foster City FCM-1 (ramps)", [room(1592, "FIT70", "PLY", "PLY", "FF70LDM", perimeter=150)],
     ["1736 SF GenieMat FIT70 (434 tiles) (9% overage)", "1738.1 SF GenieMat FF70LDM (463 panels) (9% overage)",
      "200 LF GenieMat PMI12PF (4 rolls) (15% overage)"], []),
]


def check_overage_tiers():
    R = CAT["rules"]
    assert [overage_pct(R, v) for v in (100, 250, 251, 500, 750, 751, 5000)] == [15, 15, 13, 13, 11, 9, 9]


def check_drafted_jobs():
    """Whole jobs drawn in the app, against the drafter's own sheet for the same plan
    (tests/fixtures/*.json are the saved rooms.json; the plan PDFs are in plans/)."""
    import json
    from takeoff import server
    want = {
        "cottages_cardio.json": ["44 LF GenieMat FIT30 Standard Transition", "14 LF GenieMat FIT30 ADA Transition",
                                 "4 pails of Pliteq FAS1 Adhesive"],
        "modera_mystic.json": ["36 LF GenieMat FIT30 Standard Transition", "3.5 LF GenieMat FIT30 ADA Transition",
                               "2 pails of Pliteq FAS1 Adhesive"],
        "maroon_creek.json": ["44 LF GenieMat FIT70 Standard Transition", "7 LF GenieMat FIT70 ADA Transition",
                              "3 pails of Pliteq FAS1 Adhesive"],
        # the same job with the rooms detected again (outlines traced a little differently at the doors)
        "maroon_creek_redetected.json": ["44 LF GenieMat FIT70 Standard Transition", "7 LF GenieMat FIT70 ADA Transition"],
    }
    fx = Path(__file__).parent / "fixtures"
    for name, lines in want.items():
        d = json.loads((fx / name).read_text())
        plan = Path(d["source"])
        if not plan.exists():
            plan = Path(__file__).resolve().parent.parent / "plans" / plan.name
        if not plan.exists():
            print(f"skip {name}: plan PDF not found")
            continue
        body = {"plan": str(plan), "page": d.get("page", 1), "scale": d["scale"], "in_per_ft": d["scale"]["in_per_ft"],
                "rooms": d["rooms"], **d["takeoff"]}
        _cat, t, _rooms = server._takeoff(body)
        got = [l.text() for l in t.lines]
        assert all(l in got for l in lines), (name, got)
    # Cottages: the FFNC25NBN under the ramps between the rooms, quoted as the drafter did
    d = json.loads((fx / "cottages_cardio.json").read_text())
    body = {"plan": d["source"], "page": 1, "scale": d["scale"], "in_per_ft": d["scale"]["in_per_ft"], "rooms": d["rooms"], **d["takeoff"]}
    notes = server._takeoff(body)[1].notes
    assert "1.7 pails of Pliteq FAS1 adhesive for GenieMat FFNC25NBN to slab included in the BOM." in notes, notes
    assert any(n.startswith("GenieMat FFNC25NBN to be used for the tiered ramping") for n in notes), notes


def check_transitions():
    from takeoff.ramps import ramp_for

    def run(rooms, *openings, **kw):
        T = dict({"slab": "raised", "openings": list(openings)}, **kw)
        res = compute(CAT, rooms, transitions=T)
        return res, [l.text() for l in res.lines]

    def door(w=36, k="ada", **kw):
        return dict({"length_in": w, "room": 0, "type": k}, **kw)
    # ramp choice by height: h / 70 rounded up FIT70 tiers, all but the first on wood blocking
    # (Lifetime 85 mm and Park Hyatt 86 mm: 1 + 1 over blocking; Visa 178 mm: 1 + 2x over blocking)
    assert ramp_for(25) == ("FIT30", 1, True, False) and ramp_for(30) == ("FIT30", 1, False, False)
    assert ramp_for(50) == ("FIT70", 1, True, False) and ramp_for(70) == ("FIT70", 1, False, False)
    assert ramp_for(85) == ("FIT70", 2, True, True) and ramp_for(178) == ("FIT70", 3, True, True)
    assert ramp_for(140) == ("FIT70", 2, False, True)
    # Visa Foster City FCM-1 (178 mm, 36" door with a FIT08 landing): 3 tiers -> 10.5 LF ADA, 68 LF standard
    _, lines = run([room(1592, "FIT70", "PLY", "PLY", "FF70LDM")], door(36, landing=True))
    assert [l for l in lines if "Transition" in l] == ["68 LF GenieMat FIT70 Standard Transition",
                                                        "10.5 LF GenieMat FIT70 ADA Transition"], lines
    # Park Hyatt PHV-1 (86 mm, wide door with a landing): 2 tiers of 2 ADA pieces -> 14 LF ADA, 48 LF standard
    _, lines = run([room(486, "VersaRUBBERELITE6", "RST10", "FIT70")], door(48, landing=True))
    assert [l for l in lines if "Transition" in l] == ["48 LF GenieMat FIT70 Standard Transition",
                                                        "14 LF GenieMat FIT70 ADA Transition"], lines
    # The Heartwood Library THL-1: one ADA ramp with two side wings -> 3.5 LF ADA + 8 LF standard
    _, lines = run([room(652, "FIT70")], door())
    assert [l for l in lines if "Transition" in l] == ["8 LF GenieMat FIT70 Standard Transition",
                                                        "3.5 LF GenieMat FIT70 ADA Transition"], lines
    # FIT30, two 42" ADA doors: 4 side wings of 13" (18" rounded) share pieces two at a time -> 8 LF, not 16
    _, lines = run([room(786, "FIT30")], door(42), door(42))
    assert [l for l in lines if "Transition" in l] == ["8 LF GenieMat FIT30 Standard Transition",
                                                        "7 LF GenieMat FIT30 ADA Transition"], lines
    # Cora Green Hills: 3 ADA ramps (10.5 LF), 8 LF standard: one centred, two between walls
    res, lines = run([room(1921, "FIT70")], door(), door(sides=0), door(sides=0))
    assert "8 LF GenieMat FIT70 Standard Transition" in lines and "10.5 LF GenieMat FIT70 ADA Transition" in lines, lines
    assert "ADA transition placements should always be verified by an ADA consultant." in res.notes
    assert "No transitions - recessed slab assumed." not in res.notes
    # 2500 Westchester: one ADA ramp flush between walls, no wings
    _, lines = run([room(824, "FIT70")], door(sides=0))
    assert [l for l in lines if "Transition" in l] == ["3.5 LF GenieMat FIT70 ADA Transition"], lines
    # every tier counted: 2x ADA ramp = 7 LF
    _, lines = run([room(1000, "FIT70", "FF70LDM")], door(sides=0))
    assert "7 LF GenieMat FIT70 ADA Transition" in lines, lines
    # adhesive (FAS1) = LF x depth / 12 / 500
    for code, k, depth in (("FIT30", "std", 12), ("FIT30", "ada", 13), ("FIT70", "std", 18), ("FIT70", "ada", 32.5)):
        res, _ = run([room(1000, code)], door(96 if k == "std" else 42, k, sides=0))
        lf = 8 if k == "std" else 3.5
        base = compute(CAT, [room(1000, code)]).pails_raw["FAS1"]
        assert abs(res.pails_raw["FAS1"] - base - lf * depth / 12 / 500) < 1e-6, (code, k, res.pails_raw)
    # a wide ADA opening: ADA ramp in the middle, standard wings wall to wall
    _, lines = run([room(1000, "FIT70")], door(42 + 2 * 60, ada_width="one"))   # 2 x 60" runs (2 pieces each) + 2 side wings (1 each) = 6, + 1 spare
    assert "28 LF GenieMat FIT70 Standard Transition" in lines, lines
    # partly recessed slab: the recess comes off the ramp height; note with the depth
    res, lines = run([room(1000, "FIT70", "FFNC25")], door(sides=0), slab="partial", recess_mm=25)
    assert "3.5 LF GenieMat FIT70 ADA Transition" in lines and "25mm recessed slab assumed." in res.notes, (lines, res.notes)
    res, lines = run([room(1000, "FIT70")], slab="recessed", recess_mm=85)       # 49 Ontario
    assert "No transitions - 85mm recessed slab assumed." in res.notes and not any("Transition" in l for l in lines)
    # client asked for no transitions / no ADA ramps
    res, lines = run([room(1000, "FIT70")], door(), no_transitions_date="2026-11-02")
    assert not any("Transition" in l for l in lines) and "Transitions not included as per client request on 2026-11-02." in res.notes
    res, lines = run([room(1000, "FIT70")], door(sides=0), no_ada_date="2026-11-02")
    assert "4 LF GenieMat FIT70 Standard Transition" in lines and not any("ADA Transition" in l for l in lines), lines
    assert "ADA Transitions not included as per client request on 2026-11-02." in res.notes
    # FIT08 landing at a door that swings onto the buildup: 36 x 38 in = 9.5 SF -> 1 roll
    res, lines = run([room(1000, "FIT70")], door(landing=True))
    assert "100 SF GenieMat FIT08 (1 roll) (15% overage)" in lines, lines
    assert any(n.startswith("Any doors sitting above GenieMat material") for n in res.notes), res.notes
    # Parkview PTC-1: a 48" door swinging in -> 2 ADA pieces (7 LF), FIT08 landing, standard ramps down
    # both landing sides (2 pieces each) + 2 ADA side wings; a second 48" ADA door between walls.
    # 6 standard pieces + 1 spare = 28 LF; 4 ADA = 14 LF
    res, lines = run([room(1508, "FIT70")], door(48, landing=True), door(48, sides=0))
    assert [l for l in lines if "FIT08" in l or "Transition" in l] == [
        "100 SF GenieMat FIT08 (1 roll) (15% overage)", "28 LF GenieMat FIT70 Standard Transition",
        "14 LF GenieMat FIT70 ADA Transition"], lines
    assert "2 pails of Pliteq FAS1 Adhesive" in lines, lines              # 1.0 FIT70 + 0.2 landing + 0.15 ramps = 1.35 -> 2
    # buildups touching: the ramp only covers the difference in height
    res, lines = run([room(500, "FIT70", "FFNC25"), room(500, "FIT30")], {"length_in": 36, "room": 0, "other": 1, "type": "std"})
    assert res.ramps["ramps"][0]["height_mm"] == 65 and "4 LF GenieMat FIT70 Standard Transition" in lines


def main():
    check_overage_tiers()
    bad = 0
    for name, rooms, want_lines, want_notes in CASES:
        res = compute(CAT, rooms)
        lines = [ln.text() for ln in res.lines]
        for w in want_lines:
            if w not in lines:
                bad += 1
                print(f"FAIL {name}: missing line {w!r}\n  got {lines}")
        for w in want_notes:
            if w not in res.notes:
                bad += 1
                print(f"FAIL {name}: missing note {w!r}\n  got {res.notes}")
    # overage comes from the product's total across rooms
    res = compute(CAT, [room(300, "FIT70"), room(300, "FIT70")])
    assert res.lines[0].overage_pct == 11, res.lines[0].text()
    # PMI12PF: the perimeter in LF, added automatically only for plywood, FF under concrete or FF70
    def pmi(*codes, per=82):
        return [l.text() for l in compute(CAT, [room(400, *codes, perimeter=per)]).lines if "PMI" in l.item]
    assert pmi("FIT70") == [] and pmi("FIT70", "PMI12PF") == [] and pmi("FIT70", "FFNC25") == [], "PMI where not needed"
    assert pmi("FIT70", "PLY", "FFNC25") == ["100 LF GenieMat PMI12PF (2 rolls) (15% overage)"]   # 82 x 1.15 = 94.3
    assert pmi("FIT70", "FF70LDM") and pmi("FINISH", "CONCRETE", "FF25") and not pmi("FF25", "CONCRETE")
    # total perimeter across rooms sets the PMI overage tier
    res = compute(CAT, [room(400, "FIT70", "PLY", "FF25", perimeter=200), room(400, "FIT70", "PLY", "FF25", perimeter=100)])
    assert [l.overage_pct for l in res.lines if "PMI" in l.item] == [13]
    # multi-ply FF: every ply is glued (FFNC50 = 2 x FFNC25, FFNC50NBN = 2 x FFNC25NBN)
    one = compute(CAT, [room(1000, "FFNC25")]).pails_raw["FAS1"]
    assert abs(compute(CAT, [room(1000, "FFNC50")]).pails_raw["FAS1"] - 2 * one) < 1e-6
    # systems: FIT55 = FIT30 (FAS1, onto FF) on FFNC25 (FAS1); the BOM lists the components
    res = compute(CAT, [room(1000, "FIT55")])
    items = [l.item for l in res.lines]
    assert items[:2] == ["GenieMat FIT30", "GenieMat FFNC25"], items
    a = 1000 * 1.09
    assert abs(res.pails_raw["FAS1"] - (a / 400 * 1.5151515 + a / 150) * 0.3) < 0.01, res.pails_raw
    assert res.groups[0]["callout"] == "GenieMat FIT55" and res.groups[0]["thickness_mm"] == 55
    # PeakForce = Envy (FAS3) on RST10 (FASHM) on FIT30-C1 (FAS1): one BOM line in SF, no roll count
    res = compute(CAT, [room(1000, "PeakForce")])
    assert [l.text() for l in res.lines][0] == "1116.2 SF GenieMat PeakForce (9% overage)", [l.text() for l in res.lines]
    assert {k: round(v, 2) for k, v in res.pails_raw.items()} == {"FAS3": 1.82, "FASHM": 7.27, "FAS1": 0.82}, res.pails_raw
    assert "1116.2 SF GenieMat Envy (9% overage)" in [l.text() for l in compute(CAT, [room(1000, "Envy", "RST10", "FIT30")]).lines]
    # Envy / PeakForce / PeakForce ELITE: ROUNDDOWN(CEILING(SF after overage, 8*(8/12)*(59.8/12)), 1), no unit count
    for area, want in ((1000, "1116.2"), (200, "239.2"), (24.4, "53.1")):
        got = [l.text() for l in compute(CAT, [room(area, "PeakForceELITE")]).lines][0]
        assert got.startswith(want + " SF GenieMat PeakForce ELITE ("), got
    # from the Nov 2026 review of ~100 drafted sheets
    def notes_of(*rooms):
        return compute(CAT, list(rooms)).notes
    # Rambler Northgate: VersaDUEL15 rolls on FIT70-C1 -> FASHM at 150 SF per pail on the ordered SF
    n = notes_of(room(3740, "VersaDUEL15", "FIT70"))
    assert "27.2 pails of Pliteq FASHM adhesive for VersaDUEL15 to GenieMat FIT70-C1 included in the BOM." in n, n
    assert "Pliteq FASHM is exclusively intended for the adhesion of VersaDUEL15 to GenieMat FIT tiles." in n
    # 49 Ontario: VersaTREAD09 on its own to the slab, 100 SF roll -> 0.2 pails FAS1, with the note
    assert "0.2 pails of Pliteq FAS1 adhesive for VersaTREAD09 to slab included in the BOM." in notes_of(room(80, "VersaTREAD09"))
    # 9 West 57th: VersaTREAD08 to RST10 -> 3.0 pails FAS1
    assert any(x.startswith("3.0 pails of Pliteq FAS1 adhesive for VersaTREAD08 to GenieMat RST10")
               for x in notes_of(room(1376, "VersaTREAD08", "RST10", "FIT30"))), notes_of(room(1376, "VersaTREAD08", "RST10", "FIT30"))
    # HUMN: VersaRUBBER sheets (24 SF) to slab, 6024 SF -> 12.1 pails
    res = compute(CAT, [room(5520, "VersaRUBBER10")])
    assert "6024 SF VersaRUBBER10 (251 sheets) (9% overage)" in [l.text() for l in res.lines] and abs(res.pails_raw["FAS1"] - 12.05) < 0.05
    # VersaTURF: 15' strips the long way, length to the next foot; FAS1 = net / 300
    turf = dict(room(322, "VersaTURF", "RST10", "FIT70"), outline_pt=[[0, 0], [40.4, 0], [40.4, 7.97], [0, 7.97]], _ppf=1.0)
    res = compute(CAT, [turf])
    assert "615 SF VersaTURF (91% overage)" in [l.text() for l in res.lines], [l.text() for l in res.lines]   # Hillsboro
    assert any(x.startswith("1.1 pails of Pliteq FAS1 adhesive for VersaTURF to GenieMat RST10") for x in res.notes), res.notes
    turf = dict(room(897, "VersaTURF", "FFNC50NBN"), outline_pt=[[0, 0], [60.5, 0], [60.5, 14.83], [0, 14.83]], _ppf=1.0)
    res = compute(CAT, [turf])                                                       # Anytime Fitness South Regina
    assert "915 SF VersaTURF (2% overage)" in [l.text() for l in res.lines], [l.text() for l in res.lines]
    assert any(x.startswith("3.0 pails of Pliteq FAS1 adhesive for VersaTURF to GenieMat FFNC50NBN") for x in res.notes), res.notes
    # Econofitness: FIT08 over 2x 1/2" plywood over FF70 over FFNC25NBN (FFNC25NBN to slab 22.1, FIT08 to plywood 24.3)
    n = notes_of(room(10134, "FIT08", "PLY12", "PLY12", "FF70", "FFNC25NBN"))
    assert any("22.1 pails of Pliteq FAS1 adhesive for GenieMat FFNC25NBN to slab" in x for x in n), n
    assert any("24.3 pails of Pliteq FAS1 adhesive for GenieMat FIT08 to ½\" Plywood" in x for x in n), n
    assert compute(CAT, [room(10134, "FIT08", "PLY12", "PLY12", "FF70", "FFNC25NBN")]).groups[0]["callout"].startswith(
        "GenieMat FIT08 over\n2x ½\" Plywood (supplied by others) over")
    # 290 Cole St: vinyl finished floor supplied with FAS3 (1260 SF -> 2.1 pails), RST10 to FIT55 with FASHM (8.4)
    res = compute(CAT, [room(1156, "FINISH_VINYL", "RST10", "FIT55", finish_adhesive_by_pliteq=True)])
    assert any(x.startswith("2.1 pails of Pliteq FAS3 adhesive for Vinyl Finished Floor to GenieMat RST10") for x in res.notes), res.notes
    assert res.groups[0]["callout"].startswith("Vinyl Finished Floor (supplied by others) over"), res.groups[0]["callout"]
    # colours: each colour is its own BOM line with its own overage
    def col(area, colour, *codes):
        r = room(area, *codes)
        r["layers"][0]["colour"] = colour
        return r
    res = compute(CAT, [col(1284, "A", "FIT70", "FFNC25NBN"), col(946, "B", "FIT70", "FFNC25NBN")])   # 215 Lakeshore CGB2
    lines = [l.text() for l in res.lines]
    assert lines[:2] == ["1400 SF GenieMat FIT70 (Colour A) (350 tiles) (9% overage)",
                         "1032 SF GenieMat FIT70 (Colour B) (258 tiles) (9% overage)"], lines
    assert sorted(g["callout"] for g in res.groups) == ["GenieMat FIT70 (Colour A) over\nGenieMat FFNC25NBN",
                                                         "GenieMat FIT70 (Colour B) over\nGenieMat FFNC25NBN"]
    lines = [l.text() for l in compute(CAT, [col(2010, "A", "FIT70"), col(445, "B", "FIT70")]).lines]   # 49 Ontario East
    assert "504 SF GenieMat FIT70 (Colour B) (126 tiles) (13% overage)" in lines, lines
    # FFNC17NBN: 60 SF rolls (72 Perth Ave)
    assert "2220 SF GenieMat FFNC17NBN (37 rolls) (9% overage)" in [l.text() for l in compute(CAT, [room(2030, "FIT70", "FFNC17NBN")]).lines]
    # FFNC75NBN = 3 x FFNC25NBN at 14 SF per roll; FFNC32NBN 13.2 SF per roll
    res = compute(CAT, [room(1000, "FIT70", "FFNC75NBN")])
    assert "1092 SF GenieMat FFNC75NBN (78 rolls) (9% overage)" in [l.text() for l in res.lines]
    assert abs(res.pails_raw["FAS1"] - 7.53) < 0.01, res.pails_raw
    assert "1095.6 SF GenieMat FFNC32NBN (83 rolls) (9% overage)" in [l.text() for l in compute(CAT, [room(1000, "FIT30", "FFNC32NBN")]).lines]
    # FFNC50NBN is two plies of FFNC25NBN: glued to the slab AND ply to ply
    res = compute(CAT, [room(2190, "FIT70", "FFNC50NBN")])
    assert abs(res.pails_raw["FAS1"] - 11.72) < 0.01, res.pails_raw
    # rounding: BOM rounds pails up (at least 1), notes to the nearest tenth
    from takeoff.engine import _pails
    assert [_pails(v) for v in (0.92, 7.44, 3.66, 0.25)] == ["0.9", "7.4", "3.7", "0.3"]
    assert [l.text() for l in compute(CAT, [room(300, "FIT70")]).lines][-1] == "1 pail of Pliteq FAS1 Adhesive"
    # notes that must NOT appear
    res = compute(CAT, [room(2190, "FIT70", "FFNC50NBN")])          # FIT on FFNC..NBN is still glued
    assert not any("glueless" in n for n in res.notes), res.notes
    res = compute(CAT, [room(1592, "FIT70", "PLY", "PLY", "FF70LDM")])  # 2nd-plywood note is for rolled FF only
    assert not any("2nd layer of plywood" in n for n in res.notes), res.notes
    check_transitions()
    check_drafted_jobs()
    print("All sample takeoffs match." if not bad else f"{bad} problem(s).")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
