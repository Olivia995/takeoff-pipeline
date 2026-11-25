"""Turn selected room areas into a bill of materials for a flooring system."""
import math
from dataclasses import dataclass, field

import yaml


@dataclass
class BomLine:
    kind: str            # "product", "transition" or "adhesive"
    item: str
    quantity: float      # SF/LF for products, LF for transitions, pails for adhesive
    units: int = 0       # tiles/rolls/panels/planks
    unit: str = ""
    overage_pct: float = 0
    measure: str = "SF"

    def text(self):
        if self.kind == "adhesive":
            n = int(self.quantity)
            return f"{n} pail{'s' if n != 1 else ''} of {self.item}"
        if self.kind == "transition":
            return f"{_num(self.quantity)} LF {self.item}"
        if self.units is None:                       # counted in SF only (Envy, PeakForce)
            return f"{_num(self.quantity)} {self.measure} {self.item} ({_num(self.overage_pct)}% overage)"
        unit = self.unit + ("s" if self.units != 1 else "")
        return f"{_num(self.quantity)} {self.measure} {self.item} ({self.units} {unit}) ({_num(self.overage_pct)}% overage)"


@dataclass
class Takeoff:
    system: str
    callout: str
    net_sf: float
    lines: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    pails_raw: dict = field(default_factory=dict)     # adhesive key -> fractional pails
    groups: list = field(default_factory=list)        # [{system, callout, net_sf}] one per buildup


def _num(v):
    return f"{v:g}" if abs(v - round(v)) > 1e-9 else str(int(round(v)))


def load_catalog(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_takeoff(catalog, system_name, net_sf, transitions=None, overage_pct=None):
    """Compute BOM lines for `net_sf` square feet of the named system.

    transitions: {"standard": LF, "ada": LF}
    overage_pct: overrides the system/product/default overage for SF layers.
    """
    systems = catalog["systems"]
    if system_name not in systems:
        raise SystemExit(f"Unknown system '{system_name}'. Available: {', '.join(systems)}")
    sysdef = systems[system_name]
    products = catalog["products"]
    adhesives = catalog.get("adhesives", {})
    default_ov = catalog.get("defaults", {}).get("overage_pct", 9)

    t = Takeoff(system=system_name, callout=sysdef.get("callout", system_name), net_sf=net_sf)
    pails = {}          # adhesive key -> fractional pails
    bond_notes = []

    last_gross = net_sf
    for layer in sysdef["layers"]:
        if layer.get("adhesive_only"):
            # adhesive for a layer supplied by others (e.g. finished floor), spread over the layer below
            need = last_gross / layer["sf_per_pail"]
            pails[layer["adhesive"]] = pails.get(layer["adhesive"], 0) + need
            adh = adhesives.get(layer["adhesive"], {}).get("name", layer["adhesive"])
            bond_notes.append(f"{need:.1f} pails of {adh} for {layer.get('label', 'finished floor')} to {layer.get('bonds_to', 'the layer below')}")
            continue
        name = layer["product"]
        p = products[name]
        ov = overage_pct if overage_pct is not None else layer.get(
            "overage_pct", sysdef.get("overage_pct", p.get("overage_pct", default_ov)))
        units = math.ceil(net_sf * (1 + ov / 100) / p["unit_size"] - 1e-9)
        gross = units * p["unit_size"]
        last_gross = gross
        gross_display = round(gross, 1) if abs(gross - round(gross)) > 0.05 else round(gross)
        t.lines.append(BomLine("product", name, gross_display, units, p["unit"], ov, p.get("measure", "SF")))
        if layer.get("adhesive"):
            need = gross / layer["sf_per_pail"]
            pails[layer["adhesive"]] = pails.get(layer["adhesive"], 0) + need
            if layer.get("bonds_to"):
                target = "slab" if layer["bonds_to"] == "slab" else layer["bonds_to"]
                adh = adhesives.get(layer["adhesive"], {}).get("name", layer["adhesive"])
                bond_notes.append(f"{need:.1f} pails of {adh} for {name} to {target}")

    transitions = transitions or {}
    if transitions.get("standard"):
        t.lines.append(BomLine("transition", "GenieMat FIT70 Standard Transition", transitions["standard"], measure="LF"))
    if transitions.get("ada"):
        t.lines.append(BomLine("transition", "GenieMat FIT70 ADA Transition", transitions["ada"], measure="LF"))

    for key, need in pails.items():
        t.lines.append(BomLine("adhesive", adhesives.get(key, {}).get("name", key), max(1, math.ceil(need - 1e-9))))
    t.pails_raw = dict(pails)
    t.groups = [{"system": system_name, "callout": t.callout, "net_sf": net_sf}]

    if any(transitions.values()):
        t.notes.extend(catalog.get("transition_notes", []))
    for n in sysdef.get("notes", []):
        if any(transitions.values()) and n.startswith("No transitions"):
            continue
        t.notes.append(n)
    if bond_notes:
        t.notes.append(", ".join(bond_notes) + " included in the BOM.")
    return t


def build_multi(catalog, groups, transitions=None, overage_pct=None):
    """BOM for several buildups on one sheet, e.g. FIT70 in the gym and FIT70 over FFNC50NBN
    in the free-weight area. groups: [(system_name, net_sf), ...].

    Each buildup is quantified on its own area (units rounded up per area, as on a hand
    takeoff). Lines for the same product are added together; adhesive is summed across
    buildups before rounding up to whole pails. Transitions are added once."""
    groups = [(n, sf) for n, sf in groups if sf > 0]
    if len(groups) == 1:
        return build_takeoff(catalog, groups[0][0], groups[0][1], transitions, overage_pct)
    adhesives = catalog.get("adhesives", {})
    parts = [build_takeoff(catalog, n, sf, None, overage_pct) for n, sf in groups]
    t = Takeoff(system=" + ".join(n for n, _ in groups), callout="", net_sf=sum(sf for _, sf in groups))
    merged = {}
    for part in parts:
        for ln in part.lines:
            if ln.kind != "product":
                continue
            m = merged.get(ln.item)
            if m is None:
                merged[ln.item] = BomLine(ln.kind, ln.item, ln.quantity, ln.units, ln.unit, ln.overage_pct, ln.measure)
            else:
                m.quantity = round(m.quantity + ln.quantity, 1)
                m.units += ln.units
                if m.overage_pct != ln.overage_pct:
                    m.overage_pct = max(m.overage_pct, ln.overage_pct)
    t.lines = list(merged.values())
    transitions = transitions or {}
    if transitions.get("standard"):
        t.lines.append(BomLine("transition", "GenieMat FIT70 Standard Transition", transitions["standard"], measure="LF"))
    if transitions.get("ada"):
        t.lines.append(BomLine("transition", "GenieMat FIT70 ADA Transition", transitions["ada"], measure="LF"))
    pails = {}
    for part in parts:
        for k, v in part.pails_raw.items():
            pails[k] = pails.get(k, 0) + v
    for key, need in pails.items():
        t.lines.append(BomLine("adhesive", adhesives.get(key, {}).get("name", key), max(1, math.ceil(need - 1e-9))))
    t.pails_raw = pails
    if any(transitions.values()):
        t.notes.extend(catalog.get("transition_notes", []))
    for part in parts:
        for n in part.notes:
            if any(transitions.values()) and n.startswith("No transitions"):
                continue
            if n not in t.notes:
                t.notes.append(n)
    t.groups = [g for part in parts for g in part.groups]
    return t
