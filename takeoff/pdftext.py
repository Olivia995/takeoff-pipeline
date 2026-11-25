"""The page's text, readable even when a font's character codes are shifted.

Some CAD exports embed fonts whose codes are offset from the letters they draw: the sheet
shows "EXIT" but the text layer holds "(;67" (every code 29 lower), so room names, room
numbers and the scale note can't be read. Each font's offset is found by trying shifts
and keeping the one that turns its text into common plan words.
"""
import re

import pymupdf

WORDS = {
    "FLOOR", "PLAN", "PLANS", "SCALE", "ROOM", "FITNESS", "GYM", "YOGA", "EXIT", "WALL", "WALLS", "NOTES",
    "NOTE", "TYP", "FINISH", "FINISHES", "LEVEL", "DOOR", "DOORS", "STAIR", "STAIRS", "ELEV", "ELEVATOR",
    "CORRIDOR", "OFFICE", "STORAGE", "LOBBY", "THE", "AND", "OF", "TO", "FOR", "ALL", "WITH", "SEE", "BASE",
    "CEILING", "GENERAL", "KEY", "NEW", "EXISTING", "MECH", "ELEC", "TILE", "CARPET", "RUBBER", "AREA",
    "DRAWING", "DRAWINGS", "SHEET", "PROJECT", "DATE", "REV", "ISSUED", "BY", "AT", "ON", "IN", "NOT",
    "UNLESS", "OTHERWISE", "NOTED", "PROVIDE", "CONTRACTOR", "WASHROOM", "RESTROOM", "LOCKER", "STUDIO",
    "AMENITY", "TOILET", "JANITOR", "CLOSET", "VESTIBULE", "SUITE", "UNIT", "TO", "BE", "ARE", "IS", "FROM",
    "DETAIL", "DETAILS", "SECTION", "PAINT", "VINYL", "CONCRETE", "MATERIAL", "SCHEDULE", "LEGEND",
}
_TOKEN = re.compile(r"[A-Z]{3,}|[a-z]{3,}")   # 2-letter words match too easily by chance
_CACHE = {}


def _shift(text, k):
    # codes below 32 count too: shifted fonts put digits and the space there ("\x16" = "3")
    return "".join(chr(ord(c) + k) if 32 <= ord(c) + k < 127 and ord(c) < 127 and c != " " else c for c in text)


def _hits(texts):
    # case matters: a shift of 32 only swaps upper and lower case, which isn't a decoding
    return sum(1 for t in texts for w in _TOKEN.findall(t) if w in WORDS or w.upper() in WORDS and w.islower())


def font_shifts(page):
    """{font name: code offset} for fonts whose text reads as plan words only once shifted."""
    key = (page.parent.name, id(page.parent), page.number, page.rotation)
    if key in _CACHE:
        return _CACHE[key]
    by_font = {}
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if s["text"].strip():
                    by_font.setdefault(s["font"], []).append(s["text"])
    out = {}
    for font, texts in by_font.items():
        texts = texts[:400]
        base = _hits(texts)
        best = max(((k, _hits([_shift(t, k) for t in texts])) for k in range(-40, 41) if k and abs(k) != 32), key=lambda t: t[1])
        distinct = len({w for t in texts for w in _TOKEN.findall(_shift(t, best[0]))
                        if w in WORDS or w.upper() in WORDS and w.islower()})
        if best[1] >= 6 and distinct >= 3 and best[1] > 3 * base + 2:
            out[font] = best[0]
    _CACHE[key] = out
    if len(_CACHE) > 16:
        _CACHE.clear()
    return out


def text_dict(page):
    """page.get_text("dict") with shifted fonts decoded."""
    d = page.get_text("dict")
    sh = font_shifts(page)
    if sh:
        for b in d["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    if s["font"] in sh:
                        s["text"] = _shift(s["text"], sh[s["font"]])
    return d


def text_words(page):
    """page.get_text("words") with shifted fonts decoded."""
    sh = font_shifts(page)
    if not sh:
        return page.get_text("words")
    words = []
    raw = page.get_text("rawdict")
    for bn, b in enumerate(raw["blocks"]):
        for ln, l in enumerate(b.get("lines", [])):
            cur, wn = [], 0
            def flush():
                nonlocal cur, wn
                if cur:
                    words.append((min(c[0] for c in cur), min(c[1] for c in cur), max(c[2] for c in cur),
                                  max(c[3] for c in cur), "".join(c[4] for c in cur), bn, ln, wn))
                    wn += 1
                cur = []
            for s in l["spans"]:
                k = sh.get(s["font"], 0)
                for ch in s["chars"]:
                    c = _shift(ch["c"], k) if k else ch["c"]
                    if c.isspace():
                        flush()
                    else:
                        x0, y0, x1, y1 = ch["bbox"]
                        cur.append((x0, y0, x1, y1, c))
            flush()
    return words
