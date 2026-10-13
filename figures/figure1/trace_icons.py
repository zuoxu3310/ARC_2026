"""Trace the icons of the selected concept (B) into native draw.io stencil shapes.

Each icon region of concept-B.png is split into colour layers (nearest-colour
classification on a 4x Lanczos upsample), every layer is vectorised with
potrace, and the resulting Bezier outlines are written as a draw.io stencil
(`shape=stencil(...)`). A stencil is a native draw.io vector shape: it can be
moved, scaled, grouped and recoloured in the editor (fillColor is not baked in).

Output: icons.json with, per icon, its concept-pixel bounding box and one
stencil style per colour layer.
"""
from __future__ import annotations

import base64
import json
import re
import subprocess
import tempfile
import urllib.parse
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
CONCEPT = HERE / "concept-B.png"
UP = 8  # upsampling factor before tracing

WHITE = "#FFFFFF"
LIGHTBLUE = "#E5F1FE"
NAVY_LINE = "#12223A"   # dark outline of document / database icons
NAVY = "#0E3366"        # navy fills (gears, table header)
TEAL = "#038284"
DARK = "#383F49"        # dark person
GRAY = "#858E99"        # light persons
DOCGRAY = "#7D8997"     # text lines in documents
ARROW = "#5A697A"       # connector grey
LIGHT = "#D9DDE4"       # light bands / folded corners
RED = "#EB341F"

# name: (bbox x0,y0,x1,y1 in concept px, layers [(name, colour)], background colours)
ICONS = {
    "doc_stack":   ((38, 36, 179, 206), [("outline", NAVY_LINE), ("lines", DOCGRAY)], [WHITE]),
    "plus":        ((180, 116, 221, 156), [("plus", "#4A5969")], [WHITE]),
    "brain_doc":   ((224, 71, 346, 206), [("outline", NAVY_LINE), ("brain", "#707C8B"), ("fold", LIGHT)], [WHITE]),
    "people_a":    ((72, 292, 230, 396), [("dark", DARK), ("light", GRAY)], [WHITE]),
    "lock_a":      ((258, 476, 300, 522), [("lock", WHITE)], [NAVY]),
    "arrow_up":    ((398, 314, 435, 356), [("arrow", RED)], [WHITE]),
    "arrow_down":  ((398, 689, 432, 734), [("arrow", "#058A88")], [WHITE]),
    # people inside this icon are drawn from the cleanly traced people_c (same drawing in the concept)
    "doc_people":  ((733, 28, 864, 206), [("outline", NAVY_LINE), ("lines", "#A2AAB6"), ("fold", LIGHT)], [WHITE]),
    "gear_rf":     ((703, 249, 834, 374), [("gear", NAVY)], [WHITE]),
    "gear_cnn":    ((948, 249, 1080, 374), [("gear", TEAL)], [WHITE]),
    "brace":       ((780, 677, 1074, 718), [("brace", "#3A4250")], [LIGHTBLUE, "#909CA9", "#07827F"]),
    "database":    ((1234, 48, 1398, 246), [("outline", "#1B2938"), ("body", "#7F8B98"), ("band", "#DDE0E6")], [WHITE]),
    "big_doc_c":   ((1216, 476, 1329, 632), [("outline", NAVY_LINE), ("lines", DOCGRAY), ("fold", LIGHT)], [WHITE]),
    "doc_scoring": ((1405, 386, 1466, 462), [("outline", NAVY_LINE)], [WHITE]),
    "gear_train":  ((1398, 478, 1472, 552), [("gear", NAVY)], [WHITE]),
    "docs_method": ((1396, 565, 1480, 650), [("outline", NAVY_LINE)], [WHITE]),
    "people_c":    ((1393, 665, 1482, 747), [("dark", DARK), ("light", "#6B7684")], [WHITE]),
    "bigarrow_ab": ((560, 401, 632, 448), [("arrow", "#677685")], [WHITE]),
    "bigarrow_bc": ((1151, 401, 1219, 448), [("arrow", "#677685")], [WHITE]),
}


# regions painted with the background before tracing, and icons whose enclosed
# background is assigned to a fill layer (removes gradient streaks)
PAINT = {"doc_people": [(746, 98, 856, 206)]}
INTERIOR_FILL = {"database": "band"}
FOLD_BOX = {"brain_doc": (312, 72, 346, 104), "big_doc_c": (1292, 478, 1328, 512),
            "doc_people": (825, 30, 864, 66)}


def hex2rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


def potrace_paths(mask: np.ndarray) -> list[list[tuple]]:
    """Vectorise a boolean mask; return subpaths as lists of ('M'|'L'|'C', pts)."""
    with tempfile.TemporaryDirectory() as td:
        pbm = Path(td) / "m.pbm"
        Image.fromarray((~mask).astype(np.uint8) * 255).convert("1").save(pbm)
        svg = Path(td) / "m.svg"
        subprocess.run(["potrace", str(pbm), "-b", "svg", "-o", str(svg), "-t", "150",
                        "-a", "1.0", "-O", "0.2", "--flat"], check=True)
        text = svg.read_text()
    h_px = mask.shape[0]
    tf = re.search(r'translate\(([-\d.]+),([-\d.]+)\) scale\(([-\d.]+),([-\d.]+)\)', text)
    tx, ty, sx, sy = map(float, tf.groups())
    d = " ".join(re.findall(r' d="([^"]+)"', text))
    toks = re.findall(r'[MmLlCcZz]|-?\d+\.?\d*', d)
    subpaths, cur, i = [], [], 0
    x = y = sx0 = sy0 = 0.0
    cmd = None

    def tr(px, py):
        return ((tx + sx * px) / UP, (ty + sy * py) / UP)

    while i < len(toks):
        t = toks[i]
        if t in "MmLlCcZz":
            cmd = t
            i += 1
            if t in "Zz":
                if cur:
                    subpaths.append(cur)
                cur = []
                x, y = sx0, sy0
            continue
        if cmd in "Mm":
            nx, ny = float(toks[i]), float(toks[i + 1])
            i += 2
            if cmd == "m":
                nx, ny = x + nx, y + ny
            if cur:
                subpaths.append(cur)
            cur = [("M", [tr(nx, ny)])]
            x, y, sx0, sy0 = nx, ny, nx, ny
            cmd = "l" if cmd == "m" else "L"
        elif cmd in "Ll":
            nx, ny = float(toks[i]), float(toks[i + 1])
            i += 2
            if cmd == "l":
                nx, ny = x + nx, y + ny
            cur.append(("L", [tr(nx, ny)]))
            x, y = nx, ny
        elif cmd in "Cc":
            v = [float(t) for t in toks[i:i + 6]]
            i += 6
            if cmd == "c":
                v = [x + v[0], y + v[1], x + v[2], y + v[3], x + v[4], y + v[5]]
            cur.append(("C", [tr(v[0], v[1]), tr(v[2], v[3]), tr(v[4], v[5])]))
            x, y = v[4], v[5]
    if cur:
        subpaths.append(cur)
    return subpaths


def stencil_style(subpaths, w, h, name) -> str:
    def f(v):
        return f"{v:.2f}"
    parts = []
    for sp in subpaths:
        for op, pts in sp:
            if op == "M":
                parts.append(f'<move x="{f(pts[0][0])}" y="{f(pts[0][1])}"/>')
            elif op == "L":
                parts.append(f'<line x="{f(pts[0][0])}" y="{f(pts[0][1])}"/>')
            else:
                (a, b), (c, d), (e, g) = pts
                parts.append(f'<curve x1="{f(a)}" y1="{f(b)}" x2="{f(c)}" y2="{f(d)}" x3="{f(e)}" y3="{f(g)}"/>')
        parts.append("<close/>")
    xml = (f'<shape name="{name}" w="{w}" h="{h}" aspect="variable" strokewidth="inherit">'
           f'<connections/><background><path>{"".join(parts)}</path></background>'
           f'<foreground><fill/></foreground></shape>')
    enc = urllib.parse.quote(xml, safe="-_.!~*'()")
    comp = zlib.compressobj(9, zlib.DEFLATED, -15)
    raw = comp.compress(enc.encode("ascii")) + comp.flush()
    return "shape=stencil(" + base64.b64encode(raw).decode("ascii") + ");"


def disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def text_boxes() -> list:
    """Ink boxes of every concept text line (texts.json), padded by 3 px."""
    tj = HERE / "texts.json"
    if not tj.exists():
        return []
    return [[t["target"][0] - 3, t["target"][1] - 3, t["target"][2] + 3, t["target"][3] + 3]
            for t in json.loads(tj.read_text())]


def main() -> None:
    img = Image.open(CONCEPT).convert("RGB")
    tboxes = text_boxes()
    out = {}
    for name, ((x0, y0, x1, y1), layers, bgs) in ICONS.items():
        crop = img.crop((x0, y0, x1, y1)).copy()
        # paint any concept text that falls inside the icon box with the background
        bg = tuple(int(bgs[0][i:i + 2], 16) for i in (1, 3, 5))
        for tx0, ty0, tx1, ty1 in tboxes:
            ix0, iy0, ix1, iy1 = max(x0, tx0), max(y0, ty0), min(x1, tx1), min(y1, ty1)
            if ix0 < ix1 and iy0 < iy1:
                crop.paste(bg, (int(ix0 - x0), int(iy0 - y0), int(ix1 - x0), int(iy1 - y0)))
        for px0, py0, px1, py1 in PAINT.get(name, []):
            crop.paste(bg, (px0 - x0, py0 - y0, px1 - x0, py1 - y0))
        big = np.asarray(crop.resize(((x1 - x0) * UP, (y1 - y0) * UP), Image.LANCZOS)).astype(float)
        palette = [hex2rgb(c) for _, c in layers] + [hex2rgb(c) for c in bgs]
        dist = np.stack([np.sqrt(((big - p) ** 2).sum(-1)) for p in palette], -1)
        cls = dist.argmin(-1)
        w, h = x1 - x0, y1 - y0
        entry = {"bbox": [x0, y0, x1, y1], "layers": []}
        fill_layer = INTERIOR_FILL.get(name)
        if fill_layer:
            union = cls < len(layers)
            interior = ndi.binary_fill_holes(union) & ~union
            cls = np.where(interior, [l for l, _ in layers].index(fill_layer), cls)
        names = [l for l, _ in layers]
        if name in FOLD_BOX:
            # folded corner: keep the fold layer only inside its corner box and fill the
            # gradient specks enclosed by the fold lines there
            fi, oi = names.index("fold"), names.index("outline")
            fx0, fy0, fx1, fy1 = FOLD_BOX[name]
            sl = (slice((fy0 - y0) * UP, (fy1 - y0) * UP), slice((fx0 - x0) * UP, (fx1 - x0) * UP))
            outside = np.ones(cls.shape, bool)
            outside[sl] = False
            cls = np.where(outside & (cls == fi), len(layers), cls)
            is_bg = cls[sl] >= len(layers)
            sub = ndi.binary_fill_holes(((cls == fi) | (cls == oi))[sl]) & is_bg
            cls[sl] = np.where(sub, fi, cls[sl])
        for li, (lname, colour) in enumerate(layers):
            mask = cls == li
            # remove anti-aliasing slivers that the nearest-colour split assigns to
            # intermediate layers along other layers' edges
            mask = ndi.binary_opening(mask, structure=disk(4))
            lab, n = ndi.label(mask)
            if n:
                area = ndi.sum(mask, lab, range(1, n + 1))
                keep = np.isin(lab, 1 + np.nonzero(area >= 12 * UP * UP)[0])
                mask = mask & keep
            if mask.sum() < 30:
                continue
            if name.startswith("bigarrow"):
                lab, n = ndi.label(mask)
                if n > 1:
                    mask = lab == 1 + int(np.argmax(ndi.sum(mask, lab, range(1, n + 1))))
            sps = potrace_paths(mask)
            entry["layers"].append({"name": lname, "fill": colour,
                                    "style": stencil_style(sps, w, h, f"{name}_{lname}"),
                                    "subpaths": len(sps)})
        out[name] = entry
        print(name, [(l["name"], l["subpaths"]) for l in entry["layers"]])
    (HERE / "icons.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
