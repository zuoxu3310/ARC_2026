"""Figure 1, layout B2: the faithful version-B rebuild re-spaced on an aligned three-row grid.

Coordinates are canvas units (1774 wide; 0.2837 pt per unit at the 6.99-in print width).
Boxes, lines, arrowheads, tables and dividers are native draw.io shapes; icons
are the potrace stencils from icons.json; all text is native Source Sans 3.
The only intentional departures from the concept are print-size driven: text
smaller than 8 pt at the 6.99-in print width is enlarged to >= 8 pt (table
headers, two box labels), the two small tables are widened slightly to fit
those headers, and panel letters use the IEEE template's Times New Roman.

Usage:  python build_figure1_B.py [--texts-only]
Writes Figure1_B.drawio (or work/texts_only.drawio). calib.json, if present,
holds per-text offsets measured by calibrate_text.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import ImageFont

HERE = Path(__file__).resolve().parent
W, H = 1774, 1000
PT_PER_UNIT = 503.24 / W            # template \textwidth (6.99 in) / canvas width
MIN_UNITS = 8.05 / PT_PER_UNIT      # smallest font size giving >= 8 pt in print
FONT = "Source Sans 3"   # rendered through static single-face instances (fonts/)
FONT_FILE = HERE / "fonts/SourceSans3[wght].ttf"
STATIC = {700: "SS3 Fig Bold", 800: "SS3 Fig ExtraBold"}  # TrueType in PDF

NAVY_TEXT = "#0B1535"
NAVY = "#0E3366"
NAVY_DARK = "#0E2D5C"
GREY = "#5A697A"
DIVIDER = "#C2C9D2"
RED = "#EB341F"
TEAL = "#058A88"
TEAL_TEXT = "#07827F"
LIGHTBLUE = "#E5F1FE"
HEADER_GREY = "#DCE0E6"
LABEL_FILL = "#E9ECF1"
LABEL_LINE = "#C3C8D1"

cells: list[str] = []
OFF = [0.0]   # vertical offset of the current layout block
_n = [0]
TEXT_ONLY = "--texts-only" in sys.argv
VARIANT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--variant=")), None)
CALIB = json.loads((HERE / "calib.json").read_text()) if (HERE / "calib.json").exists() else {}
TEXTS: list[dict] = []


def nid() -> str:
    _n[0] += 1
    return f"e{_n[0]}"


def q(s: str) -> str:
    return escape(s, {'"': "&quot;"})


def vertex(x, y, w, h, style, value=""):
    if TEXT_ONLY and not (style.startswith("text;") or (x == 0 and y == 0 and w == W)):
        return
    y = y + OFF[0]
    cells.append(f'<mxCell id="{nid()}" value="{q(value)}" style="{q(style)}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/></mxCell>')


def rect(x, y, w, h, fill, stroke="none", sw=0, arc=0, dashed=False, dash="16 9"):
    st = (f"rounded={1 if arc else 0};absoluteArcSize=1;arcSize={arc};whiteSpace=wrap;html=1;"
          f"fillColor={fill};strokeColor={stroke};strokeWidth={sw};")
    if dashed:
        st += f"dashed=1;fixDash=1;dashPattern={dash};"
    vertex(x, y, w, h, st)


def framed(x0, y0, x1, y1, fill, stroke, sw, arc, **kw):
    """Box whose OUTER stroke edge is (x0,y0,x1,y1), as measured in the concept."""
    h = sw / 2
    rect(x0 + h, y0 + h, x1 - x0 - sw, y1 - y0 - sw, fill, stroke, sw, arc, **kw)


def ellipse(x0, y0, x1, y1, fill, stroke, sw):
    h = sw / 2
    vertex(x0 + h, y0 + h, x1 - x0 - sw, y1 - y0 - sw,
           f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={sw};")


def tri(x, y, w, h, direction, fill=GREY):
    vertex(x, y, w, h, f"triangle;whiteSpace=wrap;html=1;direction={direction};fillColor={fill};strokeColor=none;")


def poly(points, colour=GREY, sw=6, arc=24):
    """Connector line through points (no arrowheads), rounded corners."""
    if TEXT_ONLY:
        return
    points = [(x, y + OFF[0]) for x, y in points]
    (sx, sy), (tx, ty) = points[0], points[-1]
    mid = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in points[1:-1])
    arr = f'<Array as="points">{mid}</Array>' if mid else ""
    st = (f"edgeStyle=none;html=1;rounded=1;arcSize={arc};endArrow=none;startArrow=none;"
          f"strokeColor={colour};strokeWidth={sw};jumpStyle=none;")
    cells.append(f'<mxCell id="{nid()}" value="" style="{q(st)}" edge="1" parent="1">'
                 f'<mxGeometry relative="1" as="geometry"><mxPoint x="{sx}" y="{sy}" as="sourcePoint"/>'
                 f'<mxPoint x="{tx}" y="{ty}" as="targetPoint"/>{arr}</mxGeometry></mxCell>')


ICONS = json.loads((HERE / "icons.json").read_text())


def icon(name, dx=0, dy=0):
    if TEXT_ONLY:
        return
    e = ICONS[name]
    x0, y0, x1, y1 = e["bbox"]
    gid = nid()
    cells.append(f'<mxCell id="{gid}" value="" style="group;" vertex="1" connectable="0" parent="1">'
                 f'<mxGeometry x="{x0 + dx}" y="{y0 + dy + OFF[0]}" width="{x1 - x0}" height="{y1 - y0}" as="geometry"/></mxCell>')
    for lay in e["layers"]:
        st = lay["style"] + f"fillColor={lay['fill']};strokeColor=none;"
        cells.append(f'<mxCell id="{nid()}" value="" style="{q(st)}" vertex="1" parent="{gid}">'
                     f'<mxGeometry x="0" y="0" width="{x1 - x0}" height="{y1 - y0}" as="geometry"/></mxCell>')


def icon_mapped(src, scale, src_anchor, dst_anchor):
    """Draw icon `src` scaled about src_anchor so that it lands on dst_anchor."""
    if TEXT_ONLY:
        return
    e = ICONS[src]
    x0, y0, x1, y1 = e["bbox"]
    (ax, ay), (bx, by) = src_anchor, dst_anchor
    gx, gy = bx + (x0 - ax) * scale, by + OFF[0] + (y0 - ay) * scale
    gw, gh = (x1 - x0) * scale, (y1 - y0) * scale
    gid = nid()
    cells.append(f'<mxCell id="{gid}" value="" style="group;" vertex="1" connectable="0" parent="1">'
                 f'<mxGeometry x="{gx:.2f}" y="{gy:.2f}" width="{gw:.2f}" height="{gh:.2f}" as="geometry"/></mxCell>')
    for lay in e["layers"]:
        st = lay["style"] + f"fillColor={lay['fill']};strokeColor=none;"
        cells.append(f'<mxCell id="{nid()}" value="" style="{q(st)}" vertex="1" parent="{gid}">'
                     f'<mxGeometry x="0" y="0" width="{gw:.2f}" height="{gh:.2f}" as="geometry"/></mxCell>')


def text_width(s, size, weight=700):
    f = ImageFont.truetype(str(FONT_FILE), max(1, round(size * 4)))
    f.set_variation_by_axes([weight])
    b = f.getbbox(s)
    return (b[2] - b[0]) / 4


def text(key, s, target, size, colour=NAVY_TEXT, align="center", weight=700,
         family=FONT, enlarge=True):
    """Place text so its ink box centre matches the concept's target ink box."""
    x0, y0, x1, y1 = target
    if enlarge:
        size = max(size, MIN_UNITS)
    dx, dy = CALIB.get(key, [0.0, 0.0])
    cy = (y0 + y1) / 2 + dy
    hgt = size * 1.5
    if TEXT_ONLY and colour.upper() == "#FFFFFF":
        colour = "#000000"
    tw = text_width(s, size, weight if family == FONT else 400) + 24
    if align == "center":
        cx = (x0 + x1) / 2 + dx
        gx, gw = cx - tw / 2, tw
    else:
        gx, gw = x0 + dx, tw
    fam = STATIC[weight] if family == FONT else family
    style = (f"text;html=1;whiteSpace=nowrap;overflow=visible;align={align};verticalAlign=middle;"
             f"spacing=0;spacingLeft=0;spacingRight=0;fontFamily={fam};fontSize={size:.2f};"
             f"fontColor={colour};strokeColor=none;fillColor=none;")
    value = f'<span style="font-weight:normal">{escape(s)}</span>'
    vertex(gx, cy - hgt / 2, gw, hgt, style, value)
    TEXTS.append({"key": key, "text": s, "target": list(target), "size": round(size, 2),
                  "pt": round(size * PT_PER_UNIT, 2), "colour": colour, "align": align})


# ================================================================ layout B2
# Same native objects, icons and type sizes as the faithful rebuild; blocks are
# moved vertically (never scaled) so that the three panels share a top edge
# (y~30), a middle line (panel arrows, y~480) and a bottom edge (y=880), with
# even spacing inside each panel. Canvas 1774 x 1000 (6.99 x 3.94 in printed).


def dline(x0, y0, x1, y1, colour, sw, dashed=False, dash="8 6"):
    if TEXT_ONLY:
        return
    st = (f"edgeStyle=none;html=1;endArrow=none;startArrow=none;strokeColor={colour};strokeWidth={sw};"
          + (f"dashed=1;fixDash=1;dashPattern={dash};" if dashed else ""))
    cells.append(f'<mxCell id="{nid()}" value="" style="{q(st)}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                 f'<mxPoint x="{x0}" y="{y0 + OFF[0]}" as="sourcePoint"/><mxPoint x="{x1}" y="{y1 + OFF[0]}" as="targetPoint"/>'
                 f'</mxGeometry></mxCell>')


F1RF = (0.659465, 0.602548, 0.718376)
F1CNN = (0.668293, 0.608232, 0.729400)
DAL = (0.008828, -0.015743, 0.035571)        # CNN - RF, paired 95% interval (saved fits)
AX = "#4D4D4D"
LAB_Y = (930, 970)

# ---------------------------------------------------------------- canvas
OFF[0] = 0
vertex(0, 0, W, H, "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=none;")
ARROW_DY = 55
for x in (589, 1183):
    rect(x, 13, 4, 401 + ARROW_DY - 2 - 13, DIVIDER)
    rect(x, 448 + ARROW_DY + 3, 4, 920 - (448 + ARROW_DY + 3), DIVIDER)
OFF[0] = ARROW_DY
icon("bigarrow_ab")
icon("bigarrow_bc")

# ================================================================ panel (a)
OFF[0] = 0
icon("doc_stack")
icon("plus")
icon("brain_doc")
text("added", "Added imaging", (36, 214, 247, 245), 32.7)
rect(148, 252, 6, 76, GREY)
tri(138, 326, 26, 24, "south")

OFF[0] = 60
icon("people_a")
text("different", "Different patients", (42, 401, 274, 431), 30.9)
OFF[0] = 0
rect(148, 496, 6, 86, GREY)
tri(138, 580, 26, 24, "south")

OFF[0] = 130
TA_X0, TA_X1, TA_Y0, TA_Y1 = 4, 310, 474, 709
C1, C2 = 74, 146
rect(TA_X0, TA_Y0, TA_X1 - TA_X0, TA_Y1 - TA_Y0, "#FFFFFF", arc=16)
rect(TA_X0, TA_Y0, TA_X1 - TA_X0, 50, NAVY, arc=16)
rect(TA_X0, TA_Y0 + 26, TA_X1 - TA_X0, 24, NAVY)
rect(TA_X0, 524, TA_X1 - TA_X0, 38, HEADER_GREY)
for y in (560, 596, 632, 668):
    rect(TA_X0, y, TA_X1 - TA_X0, 3, "#B1B8C3")
for x in (C1, C2):
    rect(x - 1.5, 524, 3, TA_Y1 - 524, "#B1B8C3")
cid, cfold, cpred = (TA_X0 + C1) / 2, (C1 + C2) / 2, (C2 + TA_X1) / 2
for yc in (580, 616, 652):
    ellipse(cid - 10, yc - 10, cid + 10, yc + 10, "#626F7F", "none", 0)
    ellipse(cpred - 10, yc - 10, cpred + 10, yc + 10, "#626F7F", "none", 0)
    rect(cfold - 9, yc - 2, 18, 4, "#354354")
for xc in (cid, cfold, cpred):
    for k in (-9, 0, 9):
        ellipse(xc + k - 2.5, 688.5, xc + k + 2.5, 693.5, "#1E2F3B", "none", 0)
framed(TA_X0, TA_Y0, TA_X1, TA_Y1, "none", "#1E2F3B", 3, 16)
text("saved_rf", "Saved RF predictions", (TA_X0 + 4, 492, TA_X0 + 221, 515), 27.4,
     colour="#FFFFFF", align="left")
icon("lock_a", dx=TA_X1 - 299)
text("hdr_id_a", "ID", (cid - 9, 535, cid + 9, 552), 22)
text("hdr_fold_a", "Fold", (cfold - 20, 534, cfold + 20, 552), 22)
text("hdr_pred_a", "Prediction", (cpred - 50, 534, cpred + 50, 552), 22)
text("unchanged", "Unchanged", (89, 721, 243, 751), 32)

# right column: three score boxes spread over the panel height
R1, R2, R3 = 60, 97, 134
OFF[0] = 0
SP = 323
poly([(SP, 583), (SP, 304 + R1), (346, 304 + R1)])
poly([(SP, 583), (SP, 680 + R3), (346, 680 + R3)])
poly([(SP, 583), (346, 583)])
poly([(TA_X1, 581 + 130), (SP, 581 + 130)])
for yc in (304 + R1, 583, 680 + R3):
    tri(346, yc - 13, 22, 26, "east")
OFF[0] = R1
text("mean_r", "Mean Pearson r", (364, 197, 569, 219), 31.2)
framed(367, 234, 566, 371, "#FFFFFF", RED, 4, 24)
text("pub", "Published list", (386, 253, 548, 275), 27.9)
text("n172", "172", (445, 286, 492, 309), 33, weight=800)
icon("arrow_up")
text("r756", "0.756", (442, 324, 531, 351), 40, colour=RED, weight=800)
OFF[0] = R2
framed(368, 418, 566, 555, LIGHTBLUE, NAVY, 4, 24)
text("all", "All", (449, 435, 487, 461), 34.2)
text("n226", "226", (443, 471, 492, 495), 34, weight=800)
text("r712", "0.712", (423, 507, 512, 534), 40, weight=800)
OFF[0] = R3
framed(369, 608, 567, 746, "#FFFFFF", "none", 0, 24)
for xa, xb in [(381, 395), (401, 415), (420, 434), (440, 453), (459, 472), (477, 490),
               (495, 508), (513, 527), (532, 546)]:
    rect(xa, 608, xb - xa, 4, TEAL)
for xa, xb in [(391, 406), (411, 425), (431, 446), (451, 466), (471, 486), (491, 506),
               (511, 525), (531, 544)]:
    rect(xa, 742, xb - xa, 4, TEAL)
for ya, yb in [(630, 643), (649, 662), (667, 681), (686, 699), (704, 717), (723, 736)]:
    rect(369, ya, 4, yb - ya, TEAL)
for ya, yb in [(621, 634), (639, 652), (657, 671), (676, 689), (695, 708), (714, 727)]:
    rect(563, ya, 4, yb - ya, TEAL)
poly([(371, 625), (371, 610), (378, 610)], colour=TEAL, sw=4, arc=16)
poly([(551, 610), (565, 610), (565, 617)], colour=TEAL, sw=4, arc=16)
poly([(371, 740), (371, 744), (386, 744)], colour=TEAL, sw=4, arc=16)
poly([(549, 744), (565, 744), (565, 732)], colour=TEAL, sw=4, arc=16)
text("flair", "FLAIR", (429, 628, 509, 651), 32.8)
text("n135", "135", (444, 662, 492, 685), 33, weight=800)
icon("arrow_down")
text("r625", "0.625", (444, 699, 532, 726), 40, colour=TEAL_TEXT, weight=800)
OFF[0] = 0
text("lab_a", "(a)", (271, LAB_Y[0], 319, LAB_Y[1]), 31, colour="#000000", weight=400,
     family="Times New Roman")

# ================================================================ panel (b)
OFF[0] = 0
icon("doc_people")
icon_mapped("people_c", 1.29, (1438.5, 706), (799.8, 153))
text("same173", "Same 173 patients", (879, 97, 1154, 129), 34.9, align="left")
text("samefolds", "Same folds", (879, 134, 1045, 160), 35.1, align="left")
rect(907, 173, 4, 83, GREY)
poly([(772, 274), (772, 254), (1015, 254), (1015, 274)], sw=5, arc=10)
tri(761, 272, 22, 22, "south")
tri(1004, 272, 22, 22, "south")
OFF[0] = 45
icon("gear_rf")
icon("gear_cnn")
text("devrf", "Developed RF", (682, 373, 863, 403), 30.8)
text("cnn_b", "CNN", (987, 379, 1044, 401), 30.5)
OFF[0] = 0
rect(770, 452, 5, 34, GREY)
tri(762, 484, 22, 22, "south")
OFF[0] = 70
framed(655, 436, 915, 565, LIGHTBLUE, NAVY, 4, 24)
text("thresh", "Threshold aligned", (680, 449, 890, 473), 27.1)
ellipse(671, 478, 757, 558, "#FFFFFF", RED, 4)
text("t448", "0.448", (680, 507, 748, 529), 30.5, weight=800)
rect(766, 515, 25, 5, NAVY_DARK)
tri(789, 507, 22, 21, "east", NAVY_DARK)
ellipse(811, 477, 897, 557, "#FFFFFF", NAVY, 4)
text("t659", "0.659", (820, 507, 886, 529), 30.5, weight=800)
OFF[0] = 0
rect(770, 637, 5, 10, GREY)
tri(759, 646, 27, 22, "south")
rect(1013, 452, 4, 195, GREY)
tri(1005, 646, 21, 22, "south")

# F1 box: short bars on a 0.63-0.70 window (axis break shown), the RF level as the
# no-gain line, and the paired 95% interval of CNN - RF drawn from that line
framed(650, 668, 1125, 880, LIGHTBLUE, NAVY, 4, 28)
lo_v, hi_v, base, top = 0.63, 0.70, 836, 716
yv = lambda v_: base - (v_ - lo_v) / (hi_v - lo_v) * (base - top)
ax_x = 726
dline(ax_x, top - 8, ax_x, base, AX, 2.5)
for t in (0.65, 0.70):
    dline(ax_x - 8, yv(t), ax_x, yv(t), AX, 2.5)
    lab = f"{t:.2f}"
    w = text_width(lab, MIN_UNITS) + 4
    text(f"yt{t}", lab, (ax_x - 13 - w, yv(t) - 10, ax_x - 13, yv(t) + 10), MIN_UNITS, align="left")
text("f1e", "F1", (ax_x + 8, top - 26, ax_x + 44, top - 4), 31, weight=800, align="left")
bw = 106
bars = ((812, F1RF[0], NAVY, NAVY_TEXT, "RF"), (970, F1CNN[0], TEAL, TEAL_TEXT, "CNN"))
for cx, e, col, tcol, nm in bars:
    rect(cx - bw / 2, yv(e), bw, base - yv(e), col)
    lab = f"{nm} {e:.3f}"
    w = text_width(lab, 31) + 4
    text(f"lab{nm}", lab, (cx - w / 2, base + 9, cx + w / 2, base + 29), 31, colour=tcol)
for cx, half in ((ax_x, 7), (812, bw / 2 + 3), (970, bw / 2 + 3)):
    for dy in (0, 8):
        dline(cx - half, base - 9 - dy + 4, cx + half, base - 9 - dy - 4, LIGHTBLUE, 3)
yrf = yv(F1RF[0])
dline(812 - bw / 2, yrf, 1040, yrf, "#1A1A1A", 3, dashed=True)
w = text_width("= RF", MIN_UNITS) + 4
text("rf_lvl", "= RF", (1046, yrf - 10, 1046 + w, yrf + 10), MIN_UNITS, align="left")
clo, chi = F1RF[0] + DAL[1], F1RF[0] + DAL[2]
dline(970, yv(chi), 970, yv(clo), "#0B1535", 5)
dline(970 - 18, yv(chi), 970 + 18, yv(chi), "#0B1535", 5)
dline(970 - 18, yv(clo), 970 + 18, yv(clo), "#0B1535", 5)
w = text_width("95% CI", MIN_UNITS) + 4
text("ci_lab", "95% CI", (996, yv(chi) - 10, 996 + w, yv(chi) + 10), MIN_UNITS, align="left")
text("lab_b", "(b)", (863, LAB_Y[0], 913, LAB_Y[1]), 31, colour="#000000", weight=400,
     family="Times New Roman")

# ================================================================ panel (c)
OFF[0] = 0
icon("database")
text("bank", "150-entry bank", (1215, 249, 1418, 279), 31.1)
text("cand", "Candidate records", (1456, 22, 1739, 49), 36.2)
TC_X0, TC_X1, TC_Y0, TC_Y1 = 1452, 1746, 61, 247
D1, D2 = 1515, 1585
rect(TC_X0, TC_Y0, TC_X1 - TC_X0, TC_Y1 - TC_Y0, "#FFFFFF", arc=16)
rect(TC_X0, TC_Y0, TC_X1 - TC_X0, 42, HEADER_GREY, arc=16)
rect(TC_X0, TC_Y0 + 20, TC_X1 - TC_X0, 22, HEADER_GREY)
for y in (102, 138, 174, 210):
    rect(TC_X0, y, TC_X1 - TC_X0, 3, "#CDD2DA")
for x in (D1, D2):
    rect(x - 1.5, TC_Y0, 3, TC_Y1 - TC_Y0, "#CDD2DA")
did, dfold, dpred = (TC_X0 + D1) / 2, (D1 + D2) / 2, (D2 + TC_X1) / 2
for yc in (121, 157, 193):
    ellipse(did - 10, yc - 10, did + 10, yc + 10, "#626F7F", "none", 0)
    ellipse(dpred - 10, yc - 10, dpred + 10, yc + 10, "#626F7F", "none", 0)
    rect(dfold - 9, yc - 2, 18, 4, "#354354")
for xc in (did, dfold, dpred):
    for k in (-9, 0, 9):
        ellipse(xc + k - 2.5, 227.5, xc + k + 2.5, 232.5, "#253546", "none", 0)
framed(TC_X0, TC_Y0, TC_X1, TC_Y1, "none", "#253546", 3, 16)
text("hdr_id_c", "ID", (did - 9, 74, did + 9, 92), 22)
text("hdr_fold_c", "Fold", (dfold - 20, 73, dfold + 20, 92), 22)
text("hdr_pred_c", "Prediction", (dpred - 50, 73, dpred + 50, 92), 22)
text("ids", "IDs · folds · predictions", (1465, 257, 1752, 285), 29.6)
poly([(1306, 288), (1306, 384), (1647, 384), (1647, 288)], sw=6, arc=24)
rect(1463, 384, 6, 86, GREY)
tri(1452, 468, 28, 22, "south")
OFF[0] = 141
icon("big_doc_c")
poly([(1281, 476), (1281, 421), (1375, 421)], sw=6, arc=24)
poly([(1281, 632), (1281, 706), (1375, 706)], sw=6, arc=24)
rect(1340, 421, 6, 285, GREY)
rect(1346, 511, 29, 6, GREY)
rect(1346, 604, 29, 6, GREY)
for yc in (421, 514, 607, 706):
    tri(1375, yc - 11, 22, 22, "east")
icon("doc_scoring")
icon("gear_train")
icon("docs_method")
icon("people_c")
for y0, y1 in ((396, 460), (486, 552), (578, 645), (672, 739)):
    framed(1484, y0, 1751, y1, LABEL_FILL, LABEL_LINE, 4, 20)
text("scoring", "Scoring change", (1500, 413, 1702, 443), 30.8, align="left")
text("training", "Training change", (1502, 505, 1713, 535), 30.9, align="left")
text("method", "Method difference", (1505, 601, 1735, 624), 29.5, align="left")
text("coverage", "Coverage", (1509, 693, 1637, 723), 32.1, align="left")
OFF[0] = 0
text("lab_c", "(c)", (1459, LAB_Y[0], 1505, LAB_Y[1]), 31, colour="#000000", weight=400,
     family="Times New Roman")

# ---------------------------------------------------------------- write
xml = ('<mxfile host="drawio" type="device"><diagram id="figure1B2" name="Figure 1">'
       f'<mxGraphModel dx="{W}" dy="{H}" grid="0" gridSize="10" guides="1" tooltips="1" connect="0" '
       f'arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">'
       '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) +
       '</root></mxGraphModel></diagram></mxfile>')
out = HERE / "Figure1_B2.drawio"
out.write_text(xml, encoding="utf-8")
small = [t for t in TEXTS if t["pt"] < 8.0 and t["key"] not in ("lab_a", "lab_b", "lab_c")]
print("wrote", out, "| cells", _n[0], "| texts", len(TEXTS), "| <8pt:", [t["key"] for t in small])
