"""Instantiate static Source Sans 3 faces (SIL OFL 1.1) for the figure.

draw.io's PDF export embeds variable fonts as Type 3; static instances embed as
TrueType. Each weight becomes its own single-face family so the figure can
reference it without synthetic bolding.
"""
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

SRC = Path(__file__).resolve().parent / "fonts/SourceSans3[wght].ttf"
OUT = Path(__file__).resolve().parent / "fonts"
for weight, fam in ((700, "SS3 Fig Bold"), (800, "SS3 Fig ExtraBold")):
    f = TTFont(SRC)
    inst = instancer.instantiateVariableFont(f, {"wght": weight})
    name = inst["name"]
    ps = fam.replace(" ", "")
    for rec in list(name.names):
        if rec.nameID in (16, 17, 21, 22, 25):
            name.removeNames(nameID=rec.nameID)
    name.setName(fam, 1, 3, 1, 0x409); name.setName("Regular", 2, 3, 1, 0x409)
    name.setName(f"{ps}-Regular", 6, 3, 1, 0x409); name.setName(f"{fam} Regular", 4, 3, 1, 0x409)
    name.setName(f"{ps};static instance wght={weight}", 3, 3, 1, 0x409)
    inst["OS/2"].usWeightClass = 400
    inst["OS/2"].fsSelection = (inst["OS/2"].fsSelection & ~0b1100001) | 0b1000000  # REGULAR
    inst["head"].macStyle = 0
    out = OUT / f"{ps}-Regular.ttf"
    inst.save(out)
    print("wrote", out)
