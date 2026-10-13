# Figure 1 source

`Figure1_B2.drawio` is the editable source of Figure 1. Every box, connector,
arrowhead, table, chart element and label is a native draw.io object; icons are
draw.io stencil shapes (vector outlines) stored in `icons.json`.

Rebuild and export:

```sh
python build_figure1_B2.py                      # writes Figure1_B2.drawio
drawio --export --format pdf --crop --output Figure1_B2.pdf Figure1_B2.drawio
```

- Canvas: 1774 x 1000 units, placed at the 6.99-in text width (0.2837 pt per unit);
  all labels print at 8 pt or larger.
- Panel (b) chart: severe-class F1 of the developed random forest (0.659) and the
  CNN (0.668) on a truncated 0.63-0.70 axis; the dashed line is the RF level and
  the error bar is the paired 95% interval of CNN minus RF (0.009,
  [-0.016, 0.036]) drawn from that line. Values come from the image-control
  results in the supplementary files (`paired-contrasts.csv`,
  `summary-intervals.csv`).
- Text uses Source Sans 3 (SIL Open Font License 1.1, `fonts/`). The draw.io PDF
  export embeds variable fonts as Type 3, so `make_static_fonts.py` creates
  static instances ("SS3 Fig Bold", "SS3 Fig ExtraBold") that embed as TrueType;
  install them before exporting.
- `trace_icons.py` documents how the icon stencils were vectorised (potrace,
  per colour layer) from the concept sketch used for the layout; the sketch
  itself is not included, and `icons.json` holds the resulting stencils.
