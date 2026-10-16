"""
Draw the readout-pattern diagram in the ramp fitting docs (``_includes/readout_pattern.html``).

The diagram is an inline SVG that follows the docs' light/dark theme. Run from the repository
root: ``python docs/source/_scripts/readout_pattern_html.py``.
"""

from pathlib import Path

import numpy as np

from oircal.utils.read_times import get_delta_read_times, get_first_read_mjd


OUT = Path(__file__).parents[1] / "_includes" / "readout_pattern.html"

# Small detector so every pixel can be labeled; 1 Hz clock so times are in clock ticks
Nreads, Nrows, Ncols, Nchannels = 4, 4, 9, 3
clock_rate = 1.0
Ncols_chan = Ncols // Nchannels

read_times = get_delta_read_times(Nreads, Nrows, Ncols, clock_rate, Nchannels)
first_read = get_first_read_mjd(0.0, Nrows, Ncols, clock_rate, Nchannels) * 86400.0
ticks = np.rint(first_read).astype(int)
period = int(round(read_times[1]))
mid_offset = float(np.mean(read_times))
T_MAX = Nreads * period

# Highlighted readout ticks (series 1-3), drawn in the same color in every panel
examples = [0, 4, 11]

# ---- Layout (SVG user units; the docs column renders about 0.7 px per unit) ----
W, H = 1000, 368
TITLE_Y, SUB_Y = 22, 46  # title, first line of the description (second line 20 below)
PLOT_Y0, PLOT_Y1 = 116, 260  # top and bottom of the grid and plots
TICK_Y, AXIS_Y = PLOT_Y1 + 20, PLOT_Y1 + 44  # tick labels, axis titles
CELL, GAP = 36, 8  # pixel cell size, gap between channels
LEFT_X = 8  # left edge of the titles
GX = 42  # left of the pixel grid
P2_X, PX0, PX1 = 396, 490, 718  # reads panel: title, time axis
P3_X, SX0, SX1 = 778, 806, 990  # signal panel: title, time axis
ROW_Y = [164, 200, 236]  # rows of the reads panel


def cell_x(c: int) -> float:
    return GX + c * CELL + (c // Ncols_chan) * GAP


def time_x(t: float, x0: float = PX0, x1: float = PX1) -> float:
    return x0 + (x1 - x0) * t / T_MAX


def signal_x(t: float) -> float:
    return time_x(t, SX0, SX1)


def signal_y(signal: float) -> float:
    # Signal accumulated since the pixel's first read, up to the last read
    return PLOT_Y1 - (PLOT_Y1 - PLOT_Y0 - 22) * signal / read_times[-1]


def channel_reversed(k: int) -> bool:
    # Read from the timing itself: a reversed channel starts at its right edge
    cols = ticks[0, k * Ncols_chan : (k + 1) * Ncols_chan]
    return cols[-1] < cols[0]


def description(x: float, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        add(text(x, SUB_Y + 20 * i, line, "rp-sub"))


def text(x: float, y: float, s: str, cls: str, anchor: str = "start", extra: str = "") -> str:
    return (
        f'<text class="{cls}" x="{x:g}" y="{y:g}" text-anchor="{anchor}" '
        f'dominant-baseline="central"{extra}>{s}</text>'
    )


def double_arrow(x0: float, x1: float, y: float) -> str:
    return (
        f'<line class="rp-dir" x1="{x0 + 1:g}" y1="{y:g}" x2="{x1 - 1:g}" y2="{y:g}" '
        'marker-start="url(#rp-arrow)" marker-end="url(#rp-arrow)"/>'
    )


def diamond(x: float, y: float, size: float, cls: str) -> str:
    h = size / 2
    return (
        f'<rect class="{cls}" x="{x - h:g}" y="{y - h:g}" width="{size:g}" height="{size:g}" '
        f'transform="rotate(45 {x:g} {y:g})"/>'
    )


svg = []
add = svg.append

# ---- Left: first read of each pixel ----
add(text(LEFT_X, TITLE_Y, "get_first_read_mjd", "rp-title"))
description(
    LEFT_X,
    ["MJD of each pixel's first read, (Nrows, Ncols)", "here in clock ticks after mjd_start"],
)

for k in range(Nchannels):
    x0 = cell_x(k * Ncols_chan)
    arrow = "←" if channel_reversed(k) else "→"
    add(
        text(
            x0 + Ncols_chan * CELL / 2,
            PLOT_Y0 - 16,
            f"channel {k + 1} {arrow}",
            "rp-label",
            "middle",
        ),
    )

for r in range(Nrows):
    y = PLOT_Y0 + r * CELL
    add(text(GX - 9, y + CELL / 2, str(r), "rp-axis", "end"))
    for c in range(Ncols):
        x = cell_x(c)
        tick = ticks[r, c]
        shade = 0.04 + 0.2 * tick / ticks.max()
        cls = f"rp-cell rp-hover rp-s{examples.index(tick) + 1}" if tick in examples else "rp-cell"
        add(
            f'<g class="{cls}" data-tick="{tick}" data-row="{r}" data-col="{c}">'
            f'<rect x="{x + 2}" y="{y + 2}" width="{CELL - 4}" height="{CELL - 4}" rx="6" '
            f'style="--shade:{shade:.3f}"/>'
            + text(x + CELL / 2, y + CELL / 2 - 3, str(tick), "rp-num", "middle")
            + "</g>",
        )
    # Readout direction along each row of each channel
    for k in range(Nchannels):
        x0 = cell_x(k * Ncols_chan) + 8
        x1 = cell_x((k + 1) * Ncols_chan - 1) + CELL - 8
        if channel_reversed(k):
            x0, x1 = x1, x0
        add(
            f'<line class="rp-dir" x1="{x0}" y1="{y + CELL - 7}" x2="{x1}" y2="{y + CELL - 7}" '
            'marker-end="url(#rp-arrow)"/>',
        )
for c in range(Ncols):
    add(text(cell_x(c) + CELL / 2, TICK_Y, str(c), "rp-axis", "middle"))
add(text((cell_x(0) + cell_x(Ncols - 1) + CELL) / 2, AXIS_Y, "column", "rp-axis", "middle"))
yc = PLOT_Y0 + Nrows * CELL / 2
add(text(LEFT_X + 6, yc, "row", "rp-axis", "middle", f' transform="rotate(-90 {LEFT_X + 6} {yc})"'))

# ---- Middle: every read of the highlighted pixels ----
add(text(P2_X, TITLE_Y, "get_delta_read_times", "rp-title"))
description(
    P2_X,
    [
        "time of each read after the first, (Nreads,)",
        f"same for every pixel: a read every {period} ticks",
    ],
)

for k in range(Nreads):
    x0, x1 = time_x(k * period), time_x((k + 1) * period)
    if k % 2 == 0:
        add(
            f'<rect class="rp-band" x="{x0:g}" y="{PLOT_Y0}" width="{x1 - x0:g}" '
            f'height="{PLOT_Y1 - PLOT_Y0}" rx="6"/>',
        )
    add(text((x0 + x1) / 2, PLOT_Y0 + 18, f"read {k + 1}", "rp-axis", "middle"))

for i, (tick, y) in enumerate(zip(examples, ROW_Y, strict=True)):
    times = tick + read_times
    add(f'<g class="rp-row rp-s{i + 1}" data-tick="{tick}">')
    add(text(PX0 - 10, y, f"pixels #{tick}", "rp-label", "end"))
    add(
        f'<line class="rp-track" x1="{time_x(times[0]):g}" y1="{y}" '
        f'x2="{time_x(times[-1]):g}" y2="{y}"/>',
    )
    for t in times:
        add(f'<circle class="rp-read" cx="{time_x(t):g}" cy="{y}" r="6"/>')
    add(diamond(time_x(tick + mid_offset), y, 11, "rp-mid"))
    add("</g>")

add(f'<line class="rp-baseline" x1="{PX0}" y1="{PLOT_Y1}" x2="{PX1}" y2="{PLOT_Y1}"/>')
for t in range(0, T_MAX + 1, period):
    add(text(time_x(t), TICK_Y, str(t), "rp-axis", "middle"))
add(text((PX0 + PX1) / 2, AXIS_Y, "time (clock ticks)", "rp-axis", "middle"))

# Read period: Nrows x columns per channel / clock_rate
y = AXIS_Y + 26
add(double_arrow(time_x(0), time_x(period), y))
add(
    text(
        time_x(period) + 8,
        y,
        f"{period} = {Nrows} × {Ncols_chan} pixels / clock_rate",
        "rp-sub",
    ),
)

# ---- Right: signal vs. time of the highlighted pixels ----
add(text(P3_X, TITLE_Y, "signal vs. time", "rp-title"))
description(P3_X, ["constant source: same slope,", "lagged by each first read"])

add(f'<line class="rp-baseline" x1="{SX0}" y1="{PLOT_Y1}" x2="{SX1}" y2="{PLOT_Y1}"/>')
add(f'<line class="rp-baseline" x1="{SX0}" y1="{PLOT_Y0}" x2="{SX0}" y2="{PLOT_Y1}"/>')
for t in range(0, T_MAX + 1, period):
    add(text(signal_x(t), TICK_Y, str(t), "rp-axis", "middle"))
add(text((SX0 + SX1) / 2, AXIS_Y, "time (clock ticks)", "rp-axis", "middle"))
yc = (PLOT_Y0 + PLOT_Y1) / 2
add(text(SX0 - 12, yc, "signal", "rp-axis", "middle", f' transform="rotate(-90 {SX0 - 12} {yc})"'))

for i, tick in enumerate(examples):
    times = tick + read_times
    add(f'<g class="rp-row rp-s{i + 1}" data-tick="{tick}">')
    add(
        f'<line class="rp-track" x1="{signal_x(times[0]):g}" y1="{signal_y(0):g}" '
        f'x2="{signal_x(times[-1]):g}" y2="{signal_y(read_times[-1]):g}"/>',
    )
    for t, sig in zip(times, read_times, strict=True):
        add(f'<circle class="rp-read" cx="{signal_x(t):g}" cy="{signal_y(sig):g}" r="5"/>')
    add(diamond(signal_x(tick + mid_offset), signal_y(mid_offset), 9, "rp-mid"))
    add("</g>")

# Lag between the first and last highlighted pixels, at the last read
y = signal_y(read_times[-1]) - 12
x0 = signal_x(examples[0] + read_times[-1])
x1 = signal_x(examples[-1] + read_times[-1])
add(double_arrow(x0, x1, y))
add(text(x1, y - 16, f"{examples[-1] - examples[0]} tick lag", "rp-sub", "end"))

# ---- Bottom row: hover hint and legend ----
add(
    text(
        LEFT_X,
        H - 12,
        "Hover a colored pixel to see when it is read.",
        "rp-hint",
        extra=' id="rp-info"',
    ),
)
lx = W - 190
add(f'<circle class="rp-legend" cx="{lx}" cy="{H - 12}" r="6"/>')
add(text(lx + 12, H - 12, "read", "rp-axis"))
add(diamond(lx + 74, H - 12, 10, "rp-legend rp-legend-mid"))
add(text(lx + 86, H - 12, "mid_time", "rp-axis"))

STYLE = """
.readout-diagram {
  --rp-ink: hsl(var(--foreground, 222.2 47.4% 11.2%));
  --rp-muted: hsl(var(--muted-foreground, 215.4 16.3% 46.9%));
  --rp-bg: hsl(var(--background, 0 0% 100%));
  --rp-s1: #2a78d6; --rp-s2: #eb6834; --rp-s3: #1baf7a;
  margin: 1.5rem 0 1rem;
  color: var(--rp-ink);
}
html.dark .readout-diagram { --rp-s1: #3987e5; --rp-s2: #d95926; --rp-s3: #199e70; }
.readout-diagram svg { width: 100%; height: auto; display: block; overflow: visible; }
.readout-diagram text { font-family: inherit; fill: var(--rp-ink); }
.readout-diagram .rp-title {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 19px; font-weight: 600;
}
.readout-diagram .rp-sub, .readout-diagram .rp-axis, .readout-diagram .rp-hint {
  fill: var(--rp-muted); font-size: 15.5px;
}
.readout-diagram .rp-label { font-size: 16px; font-weight: 500; }
.readout-diagram .rp-hint { font-style: italic; }
.readout-diagram .rp-num { font-size: 16px; font-variant-numeric: tabular-nums; }
.readout-diagram .rp-cell rect {
  fill: var(--rp-ink); fill-opacity: var(--shade); stroke: transparent; stroke-width: 2.5;
  transition: fill-opacity .15s, stroke .15s;
}
.readout-diagram .rp-s1 { --rp-c: var(--rp-s1); }
.readout-diagram .rp-s2 { --rp-c: var(--rp-s2); }
.readout-diagram .rp-s3 { --rp-c: var(--rp-s3); }
.readout-diagram .rp-cell.rp-s1 rect, .readout-diagram .rp-cell.rp-s2 rect,
.readout-diagram .rp-cell.rp-s3 rect { stroke: var(--rp-c); fill: var(--rp-c); fill-opacity: .14; }
.readout-diagram .rp-cell.rp-hl rect { fill-opacity: .3; }
.readout-diagram .rp-hover { cursor: pointer; }
.readout-diagram .rp-dir {
  stroke: var(--rp-muted); stroke-width: 1.3; stroke-opacity: .75; pointer-events: none;
}
.readout-diagram #rp-arrow path { fill: var(--rp-muted); }
.readout-diagram .rp-band { fill: var(--rp-ink); fill-opacity: .045; }
.readout-diagram .rp-baseline { stroke: var(--rp-muted); stroke-opacity: .5; }
.readout-diagram .rp-track { stroke: var(--rp-c); stroke-width: 2.5; stroke-linecap: round; }
.readout-diagram .rp-read { fill: var(--rp-c); stroke: var(--rp-bg); stroke-width: 2; }
.readout-diagram .rp-mid { fill: var(--rp-bg); stroke: var(--rp-c); stroke-width: 2.2; }
.readout-diagram .rp-row { transition: opacity .15s; }
.readout-diagram.rp-focus .rp-row:not(.rp-hl) { opacity: .2; }
.readout-diagram .rp-legend { fill: var(--rp-muted); }
.readout-diagram .rp-legend-mid { fill: var(--rp-bg); stroke: var(--rp-muted); stroke-width: 1.8; }
"""

SCRIPT = f"""
(() => {{
  const root = document.currentScript.previousElementSibling;
  const info = root.querySelector("#rp-info");
  const hint = info.textContent;
  const period = {period}, Nreads = {Nreads}, mid = {mid_offset:g};
  root.querySelectorAll(".rp-hover").forEach((cell) => {{
    cell.addEventListener("pointerenter", () => {{
      const t = Number(cell.dataset.tick);
      root.querySelectorAll(`[data-tick="${{t}}"]`).forEach((el) => el.classList.add("rp-hl"));
      root.classList.toggle("rp-focus", !!root.querySelector(`.rp-row[data-tick="${{t}}"]`));
      const reads = Array.from({{length: Nreads}}, (_, k) => t + k * period).join(", ");
      info.textContent = `Pixel (${{cell.dataset.row}}, ${{cell.dataset.col}}): reads at ticks ` +
        `${{reads}}; mid_time at ${{t + mid}}. One pixel per channel shares these times.`;
    }});
    cell.addEventListener("pointerleave", () => {{
      root.querySelectorAll(".rp-hl").forEach((el) => el.classList.remove("rp-hl"));
      root.classList.remove("rp-focus");
      info.textContent = hint;
    }});
  }});
}})();
"""

html = f"""<style>{STYLE}</style>
<div class="readout-diagram">
<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="rp-desc">
<desc id="rp-desc">Readout of a {Nrows} by {Ncols} pixel detector with {Nchannels} vertical \
channels. Each channel reads one pixel per clock tick, row by row, with every other channel \
reading right to left. Every read of a pixel follows its first read by a multiple of \
{period} ticks, so the ramps of pixels read at different ticks lag each other.</desc>
<defs><marker id="rp-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" \
markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z"/></marker></defs>
{chr(10).join(svg)}
</svg>
</div>
<script>{SCRIPT}</script>
"""

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(html)
print(f"Wrote {OUT}")
