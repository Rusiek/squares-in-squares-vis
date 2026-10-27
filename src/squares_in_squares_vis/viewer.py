# Copyright 2026 the squares-in-squares-vis authors
# SPDX-License-Identifier: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)
"""Packing viewer for the Kaggle competition "Squares in Squares".

PackingViewer(src, dst).build() writes one self-contained HTML page with every packing of a submission embedded as JSON;
.show(n) draws one packing inline in a notebook, .show_scores() plots score_n over n.
It does NOT check overlaps or the submission format in depth: use the official metric notebook for that.
L follows the metric's measure_side_mp (max of the x- and y-extent of all corners), but in float64, not mpmath.
"""
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_SRC = Path("/kaggle/input/squares-in-squares/sample_submission.csv")
KAGGLE_DST = Path("/kaggle/working/packings.html")
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
AXIS_TOL = 1e-9  # |angle mod pi/2| at most this: axis-aligned
NEAR_TOL = 1e-3  # below this (but above AXIS_TOL): nearly aligned
MAX_INLINE_MB = 15.0  # plotly_js="auto": embed plotly.js only while the page stays under this
COLORS = {"aligned": "#2a78d6", "near": "#1baf7a", "rotated": "#eb6834", "box": "#0b0b0b"}


def default_src() -> Path:
    """The competition's sample_submission.csv on Kaggle (any mount point), else ./sample_submission.csv."""
    if KAGGLE_SRC.exists():
        return KAGGLE_SRC
    found = sorted(Path("/kaggle/input").glob("**/sample_submission.csv")) if Path("/kaggle/input").is_dir() else []
    return found[0] if found else Path("sample_submission.csv")


def default_dst() -> Path:
    return KAGGLE_DST if KAGGLE_DST.parent.is_dir() else Path("packings.html")


def load_submission(src) -> dict[int, np.ndarray]:
    """n -> (n, 3) float array of (x, y, angle) ordered by square_id, from a submission CSV (angle in radians)."""
    exact = {} if int(pd.__version__.split(".")[0]) >= 3 else {"float_precision": "round_trip"}  # pandas 3: exact
    df = pd.read_csv(src, usecols=["n", "square_id", "x", "y", "angle"], **exact)
    df = df.sort_values(["n", "square_id"], kind="stable")
    out = {}
    for n, g in df.groupby("n", sort=True):
        n = int(n)
        if not 1 <= n <= 1000 or g["square_id"].tolist() != list(range(n)):
            warnings.warn(f"n={n}: expected square_id 0..{n - 1}, got {len(g)} rows; drawn as given")
        out[n] = g[["x", "y", "angle"]].to_numpy(dtype=float)
    return out


def corners(rows: np.ndarray) -> np.ndarray:
    """(n, 4, 2) corners of unit squares centred at (x, y) rotated by angle; same order as the official metric."""
    x, y, a = rows[:, 0], rows[:, 1], rows[:, 2]
    c, s = np.cos(a), np.sin(a)
    dx1, dy1 = 0.5 * c - 0.5 * s, 0.5 * s + 0.5 * c
    dx2, dy2 = -0.5 * c - 0.5 * s, -0.5 * s + 0.5 * c
    cx = np.stack([x + dx1, x + dx2, x - dx1, x - dx2], axis=1)
    cy = np.stack([y + dy1, y + dy2, y - dy1, y - dy2], axis=1)
    return np.stack([cx, cy], axis=2)


def side_length(rows: np.ndarray) -> float:
    """Bounding side L = max(x-extent, y-extent) over all corners (float64 version of the metric's L)."""
    c = corners(rows)
    return float(max(np.ptp(c[..., 0]), np.ptp(c[..., 1])))


def score_n(n: int, L: float) -> float:
    return L / math.sqrt(n) - 1


def grid_score(n: int) -> float:
    """score_n of the ceil(sqrt(n)) grid: what the metric gives a missing or invalid n."""
    return score_n(n, math.isqrt(n - 1) + 1)


def axis_dev(a: np.ndarray) -> np.ndarray:
    """Distance of each angle from the nearest multiple of pi/2."""
    r = np.mod(a, math.pi / 2)
    return np.minimum(r, math.pi / 2 - r)


def _num(v: float, d: int | None = None, sig: int | None = None):
    """Compact JSON number rounded to d decimals or sig significant digits; 1 not 1.0."""
    v = float(f"{v:.{sig}g}") if sig else round(float(v), d)
    return int(v) if v == int(v) else v


class PackingViewer:
    """Viewer for one submission file: build() writes the HTML page, show(n) draws one packing in a notebook."""

    def __init__(self, src=None, dst=None, xy_digits: int = 6, plotly_js: str = "auto"):
        self.src = Path(src) if src else default_src()
        self.dst = Path(dst) if dst else default_dst()
        self.xy_digits, self.plotly_js = xy_digits, plotly_js  # plotly_js: "auto" | "inline" | "cdn"
        self.packings = load_submission(self.src)
        self.L = {n: side_length(r) for n, r in self.packings.items()}

    def score(self, n: int) -> float:
        return score_n(n, self.L[n])

    def total(self) -> float:
        """Sum of score_n over n = 1..1000, missing n at the grid ceiling. Overlaps are NOT checked."""
        return sum(self.score(n) if n in self.L else grid_score(n) for n in range(1, 1001))

    def entry(self, n: int) -> dict:
        rows = self.packings[n]
        c, dev = corners(rows), axis_dev(rows[:, 2])
        rot, near = np.flatnonzero(dev >= NEAR_TOL), np.flatnonzero((dev > AXIS_TOL) & (dev < NEAR_TOL))
        d = self.xy_digits
        return {"L": self.L[n], "s": self.score(n), "o": [_num(c[..., 0].min(), 9), _num(c[..., 1].min(), 9)],
                "x": [_num(v, d) for v in rows[:, 0]], "y": [_num(v, d) for v in rows[:, 1]],
                "r": rot.tolist(), "ra": [_num(rows[k, 2], sig=9) for k in rot],
                "q": near.tolist(), "qa": [_num(rows[k, 2], sig=9) for k in near]}

    def html(self) -> str:
        """The whole page as a string: data as JSON, plotly.js inline or from the CDN (see plotly_js)."""
        data = {"label": self.src.name, "total": self.total(), "packings": {n: self.entry(n) for n in self.packings}}
        blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
        page = PAGE.replace("__LABEL__", self.src.name.replace("&", "&amp;").replace("<", "&lt;"))
        js = self._plotly_tag(len(page) + len(blob))
        return page.replace("__PLOTLY__", js).replace("__DATA__", blob)

    def _plotly_tag(self, size: int) -> str:
        mode = self.plotly_js
        if mode in ("auto", "inline"):
            try:
                from plotly.offline import get_plotlyjs
                src = get_plotlyjs()
            except ImportError:
                if mode == "inline":
                    raise
                src = None
            if src is not None and (mode == "inline" or (size + len(src)) / 1e6 <= MAX_INLINE_MB):
                self.plotly_mode = "inline"
                return "<script>" + src.replace("</script", "<\\/script") + "</script>"
        self.plotly_mode = "CDN"
        return f'<script src="{PLOTLY_CDN}" charset="utf-8"></script>'

    def build(self) -> Path:
        """Write the HTML page to dst; prints size and build time."""
        t0 = time.time()
        self.dst.parent.mkdir(parents=True, exist_ok=True)
        self.dst.write_text(self.html(), encoding="utf-8")
        self.size_mb = self.dst.stat().st_size / 1e6
        print(f"{len(self.packings)} packings from {self.src.name} -> {self.dst} "
              f"({self.size_mb:.2f} MB, plotly.js {self.plotly_mode}, {time.time() - t0:.1f} s)")
        return self.dst

    def figure(self, n: int):
        """plotly Figure of packing n: squares coloured by angle class, bounding square of side L."""
        import plotly.graph_objects as go
        rows, e = self.packings[n], self.entry(n)
        c = corners(rows)
        rot, near = set(e["r"]), set(e["q"])
        groups = {"aligned": [k for k in range(len(rows)) if k not in rot and k not in near],
                  "near": e["q"], "rotated": e["r"]}
        names = {"aligned": "axis-aligned", "near": "nearly aligned (< 1e-3 rad)", "rotated": "rotated"}
        fig = go.Figure()
        for key, idx in groups.items():
            X, Y = [], []
            for k in idx:
                X += [*c[k, :, 0], c[k, 0, 0], None]
                Y += [*c[k, :, 1], c[k, 0, 1], None]
            fig.add_scatter(x=X, y=Y, mode="lines", fill="toself", line={"color": COLORS[key], "width": 1},
                            name=f"{names[key]} ({len(idx)})", hoverinfo="skip", visible=True if idx else "legendonly")
        (x0, y0), L = e["o"], self.L[n]
        fig.add_scatter(x=[x0, x0 + L, x0 + L, x0, x0], y=[y0, y0, y0 + L, y0 + L, y0], mode="lines",
                        line={"color": COLORS["box"], "width": 1.5}, name="bounding square", hoverinfo="skip")
        fig.update_layout(title=f"n = {n}   L = {L:.12f}   score_n = {self.score(n):.12f}   "
                                f"rotated: {len(rot)} (+{len(near)} nearly aligned)",
                          width=640, height=680, template="plotly_white", margin={"l": 40, "r": 10, "t": 50, "b": 30},
                          legend={"orientation": "h", "y": -0.05}, yaxis={"scaleanchor": "x", "scaleratio": 1})
        return fig

    def show(self, n: int):
        self.figure(n).show()

    def show_scores(self):
        self.scores_figure().show()

    def scores_figure(self):
        """score_n over n, with the grid ceiling for reference."""
        import plotly.graph_objects as go
        ns = sorted(self.L)
        fig = go.Figure()
        fig.add_scatter(x=ns, y=[grid_score(n) for n in ns], mode="lines", name="grid ceiling",
                        line={"color": "#b0afa9", "width": 1})
        fig.add_scatter(x=ns, y=[self.score(n) for n in ns], mode="lines", name="score_n",
                        line={"color": COLORS["aligned"], "width": 1.5})
        fig.update_layout(title=f"{self.src.name}: total {self.total():.6f} (overlaps not checked)", width=760,
                          height=360, template="plotly_white", xaxis_title="n", yaxis_title="score_n = L/√n − 1")
        return fig


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Packing Viewer</title>
__PLOTLY__
<style>
:root {
  color-scheme: light;
  --surface: #fcfcfb; --surface-2: #f1f0ed; --border: #d6d5d0;
  --text: #0b0b0b; --text-2: #52514e; --muted: #8a8984; --grid: #e6e5e1;
  --aligned: #2a78d6; --rotated: #eb6834; --near: #1baf7a; --box: #0b0b0b; --ceil: #b0afa9;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --surface-2: #262624; --border: #3d3d3a;
    --text: #f0efec; --text-2: #c3c2b7; --muted: #8f8e88; --grid: #2e2e2c;
    --aligned: #3987e5; --rotated: #d95926; --near: #199e70; --box: #f0efec; --ceil: #5c5b57;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --surface-2: #262624; --border: #3d3d3a;
  --text: #f0efec; --text-2: #c3c2b7; --muted: #8f8e88; --grid: #2e2e2c;
  --aligned: #3987e5; --rotated: #d95926; --near: #199e70; --box: #f0efec; --ceil: #5c5b57;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--text);
       font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 16px; }
h1 { font-size: 18px; font-weight: 600; margin: 0 0 2px; }
.sub { color: var(--text-2); margin: 0 0 12px; word-break: break-all; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 10px; }
.controls input { width: 7em; font: inherit; padding: 5px 8px; border: 1px solid var(--border); border-radius: 6px;
                  background: var(--surface); color: var(--text); }
.controls button { font: inherit; padding: 5px 12px; border: 1px solid var(--border); border-radius: 6px;
                   background: var(--surface-2); color: var(--text); cursor: pointer; }
.controls button:disabled { opacity: .45; cursor: default; }
.controls .hint { color: var(--muted); font-size: 12px; }
#info { font-variant-numeric: tabular-nums; color: var(--text-2); min-height: 2.9em; margin-bottom: 6px; }
#info b { color: var(--text); font-weight: 600; }
#plot { width: 100%; aspect-ratio: 1 / 1; }
#scores { width: 100%; height: 260px; margin-top: 8px; }
.note { color: var(--muted); font-size: 12px; margin-top: 8px; }
</style>
</head>
<body>
<main>
  <h1>Squares in Squares: packing viewer</h1>
  <p class="sub">source: __LABEL__ · <span id="count"></span></p>
  <div class="controls">
    <button id="prev" title="previous n (←)">‹ prev</button>
    <label for="n">n</label>
    <input id="n" type="number" min="1" max="1000" step="1" inputmode="numeric">
    <button id="go">show</button>
    <button id="next" title="next n (→)">next ›</button>
    <span class="hint">Enter to show · ←/→ to step · click the score plot to jump</span>
  </div>
  <div id="info"></div>
  <div id="plot"></div>
  <div id="scores"></div>
  <p class="note">L = max(x-extent, y-extent) of all square corners, in float64. Overlaps are not checked:
    the official metric is the only judge of validity.</p>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const P = D.packings;
const NS = Object.keys(P).map(Number).sort((a, b) => a - b);
const inp = document.getElementById("n"), info = document.getElementById("info");
let cur = null;
inp.min = NS[0]; inp.max = NS[NS.length - 1];
document.getElementById("count").textContent = NS.length + " packings (n = " + NS[0] + "…" + NS[NS.length - 1]
  + ") · total score " + D.total.toFixed(6) + " (missing n at the grid ceiling, overlaps not checked)";

const css = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
const fx = (v, d) => Number(v).toFixed(d);
const ceilScore = n => Math.ceil(Math.sqrt(n) - 1e-12) / Math.sqrt(n) - 1;

function hexA(hex, a) {
  const h = hex.replace("#", "");
  const [r, g, b] = [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16));
  return `rgba(${r},${g},${b},${a})`;
}

// same corner order as the official metric (compute_corners_float)
function polys(e, idx, ang) {
  const X = [], Y = [];
  for (const k of idx) {
    const x = e.x[k], y = e.y[k], a = ang.get(k) || 0, c = Math.cos(a), s = Math.sin(a);
    const dx1 = 0.5 * c - 0.5 * s, dy1 = 0.5 * s + 0.5 * c, dx2 = -0.5 * c - 0.5 * s, dy2 = -0.5 * s + 0.5 * c;
    X.push(x + dx1, x + dx2, x - dx1, x - dx2, x + dx1, null);
    Y.push(y + dy1, y + dy2, y - dy1, y - dy2, y + dy1, null);
  }
  return [X, Y];
}

function draw(n) {
  const e = P[n];
  if (!e) {
    info.innerHTML = `<b>n = ${n}</b> is not in this file`;
    return;
  }
  cur = n;
  inp.value = n;
  history.replaceState(null, "", "#n=" + n);
  document.getElementById("prev").disabled = n <= NS[0];
  document.getElementById("next").disabled = n >= NS[NS.length - 1];

  const ang = new Map(), rot = new Set(e.r), near = new Set(e.q);
  e.r.forEach((k, i) => ang.set(k, e.ra[i]));
  e.q.forEach((k, i) => ang.set(k, e.qa[i]));
  const all = [...Array(e.x.length).keys()], al = all.filter(k => !rot.has(k) && !near.has(k));
  const kind = k => near.has(k) ? "nearly aligned" : rot.has(k) ? "rotated" : "axis-aligned";
  const gs = ceilScore(n);
  info.innerHTML = `<b>n = ${n}</b> · L = <b>${fx(e.L, 12)}</b> · score_n = <b>${fx(e.s, 12)}</b>`
    + ` · grid ceiling ${fx(gs, 6)} (gain ${fx(gs - e.s, 6)})`
    + `<br>rotated squares: <b>${e.r.length}</b> / ${e.x.length}`
    + (e.q.length ? ` · nearly aligned (|angle mod π/2| < 1e-3 rad): <b>${e.q.length}</b>` : "");

  const text = css("--text-2"), grid = css("--grid"), traces = [];
  for (const [idx, col, name] of [[al, css("--aligned"), "axis-aligned"],
                                   [e.q, css("--near"), "nearly aligned, < 1e-3 rad"],
                                   [e.r, css("--rotated"), "rotated"]]) {
    const [X, Y] = polys(e, idx, ang);
    traces.push({type: "scatter", mode: "lines", x: X, y: Y, fill: "toself", fillcolor: hexA(col, 0.35),
                 line: {color: col, width: 1}, name: `${name} (${idx.length})`, hoverinfo: "skip",
                 visible: idx.length ? true : "legendonly"});
  }
  const o = e.o, L = e.L;
  traces.push({type: "scatter", mode: "lines", name: "bounding square", hoverinfo: "skip",
               x: [o[0], o[0] + L, o[0] + L, o[0], o[0]], y: [o[1], o[1], o[1] + L, o[1] + L, o[1]],
               line: {color: css("--box"), width: 1.5}});
  traces.push({type: "scatter", mode: "markers", x: e.x, y: e.y, showlegend: false, name: "",
               marker: {size: e.x.length > 300 ? 2 : 4, color: text, opacity: 0.6},
               customdata: all.map(k => [k, ang.get(k) || 0, (ang.get(k) || 0) * 180 / Math.PI, kind(k)]),
               hovertemplate: "square_id %{customdata[0]} (%{customdata[3]})<br>angle %{customdata[1]:.9g} rad"
                 + " = %{customdata[2]:.4f}°<br>centre (%{x}, %{y})<extra></extra>"});
  const pad = Math.max(0.15, 0.03 * L);
  const ax = lo => ({range: [lo - pad, lo + L + pad], gridcolor: grid, zerolinecolor: grid,
                     color: text, tickfont: {size: 11}, constrain: "domain"});
  Plotly.react("plot", traces, {
    xaxis: ax(o[0]), yaxis: Object.assign(ax(o[1]), {scaleanchor: "x", scaleratio: 1}),
    margin: {l: 40, r: 10, t: 10, b: 30}, paper_bgcolor: css("--surface"), plot_bgcolor: css("--surface"),
    font: {color: text}, hovermode: "closest", hoverdistance: 30,
    legend: {orientation: "h", x: 0, y: 1.0, yanchor: "bottom", font: {color: text}},
    hoverlabel: {bgcolor: css("--surface-2"), bordercolor: css("--border"), font: {color: css("--text")}},
  }, {responsive: true, displaylogo: false, scrollZoom: true});
  drawScores();
}

function drawScores() {
  const text = css("--text-2"), grid = css("--grid");
  const traces = [
    {type: "scatter", mode: "lines", x: NS, y: NS.map(ceilScore), name: "grid ceiling", hoverinfo: "skip",
     line: {color: css("--ceil"), width: 1}},
    {type: "scatter", mode: "lines+markers", x: NS, y: NS.map(n => P[n].s), name: "score_n",
     line: {color: css("--aligned"), width: 1.5}, marker: {size: 3},
     hovertemplate: "n = %{x}<br>score_n = %{y:.9f}<extra></extra>"},
    {type: "scatter", mode: "markers", x: [cur], y: [P[cur].s], name: "n = " + cur, hoverinfo: "skip",
     marker: {size: 10, color: css("--rotated"), symbol: "circle-open", line: {width: 2}}},
  ];
  Plotly.react("scores", traces, {
    xaxis: {title: {text: "n", font: {size: 11}}, gridcolor: grid, zerolinecolor: grid, color: text},
    yaxis: {title: {text: "score_n = L/√n − 1", font: {size: 11}}, gridcolor: grid, zerolinecolor: grid, color: text},
    margin: {l: 55, r: 10, t: 10, b: 40}, paper_bgcolor: css("--surface"), plot_bgcolor: css("--surface"),
    font: {color: text}, hovermode: "closest", showlegend: false,
  }, {responsive: true, displaylogo: false});
}

function step(d) {
  if (cur === null) return;
  const i = NS.indexOf(cur) + d;
  if (i >= 0 && i < NS.length) draw(NS[i]);
}
function go() {
  const n = parseInt(inp.value, 10);
  if (Number.isInteger(n)) draw(n);
}
document.getElementById("go").onclick = go;
document.getElementById("prev").onclick = () => step(-1);
document.getElementById("next").onclick = () => step(1);
inp.addEventListener("keydown", ev => { if (ev.key === "Enter") go(); });
document.addEventListener("keydown", ev => {
  if (ev.target === inp) return;
  if (ev.key === "ArrowLeft") step(-1);
  if (ev.key === "ArrowRight") step(1);
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => cur !== null && draw(cur));

if (typeof Plotly === "undefined") {
  info.textContent = "plotly.js did not load (this page was built with the CDN version and needs network access).";
} else {
  const h = /n=(\d+)/.exec(location.hash);
  draw(h && P[+h[1]] ? +h[1] : NS[0]);
  document.getElementById("scores").on("plotly_click", ev => ev.points.length && draw(ev.points[0].x));
}
</script>
</body>
</html>
"""


def main(argv=None) -> None:
    """CLI: sis-vis SRC.csv [DST.html] [--cdn] [--xy-digits N]."""
    import argparse
    ap = argparse.ArgumentParser(prog="sis-vis", description="Build the packing viewer HTML from a submission CSV.")
    ap.add_argument("src", nargs="?", help="submission CSV (default: sample_submission.csv in the current directory)")
    ap.add_argument("dst", nargs="?", help="output HTML (default: ./packings.html)")
    ap.add_argument("--cdn", action="store_true", help="load plotly.js from the CDN instead of embedding it")
    ap.add_argument("--xy-digits", type=int, default=6, help="decimals kept for centres in the page (display only)")
    args = ap.parse_args(argv)
    PackingViewer(args.src, args.dst, args.xy_digits, "cdn" if args.cdn else "auto").build()


if __name__ == "__main__":
    main()
