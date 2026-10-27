# SPDX-License-Identifier: Apache-2.0
"""Tests: L on hand-made packings, HTML contents, the plotly figure."""
import json
import math
import re
from pathlib import Path

import pytest

from squares_in_squares_vis import viewer as pv  # noqa: E402
S5 = 2 + 1 / math.sqrt(2)  # Goebel's n=5: four corner squares + one at pi/4 in the middle


def _write(path: Path, packings: dict) -> Path:
    lines, i = ["id,n,square_id,x,y,angle"], 0
    for n, rows in packings.items():
        for k, (x, y, a) in enumerate(rows):
            lines.append(f"{i},{n},{k},{x!r},{y!r},{a!r}")
            i += 1
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def tiny(tmp_path):
    c = S5 / 2
    goebel = [(0.5, 0.5, 0.0), (S5 - 0.5, 0.5, 0.0), (0.5, S5 - 0.5, math.pi / 2), (S5 - 0.5, S5 - 0.5, 0.0),
              (c, c, math.pi / 4)]
    near = [(0.5, 0.5, 0.0), (1.5, 0.5, 2e-4)]  # n=2 with one nearly aligned square
    return _write(tmp_path / "tiny.csv", {1: [(0.5, 0.5, 0.0)], 2: near, 5: goebel})


def _blob(html: str) -> dict:
    return json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1))


def test_side_length_and_score(tiny):
    P = pv.load_submission(tiny)
    assert sorted(P) == [1, 2, 5]
    assert pv.side_length(P[1]) == 1.0
    assert pv.side_length(P[2]) == pytest.approx(2 + 0.5 * (math.cos(2e-4) + math.sin(2e-4)) - 0.5, abs=1e-15)
    assert pv.side_length(P[5]) == pytest.approx(S5, abs=1e-14)
    assert pv.score_n(5, S5) == pytest.approx(S5 / math.sqrt(5) - 1)
    assert pv.grid_score(5) == pytest.approx(3 / math.sqrt(5) - 1) and pv.grid_score(4) == 0


def test_html_has_every_n_and_L(tiny, tmp_path):
    v = pv.PackingViewer(tiny, tmp_path / "out" / "p.html", plotly_js="cdn")
    html = v.build().read_text()
    d = _blob(html)
    assert sorted(map(int, d["packings"])) == [1, 2, 5]
    for n in (1, 2, 5):
        e = d["packings"][str(n)]
        assert len(e["x"]) == len(e["y"]) == n
        assert e["L"] == v.L[n] and f'"L":{v.L[n]!r}' in html  # full float repr, not rounded
    assert d["packings"]["5"]["r"] == [4] and d["packings"]["5"]["ra"] == [pytest.approx(math.pi / 4)]
    assert d["packings"]["2"]["q"] == [1] and d["packings"]["2"]["r"] == []  # 2e-4 rad: nearly aligned
    assert d["packings"]["5"]["q"] == []  # pi/2 counts as axis-aligned
    grid_rest = sum(pv.grid_score(n) for n in range(1, 1001) if n not in (1, 2, 5))
    assert d["total"] == pytest.approx(v.score(2) + v.score(5) + grid_rest)
    assert "cdn.plot.ly" in html and "__" not in html.replace("__proto__", "")  # all placeholders filled
    assert v.size_mb < 0.1


def test_figure(tiny):
    pytest.importorskip("plotly")
    fig = pv.PackingViewer(tiny, "unused.html").figure(5)
    assert [t.name for t in fig.data][:3] == ["axis-aligned (4)", "nearly aligned (< 1e-3 rad) (0)", "rotated (1)"]




def test_static_previews(tiny, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from squares_in_squares_vis import static
    v = pv.PackingViewer(tiny, tmp_path / "o.html")
    fig = static.draw(v, 5, show=False)
    assert fig.axes[0].get_title().startswith("n = 5")
    assert static.draw_scores(v, show=False).axes[0].get_xlabel() == "n"
