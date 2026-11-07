"""results/progress.png: score per experiment, running-best step line, FA2-parity reference."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from driver.results import read_results  # noqa: E402

ACCENT, GREY, INK, INK2, SURFACE = "#2a78d6", "#9a9892", "#0b0b0b", "#52514e", "#fcfcfb"


def load_points(tsv: Path):
    rows = [r for r in read_results(tsv) if r["status"] != "usage_limit"]
    pts, best_x, best_y = [], [], []
    for r in rows:
        if not r["score"]:
            continue
        x, y = int(r["exp_id"]), float(r["score"])
        pts.append((x, y))
        if r["status"] in ("kept", "baseline"):
            best_x.append(x)
            best_y.append(float(r["best_score"] or y))
    return pts, best_x, best_y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-tsv", type=Path, default=Path("results.tsv"))
    ap.add_argument("--out", type=Path, default=Path("results/progress.png"))
    a = ap.parse_args()
    pts, bx, by = load_points(a.results_tsv)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=22, color=GREY, alpha=0.8,
               edgecolors=SURFACE, linewidths=1, label="every scored experiment", zorder=2)
    xmax = max([q[0] for q in pts] + [1])
    ax.step(bx + [xmax], by + [by[-1]] if by else [], where="post", color=ACCENT, lw=2, label="running best (kept)", zorder=3)
    ax.scatter(bx, by, s=34, color=ACCENT, edgecolors=SURFACE, linewidths=1.5, zorder=4)
    ax.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate("FA2 parity", (0.995, 1.0), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=9, color=INK2)
    if by:
        ax.annotate(f"seed {by[0]:.2f}", (bx[0], by[0]), xytext=(8, -4), textcoords="offset points",
                    fontsize=9, color=INK2, va="top")
        ax.annotate(f"best {by[-1]:.3f}", (bx[-1], by[-1]), xytext=(-6, 8), textcoords="offset points",
                    fontsize=9, color=INK2, ha="right")
    ax.set_xlabel("experiment id", color=INK2)
    ax.set_ylabel("score (geomean FA2 ms / ours ms)", color=INK2)
    ax.set_title("Triton GQA attention vs FA2, RTX 5090", loc="left", color=INK, fontsize=12)
    ax.set_ylim(0, max([1.1] + [p[1] for p in pts]) * 1.08)
    ax.set_xlim(-0.5, max([p[0] for p in pts] + [1]) + 0.5)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK2, fontsize=9, bbox_to_anchor=(1, 0.06))
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor=SURFACE)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
