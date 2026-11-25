#!/usr/bin/env python3
"""Benchmark bar chart from a JSON of rows, in a light and a dark version (PNG + SVG each).

  python tools/make_charts.py assets/bench_cuda.json -o assets/benchmark   # -> benchmark-{light,dark}.{png,svg}

JSON:
  {"title": "...", "subtitle": "...", "footnote": "...", "unit": "s", "baseline": "<label of the 1.00x row>",
   "rows": [{"label": "official, compiled", "value": 6.470, "group": "official"},
            {"label": "fast", "value": 4.221, "group": "port", "detail": "kernels + FA3"}, ...]}
group "official" bars are drawn neutral, "port" bars in the accent colour; every bar is labelled with its value and
its speed-up against the baseline row (lower time is better). Horizontal bars, rows in the given order.
Embed in Markdown with <picture> so GitHub picks the version matching the reader's theme.
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {  # chart surface, ink, muted ink, grid, neutral bar, accent bar (validated reference palette slot 1)
    'light': dict(bg='#ffffff', ink='#0b0b0b', muted='#52514e', grid='#e4e3df', neutral='#a3a29c', accent='#2a78d6'),
    'dark': dict(bg='#0d1117', ink='#f0efec', muted='#c3c2b7', grid='#30302d', neutral='#6f6e68', accent='#3987e5'),
}


def draw(spec, theme, out):
    c = THEMES[theme]
    rows = spec['rows']
    base = next(r['value'] for r in rows if r['label'] == spec['baseline'])
    n = len(rows)
    fig, ax = plt.subplots(figsize=(9, 1.05 + 0.52 * n), dpi=200)
    fig.patch.set_facecolor(c['bg'])
    ax.set_facecolor(c['bg'])
    y = list(range(n))[::-1]
    vmax = max(r['value'] for r in rows)
    colors = [c['accent'] if r.get('group') == 'port' else c['neutral'] for r in rows]
    ax.barh(y, [r['value'] for r in rows], height=0.62, color=colors, edgecolor=c['bg'], linewidth=1.5, zorder=2)
    for yi, r in zip(y, rows):
        sp = base / r['value']
        txt = f"{r['value']:.2f} {spec.get('unit', 's')}  ·  {sp:.2f}×"
        ax.text(r['value'] + vmax * 0.012, yi, txt, va='center', ha='left', fontsize=9.5,
                color=c['ink'], fontweight='bold' if r.get('group') == 'port' else 'normal', zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r['label'] + (f"\n{r['detail']}" if r.get('detail') else '') for r in rows],
                       fontsize=9.5, color=c['ink'])
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_linespacing(1.15)
    ax.set_xlim(0, vmax * 1.30)
    ax.xaxis.grid(True, color=c['grid'], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', colors=c['muted'], labelsize=8.5, length=0)
    ax.tick_params(axis='y', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(spec.get('xlabel', 'seconds per request (lower is better)'), color=c['muted'], fontsize=9)
    fig.text(0.012, 0.985, spec['title'], ha='left', va='top', fontsize=13, fontweight='bold', color=c['ink'])
    if spec.get('subtitle'):
        fig.text(0.012, 0.985 - 0.30 / (1.05 + 0.52 * n) * 1.6, spec['subtitle'], ha='left', va='top', fontsize=9,
                 color=c['muted'])
    if spec.get('footnote'):
        import textwrap
        fig.text(0.012, 0.012, textwrap.fill(spec['footnote'], 150), ha='left', va='bottom', fontsize=7.5,
                 color=c['muted'])
    fig.tight_layout(rect=(0, 0.09 if spec.get('footnote') else 0, 1, 1 - 0.75 / (1.05 + 0.52 * n)))
    for ext in ('png', 'svg'):
        fig.savefig(f'{out}-{theme}.{ext}', facecolor=c['bg'])
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('spec')
    p.add_argument('-o', '--out', required=True, help='output prefix')
    a = p.parse_args()
    spec = json.load(open(a.spec))
    for theme in THEMES:
        draw(spec, theme, a.out)


if __name__ == '__main__':
    main()
