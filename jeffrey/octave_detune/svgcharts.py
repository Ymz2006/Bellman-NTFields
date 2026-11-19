"""Small inline-SVG chart helpers for the octave-detune findings page (theme-aware through CSS classes; hover text in
data-tip attributes, shown by the page's tooltip script; every value is also in a table on the page).

Classes the page styles: .ax (axis text), .grid (hairline), .base (baseline), .s1 .. .s8 (series stroke / fill),
.band (noise band), .ref (reference rule), .lbl (direct label), .hit (transparent hover target).
"""
import html
import math

W_DEFAULT = 460


def esc(s):
    return html.escape(str(s), quote=True)


def nice_ticks(lo, hi, count=5):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / max(count - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = math.ceil(lo / step - 1e-9) * step
    out = []
    v = start
    while v <= hi + 1e-9 * step:
        out.append(round(v, 10))
        v += step
    return out


class Frame:
    """A plot area with linear (or log10) axes; returns pixel positions."""

    def __init__(self, x0, x1, y0, y1, w=W_DEFAULT, h=260, left=48, right=16, top=14, bottom=40, ylog=False):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.w, self.h, self.l, self.r, self.t, self.b = w, h, left, right, top, bottom
        self.ylog = ylog

    def X(self, x):
        return self.l + (x - self.x0) / (self.x1 - self.x0) * (self.w - self.l - self.r)

    def Y(self, y):
        if self.ylog:
            y, y0, y1 = math.log10(max(y, 1e-12)), math.log10(self.y0), math.log10(self.y1)
        else:
            y0, y1 = self.y0, self.y1
        return self.h - self.b - (y - y0) / (y1 - y0) * (self.h - self.t - self.b)

    def axes(self, xticks, yticks, xlabel='', ylabel='', xfmt=str, yfmt=str):
        out = []
        for v in yticks:
            y = self.Y(v)
            out.append(f'<line class="grid" x1="{self.l}" x2="{self.w - self.r}" y1="{y:.1f}" y2="{y:.1f}"/>')
            out.append(f'<text class="ax" x="{self.l - 6}" y="{y + 4:.1f}" text-anchor="end">{esc(yfmt(v))}</text>')
        yb = self.h - self.b
        out.append(f'<line class="base" x1="{self.l}" x2="{self.w - self.r}" y1="{yb}" y2="{yb}"/>')
        for v in xticks:
            x = self.X(v)
            out.append(f'<line class="base" x1="{x:.1f}" x2="{x:.1f}" y1="{yb}" y2="{yb + 4}"/>')
            out.append(f'<text class="ax" x="{x:.1f}" y="{yb + 17}" text-anchor="middle">{esc(xfmt(v))}</text>')
        if xlabel:
            out.append(f'<text class="ax axt" x="{(self.l + self.w - self.r) / 2:.1f}" y="{self.h - 4}" text-anchor="middle">{esc(xlabel)}</text>')
        if ylabel:
            out.append(f'<text class="ax axt" x="12" y="{(self.t + yb) / 2:.1f}" text-anchor="middle" transform="rotate(-90 12 {(self.t + yb) / 2:.1f})">{esc(ylabel)}</text>')
        return out


def svg(frame, body, label):
    return (f'<svg viewBox="0 0 {frame.w} {frame.h}" role="img" aria-label="{esc(label)}" preserveAspectRatio="xMidYMid meet">'
            + ''.join(body) + '</svg>')


def polyline(frame, pts, cls, width=2, dash=None):
    if len(pts) < 2:
        return ''
    d = ' '.join(f'{frame.X(x):.1f},{frame.Y(y):.1f}' for x, y in pts)
    da = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<polyline class="{cls} ln" fill="none" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{da} points="{d}"/>'


def dot(frame, x, y, cls, r=4, tip=None, ring=True):
    cx, cy = frame.X(x), frame.Y(y)
    t = f' data-tip="{esc(tip)}" tabindex="0"' if tip else ''
    ringc = ' ring' if ring else ''
    return (f'<g class="pt"{t}><circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="12"/>'
            f'<circle class="{cls} dot{ringc}" cx="{cx:.1f}" cy="{cy:.1f}" r="{r}"/></g>')


def hrule(frame, y, cls, text=None, side='right', tip=None):
    yy = frame.Y(y)
    out = [f'<line class="{cls}" x1="{frame.l}" x2="{frame.w - frame.r}" y1="{yy:.1f}" y2="{yy:.1f}"/>']
    if text:
        x = frame.w - frame.r - 2 if side == 'right' else frame.l + 4
        anchor = 'end' if side == 'right' else 'start'
        out.append(f'<text class="lbl small" x="{x}" y="{yy - 4:.1f}" text-anchor="{anchor}">{esc(text)}</text>')
    if tip:
        out.append(f'<rect class="hit" data-tip="{esc(tip)}" tabindex="0" x="{frame.l}" y="{yy - 6:.1f}" width="{frame.w - frame.l - frame.r}" height="12"/>')
    return out


def band(frame, lo, hi, cls='band'):
    y1, y0 = frame.Y(hi), frame.Y(lo)
    return f'<rect class="{cls}" x="{frame.l}" y="{y1:.1f}" width="{frame.w - frame.l - frame.r}" height="{max(y0 - y1, 0.5):.1f}"/>'


def vband(frame, lo, hi, cls='band'):
    x0, x1 = frame.X(lo), frame.X(hi)
    return f'<rect class="{cls}" x="{x0:.1f}" y="{frame.t}" width="{max(x1 - x0, 0.5):.1f}" height="{frame.h - frame.t - frame.b}"/>'


def legend(items):
    """HTML legend: items = [(class, text, kind)] with kind 'line' | 'dot' | 'rule'."""
    out = ['<div class="legend">']
    for cls, text, kind in items:
        if kind == 'dot':
            sw = f'<svg width="14" height="14" aria-hidden="true"><circle class="{cls} dot" cx="7" cy="7" r="4"/></svg>'
        elif kind == 'rule':
            sw = f'<svg width="22" height="14" aria-hidden="true"><line class="{cls}" x1="1" x2="21" y1="7" y2="7"/></svg>'
        elif kind == 'dot-line':
            sw = f'<svg width="22" height="14" aria-hidden="true"><line class="{cls} ln" stroke-width="2" stroke-dasharray="1.5 3" x1="1" x2="21" y1="7" y2="7"/></svg>'
        elif kind == 'dash':
            sw = f'<svg width="22" height="14" aria-hidden="true"><line class="{cls} ln" stroke-width="2" stroke-dasharray="5 4" x1="1" x2="21" y1="7" y2="7"/></svg>'
        elif kind == 'band':
            sw = f'<svg width="22" height="14" aria-hidden="true"><rect class="{cls}" x="1" y="2" width="20" height="10"/></svg>'
        else:
            sw = f'<svg width="22" height="14" aria-hidden="true"><line class="{cls} ln" stroke-width="2" x1="1" x2="21" y1="7" y2="7"/></svg>'
        out.append(f'<span class="key">{sw}<span>{esc(text)}</span></span>')
    out.append('</div>')
    return ''.join(out)
