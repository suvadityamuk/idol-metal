#!/usr/bin/env python3
"""Turntable GIF of Idol .ply files drawn as coloured POINTS on the CPU (numpy + Pillow): each Gaussian's centre is
drawn as one pixel-sized dot with its SH degree-0 colour (z-buffered, 2x supersampled), for Gaussians with opacity
above a threshold. This is NOT a Gaussian-splat render (no scales, rotations or alpha blending); there is no 3DGS
rasterizer for MPS, so this is the cheap way to show a .ply this backend wrote.

Idol's world is y-down (the object's up is -y) and the front faces +x; the orbit is around the y axis.

  python tools/point_turntable.py out.ply -o out.gif [--input photo.png] [--labels "fast build"] [--frames 48]
  python tools/point_turntable.py ref.ply fast.ply --labels "official (MPS)" "fast build" --input in.png -o hero.gif

Needs numpy and Pillow only.
"""
import argparse
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (12, 12, 12)
SH_C0 = 0.28209479177387814


def load_ply(path, min_opacity=0.3):
    with open(path, 'rb') as f:
        n, props = 0, []
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
            elif line.startswith('property'):
                props.append(line.split()[-1])
            elif line == 'end_header':
                break
        data = np.frombuffer(f.read(), dtype='<f4').reshape(n, len(props))
    c = {p: data[:, i] for i, p in enumerate(props)}
    xyz = np.stack([c['x'], c['y'], c['z']], 1).astype(np.float64)
    rgb = np.clip(0.5 + SH_C0 * np.stack([c[f'f_dc_{i}'] for i in range(3)], 1), 0, 1)
    opac = 1 / (1 + np.exp(-c['opacity']))
    keep = opac > min_opacity
    return xyz[keep], rgb[keep]


def view(az_deg, el_deg, radius):
    a, e = math.radians(az_deg), math.radians(el_deg)
    eye = np.array([radius * math.cos(e) * math.sin(a), -radius * math.sin(e), radius * math.cos(e) * math.cos(a)])
    up = np.array([0.0, -1.0, 0.0])
    f = -eye / np.linalg.norm(eye)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, -u, f], 0)  # x right, y down, z forward
    return R, -R @ eye


def render(xyz, rgb, az, size=384, elev=20.0, radius=2.0, fov=35.0, ss=2, bg=BG):
    S = size * ss
    R, t = view(az, elev, radius)
    p = xyz @ R.T + t
    z = p[:, 2]
    fpx = 0.5 * S / math.tan(math.radians(fov) / 2)
    u = np.round(p[:, 0] / z * fpx + S / 2).astype(np.int64)
    v = np.round(p[:, 1] / z * fpx + S / 2).astype(np.int64)
    ok = (z > 0.05) & (u >= 0) & (u < S) & (v >= 0) & (v < S)
    u, v, z, col = u[ok], v[ok], z[ok], rgb[ok]
    img = np.empty((S, S, 3)) ; img[:] = np.array(bg) / 255
    order = np.argsort(-z)                     # far to near: later writes (nearer) win
    idx = v[order] * S + u[order]
    flat = img.reshape(-1, 3)
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):   # 2x2 dots at the supersampled size
        j = idx + dv * S + du
        m = (u[order] + du < S) & (v[order] + dv < S)
        flat[j[m]] = col[order][m]
    img = img.reshape(size, ss, size, ss, 3).mean((1, 3))
    return (img * 255).round().astype(np.uint8)


def fit_input(path, size, bg=BG):
    im = Image.open(path).convert('RGBA')
    im.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new('RGBA', (size, size), bg + (255,))
    canvas.alpha_composite(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.array(canvas.convert('RGB'))


def font(px):
    for name in ('/System/Library/Fonts/Helvetica.ttc', '/System/Library/Fonts/SFNS.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            pass
    return ImageFont.load_default()


def compose(columns, labels, caption, size):
    pad, top, bottom = 8, 30 if labels else 0, 26 if caption else 0
    W = len(columns) * size + (len(columns) + 1) * pad
    H = size + 2 * pad + top + bottom
    nframes = max(len(c) for c in columns)
    out = []
    fl, fc = font(17), font(13)
    for i in range(nframes):
        im = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(im)
        for k, col in enumerate(columns):
            x = pad + k * (size + pad)
            im.paste(Image.fromarray(col[i % len(col)]), (x, pad + top))
            if labels and labels[k]:
                d.text((x + size / 2, pad + top / 2), labels[k], fill=(235, 235, 230), font=fl, anchor='mm')
        if caption:
            d.text((W / 2, H - bottom / 2 - 2), caption, fill=(160, 160, 155), font=fc, anchor='mm')
        out.append(im)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('plys', nargs='+')
    p.add_argument('-o', '--out', required=True, help='output .gif (a .png still of the first frame is also written)')
    p.add_argument('--input', default=None, help='input image shown in the first column')
    p.add_argument('--labels', nargs='*', default=None, help='one label per column (input first if given)')
    p.add_argument('--caption', default='point rendering of Gaussian centres (SH0 colour), not a splat render')
    p.add_argument('--frames', type=int, default=48)
    p.add_argument('--size', type=int, default=320)
    p.add_argument('--fps', type=int, default=16)
    p.add_argument('--min-opacity', type=float, default=0.3)
    a = p.parse_args()
    cols = []
    if a.input:
        cols.append([fit_input(a.input, a.size)])
    for ply in a.plys:
        xyz, rgb = load_ply(ply, a.min_opacity)
        cols.append([render(xyz, rgb, 90.0 + 360.0 * i / a.frames, a.size) for i in range(a.frames)])
    frames = compose(cols, a.labels, a.caption, a.size)
    pal = [f.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE) for f in frames]
    pal[0].save(a.out, save_all=True, append_images=pal[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
    frames[0].save(a.out[:-4] + '.png')
    print(a.out, len(frames), 'frames')


if __name__ == '__main__':
    main()
