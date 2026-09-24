"""Generate the cbjev logo: SVG sources plus PNG renders (needs inkscape for the PNGs).

The mark is one stroke that forks. The trunk on the left is the state, read once; it splits into
three branches that land on three different endpoints -- a dot (a choice), a short bar (a score on
a scale) and a tick (a yes/no) -- all resolved in the same pass. That is the whole idea of cbjev:
one encoding, every question answered from it.

    python assets/make_logo.py
      -> assets/logo-mark.svg / .png, logo-lockup.svg / .png, logo-lockup-dark.svg / .png,
         docs/favicon.svg
"""
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

BLUE, BLUE2 = "#2563eb", "#7c3aed"
INK, PAPER = "#0f172a", "#e8ecf3"


def mark(size=64, bg=True):
    s = size / 64.0
    g = []
    if bg:
        g.append('<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
                 '<stop offset="0" stop-color="%s"/><stop offset="1" stop-color="%s"/></linearGradient></defs>'
                 % (BLUE, BLUE2))
        g.append('<rect x="0" y="0" width="64" height="64" rx="15" fill="url(#g)"/>')
    col = "#ffffff" if bg else BLUE
    sw = 5.2
    # trunk and three branches (cubic curves from one fork point)
    g.append('<g fill="none" stroke="%s" stroke-width="%.1f" stroke-linecap="round">' % (col, sw))
    g.append('<path d="M10 32 H24"/>')
    g.append('<path d="M24 32 C31 32 31 17 38 17"/>')
    g.append('<path d="M24 32 H38"/>')
    g.append('<path d="M24 32 C31 32 31 47 38 47"/>')
    g.append('</g>')
    # endpoints, each after a small gap: dot (choice), bar (score), tick (noul)
    g.append('<circle cx="49" cy="17" r="4.8" fill="%s"/>' % col)
    g.append('<rect x="44.2" y="28.6" width="11" height="6.8" rx="2.6" fill="%s"/>' % col)
    g.append('<path d="M44.6 46.6 l3.4 3.6 l6.6 -7.6" fill="none" stroke="%s" stroke-width="4.2" '
             'stroke-linecap="round" stroke-linejoin="round"/>' % col)
    body = "".join(g)
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 64 64">%s</svg>'
            % (size, size, body)), body


def lockup(dark=False):
    _, body = mark()
    text = PAPER if dark else INK
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="330" height="96" viewBox="0 0 330 96">'
            '<g transform="translate(8,8) scale(1.25)">%s</g>'
            '<text x="112" y="66" font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif" font-size="56" '
            'font-weight="800" letter-spacing="-1.5" fill="%s">cb<tspan fill="%s">jev</tspan></text></svg>'
            % (body, text, BLUE if not dark else "#8fb0ff"))


def write(path, svg):
    with open(path, "w") as f:
        f.write(svg)


def png(svg_path, png_path, width):
    if shutil.which("inkscape"):
        subprocess.run(["inkscape", svg_path, "--export-type=png", "--export-filename=" + png_path,
                        "--export-width=%d" % width], check=True, capture_output=True)


def main():
    svg, _ = mark()
    write(os.path.join(HERE, "logo-mark.svg"), svg)
    write(os.path.join(HERE, "logo-lockup.svg"), lockup())
    write(os.path.join(HERE, "logo-lockup-dark.svg"), lockup(dark=True))
    png(os.path.join(HERE, "logo-mark.svg"), os.path.join(HERE, "logo-mark.png"), 512)
    png(os.path.join(HERE, "logo-lockup.svg"), os.path.join(HERE, "logo-lockup.png"), 990)
    png(os.path.join(HERE, "logo-lockup-dark.svg"), os.path.join(HERE, "logo-lockup-dark.png"), 990)
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    write(os.path.join(ROOT, "docs", "favicon.svg"), svg)
    print("logo written")


if __name__ == "__main__":
    main()
