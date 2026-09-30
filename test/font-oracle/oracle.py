"""Compare sfnt.almd with fontTools over a font's characters.

  python3 test/font-oracle/oracle.py FONT [FACE] [--codes all|sample|N] [--no-build]

Builds dump.almd once, runs it over the chosen code points, and checks each
glyph id, advance and outline against fontTools. Outlines are compared as
closed cycles of segments (quadratics raised to cubics, the implicit closing
line made explicit, the start rotated to a canonical point), since a contour
may legitimately start at any of its on-curve points.
"""
import subprocess, sys, os, random
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen

TOL = 1e-6
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


class Rec(BasePen):
    def __init__(self, gs):
        super().__init__(gs)
        self.contours = []
    def _moveTo(self, p):
        self.contours.append([p])
    def _lineTo(self, p):
        self.contours[-1].append(("L", p))
    def _curveToOne(self, a, b, c):
        self.contours[-1].append(("C", a, b, c))
    def _closePath(self):
        pass
    def _endPath(self):
        pass


def canon(contours):
    out = []
    for c in contours:
        start, segs = c[0], list(c[1:])
        end = segs[-1][-1] if segs else start
        if not close(end, start):
            segs.append(("L", start))
        segs = [s for i, s in enumerate(segs)
                if not (s[0] == "L" and close(s[1], (segs[i - 1][-1] if i else start)))]
        if not segs:
            continue
        rounded = [tuple([s[0]] + [(round(x, 3), round(y, 3)) for x, y in s[1:]]) for s in segs]
        k = min(range(len(rounded)), key=lambda i: (rounded[i - 1][-1], rounded[i:] + rounded[:i]))
        out.append(segs[k:] + segs[:k])
    out.sort(key=lambda s: [(x[0],) + tuple(x[1:]) for x in s])
    return out


def shift(p, dx):
    return (p[0] + dx, p[1])


def close(p, q):
    return abs(p[0] - q[0]) <= TOL and abs(p[1] - q[1]) <= TOL


def same(a, b):
    if len(a) != len(b):
        return False
    for ca, cb in zip(a, b):
        if len(ca) != len(cb):
            return False
        for sa, sb in zip(ca, cb):
            if sa[0] != sb[0] or not all(close(p, q) for p, q in zip(sa[1:], sb[1:])):
                return False
    return True


def parse(lines):
    head = lines[0].split()
    glyphs, cur = {}, None
    for ln in lines[1:]:
        t = ln.split()
        if t[0] == "G":
            cur = [int(t[2]), int(t[3]), []]
            glyphs[int(t[1])] = cur
        elif t[0] == "M":
            cur[2].append([(float(t[1]), float(t[2]))])
        elif t[0] == "L":
            cur[2][-1].append(("L", (float(t[1]), float(t[2]))))
        elif t[0] == "C":
            v = list(map(float, t[1:]))
            cur[2][-1].append(("C", (v[0], v[1]), (v[2], v[3]), (v[4], v[5])))
    return [int(x) for x in head[1:]], glyphs


def main():
    font_path = sys.argv[1]
    face = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 0
    mode = sys.argv[sys.argv.index("--codes") + 1] if "--codes" in sys.argv else "sample"
    ft = TTFont(font_path, fontNumber=face)
    cmap = ft.getBestCmap()
    codes = sorted(cmap)
    if mode == "sample":
        random.seed(0)
        fixed = [c for c in [0x20, 0x41, 0x61, 0x67, 0x40, 0x3042, 0x30A2, 0x6F22, 0x5B57, 0x9F98, 0xFF01, 0x3001]
                 if c in cmap]
        codes = fixed + random.sample(codes, min(500, len(codes)))
    elif mode != "all":
        random.seed(0)
        codes = random.sample(codes, min(int(mode), len(codes)))
    codes += [0x10FFFF]  # absent: glyph 0
    exe = os.path.join(HERE, "dump")
    if "--no-build" not in sys.argv:
        subprocess.run(["almide", "build", os.path.join(HERE, "dump.almd"), "-o", exe], cwd=ROOT, check=True)
    lines = []
    for i in range(0, len(codes), 2000):
        r = subprocess.run([exe, font_path, str(face)] + [str(c) for c in codes[i:i + 2000]],
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("dump failed: " + r.stderr)
        got = r.stdout.splitlines()
        lines += got if not lines else got[1:]
    head, glyphs = parse(lines)
    order = ft.getGlyphOrder()
    hhea = ft["hhea"]
    want_head = [ft["head"].unitsPerEm, hhea.ascent, hhea.descent, hhea.lineGap, ft["maxp"].numGlyphs]
    bad = 0
    if head != want_head:
        print("header", head, "want", want_head); bad += 1
    gs = ft.getGlyphSet()
    for c in codes:
        name = cmap.get(c, ".notdef")
        gid = ft.getGlyphID(name)
        adv = ft["hmtx"][name][0]
        pen = Rec(gs)
        gs[name].draw(pen)
        if "glyf" in ft and ft["glyf"][name].isComposite():
            # fontTools places a composite unshifted; the phantom-point rule
            # it applies to simple glyphs (lsb - xMin) holds for these too.
            dx = ft["hmtx"][name][1] - ft["glyf"][name].xMin
            pen.contours = [[shift(c[0], dx)] + [(s[0],) + tuple(shift(p, dx) for p in s[1:]) for s in c[1:]]
                            for c in pen.contours]
        g = glyphs.get(c)
        if g is None or g[0] != gid or g[1] != adv:
            print(f"U+{c:04X}: gid/adv {g and g[:2]} want {[gid, adv]}"); bad += 1; continue
        if not same(canon(g[2]), canon(pen.contours)):
            print(f"U+{c:04X} ({name}): outline differs"); bad += 1
    segs = sum(len(s) for g in glyphs.values() for s in g[2])
    print(f"{len(codes)} code points, {segs} segments, {bad} mismatches")
    sys.exit(1 if bad else 0)


main()
