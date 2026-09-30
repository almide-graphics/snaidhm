"""Build the synthetic font collection sfnt_test.almd reads, and its goldens.

  python3 test/font-oracle/make_fixture.py

Writes src/sfnt_test.almd: a four-face TrueType Collection (base64) made to
walk every path of sfnt.almd, cff.almd and truetype.almd, and per face the
text sfnt_describe.almd must produce. The goldens are this implementation's
own output — written only after oracle.py has checked every one of them
against fontTools, so they are fontTools' answers in sfnt_describe's form.

  face 0  CID-keyed CFF, FDSelect format 3, cmap 4 + 12
  face 1  CID-keyed CFF, FDSelect format 0
  face 2  name-keyed CFF (Private DICT on the Top DICT)
  face 3  TrueType glyf: composites (offset, scale, x/y scale, 2x2, nested,
          word arguments), contours starting off-curve and all off-curve,
          hmtx lsb unlike xMin; cmap 4 only

The CFF table is encoded here by hand — fontTools only decodes it — so the
two readers under comparison share no encoder. Charstrings are compiled from
fontTools programs (T2CharString), which leaves every operator as written.
"""
import base64, os, struct, subprocess, sys
from fontTools.fontBuilder import FontBuilder
from fontTools.misc.psCharStrings import T2CharString
from fontTools.ttLib import TTFont, TTCollection, newTable
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.ttLib.tables._g_l_y_f import Glyph, GlyphComponent, GlyphCoordinates

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT_FONT = os.path.join(HERE, "fixture.ttc")
OUT_TEST = os.path.join(ROOT, "src", "sfnt_test.almd")

# ── CFF encoding ──


def index(items):
    items = [bytes(i) for i in items]
    if not items:
        return b"\x00\x00"
    offs = [1]
    for i in items:
        offs.append(offs[-1] + len(i))
    size = 1 if offs[-1] < 0x100 else 2 if offs[-1] < 0x10000 else 3 if offs[-1] < 0x1000000 else 4
    out = struct.pack(">HB", len(items), size)
    for o in offs:
        out += o.to_bytes(size, "big")
    return out + b"".join(items)


def dict_num(v):
    # Offsets and sizes as the 5-byte form so a DICT's length never depends
    # on the values it holds (the layout is computed before they are known).
    return b"\x1d" + struct.pack(">i", v)


def dict_int(v):
    if -107 <= v <= 107:
        return bytes([v + 139])
    if 108 <= v <= 1131:
        v -= 108
        return bytes([(v >> 8) + 247, v & 0xFF])
    if -1131 <= v <= -108:
        v = -v - 108
        return bytes([(v >> 8) + 251, v & 0xFF])
    if -32768 <= v <= 32767:
        return b"\x1c" + struct.pack(">h", v)
    return dict_num(v)


def dict_real(s):
    nib = {".": 0xA, "E": 0xB, "-": 0xE}
    ns = []
    i = 0
    while i < len(s):
        if s[i : i + 2] == "E-":
            ns.append(0xC)
            i += 2
            continue
        ns.append(nib[s[i]] if s[i] in nib else int(s[i]))
        i += 1
    ns.append(0xF)
    if len(ns) % 2:
        ns.append(0xF)
    return b"\x1e" + bytes(ns[k] * 16 + ns[k + 1] for k in range(0, len(ns), 2))


def op(code):
    return bytes([12, code - 1200]) if code >= 1200 else bytes([code])


def charstring(program):
    cs = T2CharString(program=program)
    cs.compile()
    return cs.bytecode


def cff_table(name, glyphs, gsubrs, fds, fd_of, cid, fdselect_format):
    """glyphs: charstring programs; gsubrs: programs; fds: per font DICT its
    local subr programs; fd_of: gid -> fd (CID) — for a name-keyed font fds
    has one entry, the Top DICT's Private."""
    n = len(glyphs)
    header = b"\x01\x00\x04\x04"
    names = index([name.encode()])
    strings = index([b"Adobe", b"Identity"] if cid else [])
    gsub = index([charstring(p) for p in gsubrs])
    charstrings = index([charstring(p) for p in glyphs])
    if cid:
        charset = b"\x02" + struct.pack(">HH", 1, n - 2)  # CIDs 1..n-1
    else:
        charset = b""  # ISOAdobe, predefined (offset 0)
    if fdselect_format == 3:
        ranges = []
        for g in range(n):
            if not ranges or ranges[-1][1] != fd_of(g):
                ranges.append((g, fd_of(g)))
        fdselect = b"\x03" + struct.pack(">H", len(ranges))
        fdselect += b"".join(struct.pack(">HB", g, f) for g, f in ranges) + struct.pack(">H", n)
    else:
        fdselect = b"\x00" + bytes(fd_of(g) for g in range(n))

    def private(subrs):
        # nominalWidthX / defaultWidthX, then Subrs (offset from the Private
        # DICT) when there are any.
        body = dict_int(500) + op(20) + dict_real("1.5") + op(21)
        if not subrs:
            return body, b""
        size = len(body) + 6
        return body + dict_num(size) + op(19), index([charstring(p) for p in subrs])

    privs = [private(s) for s in fds]

    def top(cs_at, charset_at, fdarray_at, fdselect_at, priv_size, priv_at):
        d = b""
        if cid:
            d += dict_int(391) + dict_int(392) + dict_int(0) + op(1230)
            d += dict_int(n) + op(1234)
        d += dict_int(1) + dict_int(2) + dict_real("0.5") + dict_real("-0.25") + op(5)  # FontBBox
        d += dict_num(charset_at) + op(15)
        d += dict_num(cs_at) + op(17)
        if cid:
            d += dict_num(fdarray_at) + op(1236) + dict_num(fdselect_at) + op(1237)
        else:
            d += dict_num(priv_size) + dict_num(priv_at) + op(18)
        return d

    probe = index([top(0, 0, 0, 0, 0, 0)])
    at = len(header) + len(names) + len(probe) + len(strings) + len(gsub)
    charset_at = at if cid else 0
    at += len(charset)
    fdselect_at = at if cid else 0
    at += len(fdselect) if cid else 0
    cs_at = at
    at += len(charstrings)
    # Private DICTs then their subrs, each after the other.
    priv_at = []
    for body, subrs in privs:
        priv_at.append(at)
        at += len(body) + len(subrs)
    fdarray_at = at
    if cid:
        fd_dicts = [dict_num(len(body)) + dict_num(pa) + op(18) for (body, _), pa in zip(privs, priv_at)]
        fdarray = index(fd_dicts)
    else:
        fdarray = b""
    tops = index([top(cs_at, charset_at, fdarray_at, fdselect_at, len(privs[0][0]), priv_at[0])])
    assert len(tops) == len(probe)
    out = header + names + tops + strings + gsub + charset + (fdselect if cid else b"") + charstrings
    for body, subrs in privs:
        out += body + subrs
    return out + fdarray


# ── Charstrings ──

hm = "hintmask"


def cff_glyphs():
    """Eight glyphs, every Type 2 path operator, hints, widths and calls."""
    return [
        # 0 .notdef: width, then endchar alone.
        [120, "endchar"],
        # 1: stems with a width, hintmask, every line form, odd and even
        # h/vlineto, 16.16 fixed and two-byte numbers.
        [37, 10, 20, 300, 40, "hstemhm", 50, 60, hm, b"\xc0", 100, 0, "rmoveto",
         500, 0, 0, 400.5, -250.25, 120, "rlineto", -150, 60, -100, "hlineto",
         -80, 30, "vlineto", 1000, -1000, "rlineto", "endchar"],
        # 2: hmoveto with a width, vstem, hintmask whose stems are implied,
        # every curve form, a second contour.
        [-20, 50, "hmoveto", 10, 30, "vstem", 70, 20, hm, b"\x80",
         100, 50, 50, 100, 100, "hvcurveto",
         20, 40, 60, 80, 10, 20, 30, 40, "vhcurveto",
         5, 10, 20, 30, 40, 10, 20, 30, 40, "hhcurveto",
         7, 10, 20, 30, 40, "vvcurveto",
         10, 20, 30, 40, 50, 60, 70, 80, "rcurveline",
         10, 20, 30, 40, 50, 60, 70, 80, "rlinecurve",
         10, 20, 30, 40, 50, 60, "rrcurveto",
         0, -300, "rmoveto", 40, 0, 0, 40, -40, 0, "rlineto", "endchar"],
        # 3: a local subroutine (FD 1's) that calls a global one (bias 1131).
        [300, 300, "rmoveto", -107, "callsubr", "endchar"],
        # 4: vmoveto with a width, the four flex forms, cntrmask.
        [700, 400, "vmoveto", 1, 2, 3, 4, "hstemhm", "cntrmask", b"\xc0",
         10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 50, "flex",
         30, 20, 10, 40, 50, 60, 70, "hflex",
         10, 5, 20, 10, 30, 40, 20, -8, 30, "hflex1",
         10, 60, 20, 50, 30, 40, 10, 40, 20, 30, 100, "flex1",
         60, 10, 50, 20, 40, 30, 40, 10, 30, 20, 100, "flex1",
         "endchar"],
        # 5: rmoveto with a width; a global subroutine that ends the glyph
        # (endchar, no return).
        [555, 100, 100, "rmoveto", 1, "callgsubr"],
        # 6: FD 0's local subroutine 0 — not the one glyph 3 reaches.
        [0, 0, "rmoveto", -107, "callsubr", "endchar"],
        # 7: nothing at all.
        ["endchar"],
    ]


def cff_gsubrs():
    # 1300 of them, so the bias is 1131: index -1131 is subr 0.
    subrs = [["return"]] * 1300
    subrs = list(subrs)
    subrs[0] = [50, 0, 0, 50, -50, 0, "rlineto", "return"]
    subrs[1132] = [200, 0, 0, 200, -200, 0, "rlineto", "endchar"]
    return subrs


# Local subr 0 of FD 1 calls global subr 0 (index -1131 with the 1131 bias);
# FD 0's draws something else. With 108 local subrs the bias is 107: index
# -107 is subr 0.
FD_SUBRS = [
    [[25, 25, 0, -25, "rlineto", "return"]] + [["return"]] * 107,
    [[-1131, "callgsubr", 10, 10, "rlineto", "return"]] + [["return"]] * 107,
]


def fd_of(g):
    return 1 if g in (3, 4) else 0


def cff_font(name, cid, fdselect_format):
    glyphs = cff_glyphs()
    n = len(glyphs)
    if cid:
        order = [".notdef"] + ["cid%05d" % i for i in range(1, n)]
        raw = cff_table(name, glyphs, cff_gsubrs(), FD_SUBRS, fd_of, True, fdselect_format)
    else:
        # ISOAdobe names; one Private, FD 1's subrs (glyphs 3 and 6 both use
        # them here).
        order = [".notdef", "space", "exclam", "quotedbl", "numbersign", "dollar", "percent", "ampersand"]
        raw = cff_table(name, glyphs, cff_gsubrs(), [FD_SUBRS[1]], lambda g: 0, False, 0)
    fb = FontBuilder(1000, isTTF=False)
    fb.setupGlyphOrder(order)
    # BMP code points map through format 4 — U+3042..U+3049 through its glyph
    # index array, their glyphs being out of order; U+20B9F needs format 12.
    cmap = {0x41: 1, 0x42: 2, 0x20B9F: 7}
    for i, g in enumerate([3, 5, 4, 6, 3, 5, 4, 6]):
        cmap[0x3042 + i] = g
    fb.setupCharacterMap({c: order[g] for c, g in cmap.items()})
    t = DefaultTable("CFF ")
    t.data = raw
    fb.font["CFF "] = t
    fb.setupHorizontalMetrics({g: (600 + 10 * i, 0) for i, g in enumerate(order)})
    fb.setupHorizontalHeader(ascent=880, descent=-120)
    fb.font["hhea"].lineGap = 90
    fb.setupNameTable({"familyName": name, "styleName": "Regular"})
    fb.setupOS2()
    fb.setupPost()
    fb.setupMaxp()
    # The CFF table is raw bytes to fontTools here, so head's bounding box
    # stays as set rather than being recomputed from it.
    fb.font.recalcBBoxes = False
    return fb.font


# ── TrueType ──


def simple(points, ends):
    """points: (x, y, on_curve)."""
    g = Glyph()
    g.numberOfContours = len(ends)
    g.coordinates = GlyphCoordinates([(x, y) for x, y, _ in points])
    g.flags = bytearray(1 if on else 0 for _, _, on in points)
    g.endPtsOfContours = ends
    g.program = None
    from fontTools.ttLib.tables import ttProgram
    g.program = ttProgram.Program()
    g.program.fromBytecode(b"")
    return g


def composite(parts):
    """parts: (glyph name, dx, dy, transform or None)."""
    g = Glyph()
    g.numberOfContours = -1
    g.components = []
    for name, dx, dy, tr in parts:
        c = GlyphComponent()
        c.glyphName = name
        c.x, c.y = dx, dy
        c.flags = 0x4  # ROUND_XY_TO_GRID, as fonts usually set it
        if tr is not None:
            c.transform = tr
        g.components.append(c)
    return g


def glyf_font():
    order = [".notdef", "box", "curvy", "offstart", "alloff", "ring", "boxring", "half", "wide",
             "rot", "nested", "far", "turned"]
    box = simple([(50, 0, 1), (450, 0, 1), (450, 600, 1), (50, 600, 1)], [3])
    # Quadratics: two off-curve points in a row imply the midpoint.
    curvy = simple([(100, 0, 1), (300, 0, 0), (400, 200, 0), (400, 400, 1), (200, 600, 0), (100, 400, 1),
                    (150, 100, 1), (250, 150, 0), (200, 250, 1)], [5, 8])
    # Starts off-curve: the contour begins at its first on-curve point.
    offstart = simple([(0, 300, 0), (0, 0, 1), (300, 0, 0), (300, 300, 1)], [3])
    # No on-curve point at all: starts halfway between the last and first.
    alloff = simple([(0, 0, 0), (400, 0, 0), (400, 400, 0), (0, 400, 0)], [3])
    ring = simple([(0, 0, 1), (100, 0, 0), (100, 100, 1), (0, 100, 0)], [3])
    glyphs = {
        ".notdef": simple([(0, 0, 1), (500, 0, 1), (500, 700, 1)], [2]),
        "box": box, "curvy": curvy, "offstart": offstart, "alloff": alloff, "ring": ring,
        "boxring": composite([("box", 0, 0, None), ("ring", 175, 650, None)]),
        "half": composite([("box", 20, -10, [[0.5, 0], [0, 0.5]])]),
        "wide": composite([("curvy", 0, 0, [[1.5, 0], [0, 0.75]])]),
        "rot": composite([("offstart", 300, 50, [[0, 1], [-1, 0]]), ("alloff", -5, 3, [[0.5, 0.25], [-0.25, 0.5]])]),
        "nested": composite([("boxring", 10, 20, [[0.5, 0], [0, 0.5]]), ("half", -40, 40, None)]),
        # Offsets beyond a byte: word arguments.
        "far": composite([("ring", 1000, -300, None), ("box", -200, 400, None)]),
        # A composite under a rotation: each transform composes with the one
        # above it.
        "turned": composite([("half", 100, 0, [[0.8, 0.6], [-0.6, 0.8]])]),
    }
    fb = FontBuilder(2048, isTTF=True)
    fb.setupGlyphOrder(order)
    # BMP only, so format 4 alone; U+3042..U+3049 go through its glyph index
    # array.
    cmap = {0x41 + i: g for i, g in enumerate(order[1:])}
    for i, g in enumerate(["ring", "box", "turned", "curvy", "ring", "far", "alloff", "box"]):
        cmap[0x3042 + i] = g
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyphs)
    glyf = fb.font["glyf"]
    metrics = {}
    for i, g in enumerate(order):
        glyf[g].recalcBounds(glyf)
        xmin = getattr(glyf[g], "xMin", 0)
        # Two glyphs, one simple and one composite, whose lsb is not their
        # xMin: the outline moves by the difference.
        lsb = xmin + (30 if g in ("curvy", "boxring") else 0)
        metrics[g] = (700 + 5 * i, lsb)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=1900, descent=-500)
    fb.setupNameTable({"familyName": "Fixture Glyf", "styleName": "Regular"})
    fb.setupOS2()
    fb.setupPost()
    fb.setupMaxp()
    return fb.font


# ── Collection, oracle, goldens ──

CODES = [0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D,
         0x3042, 0x3043, 0x3044, 0x3045, 0x3046, 0x3047, 0x3048, 0x3049, 0x20B9F, 0x7A]


def main():
    fonts = [cff_font("FixtureCID3", True, 3), cff_font("FixtureCID0", True, 0),
             cff_font("FixtureNamed", False, 0), glyf_font()]
    coll = TTCollection()
    coll.fonts = fonts
    coll.save(OUT_FONT)
    goldens = []
    for face in range(len(fonts)):
        r = subprocess.run([sys.executable, os.path.join(HERE, "oracle.py"), OUT_FONT, str(face), "--codes", "all"],
                           capture_output=True, text=True)
        print(r.stdout.strip().splitlines()[-1])
        if r.returncode != 0:
            sys.exit(f"face {face}: the oracle disagrees\n{r.stdout}{r.stderr}")
        dump = subprocess.run([os.path.join(HERE, "dump"), OUT_FONT, str(face)] + [str(c) for c in CODES],
                              capture_output=True, text=True, check=True).stdout
        goldens.append(dump)
    data = base64.b64encode(open(OUT_FONT, "rb").read()).decode()
    write_test(data, goldens)
    print(f"wrote {os.path.relpath(OUT_TEST, ROOT)} ({len(data)} base64 bytes)")


def write_test(data, goldens):
    lines = [
        "// sfnt tests — generated by test/font-oracle/make_fixture.py; do not edit.",
        "//",
        "// A four-face collection built to reach every path of sfnt, cff and",
        "// truetype (see the generator for what each face holds), and per face",
        "// the description fontTools agrees with.",
        "",
        "import base64",
        "import self.sfnt as sfnt",
        "import self.sfnt_describe as describe",
        "",
        "let CODES = [" + ", ".join(str(c) for c in CODES) + "]",
        "",
        "let FIXTURE = \"" + data + "\"",
        "",
    ]
    lines += [
        "fn data() -> Bytes = base64.decode(FIXTURE) ?? bytes.new(0)",
        "",
        "fn check(face: Int, want: String) -> Unit = match sfnt.load(data(), face) {",
        "  ok(f) => assert_eq(describe.describe(f, CODES), want),",
        "  err(e) => assert_eq(e, \"a loaded face\"),",
        "}",
    ]
    names = ["CID-keyed CFF, FDSelect format 3", "CID-keyed CFF, FDSelect format 0",
             "name-keyed CFF", "TrueType glyf"]
    for face, (name, g) in enumerate(zip(names, goldens)):
        body = "\n".join("  " + ln for ln in g.rstrip("\n").split("\n"))
        lines += ["", f"test \"face {face}: {name}\" {{", f"  check({face}, \"\"\"", body, "  \"\"\" + \"\\n\")", "}"]
    lines += [
        "",
        "test \"face count and an absent face\" {",
        "  assert_eq(sfnt.face_count(data()), 4)",
        "  assert_eq(sfnt.load(data(), 4) |> result.is_err, true)",
        "}",
        "",
        "test \"not a font\" {",
        "  assert_eq(sfnt.load(bytes.new(4), 0) |> result.is_err, true)",
        "}",
    ]
    open(OUT_TEST, "w").write("\n".join(lines) + "\n")


main()
