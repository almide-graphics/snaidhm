> **Moved.** snaidhm is now developed in [almide-graphics/ceangal2](https://github.com/almide-graphics/ceangal2) (`snaidhm/`), rebuilt around one host ABI with per-context GPU state and a WebGPU / wgpu host for every platform. This repository is archived and kept for reference; existing git dependencies keep working.

# snaidhm

> Irish: *snaidhm* — knot. As in Celtic knots: interleaving paths with no beginning and no end.

Compute-shader-first 2D/3D renderer for [Almide](https://github.com/almide/almide), built on WebGPU.

## Status

Phase 0 — bootstrapping.

## Architecture

```
snaidhm
├── gpu/       WebGPU abstraction, frame arena, pipeline management
├── path/      2D path renderer (flatten → tile → sort → fine raster)
├── mesh/      3D vertex pipeline
├── text/      Font reader (OpenType/TrueType → glyph outlines → Path)
└── image/     Texture decode
```

## Design

- **GPU compute-first** — no CPU rasterizer, all path rendering via compute shaders
- **Frame arena** — GPU buffers default to per-frame lifetime, `@persistent` for long-lived resources
- **Single language** — host code and shaders both written in Almide (`@gpu fn` → WGSL)
- **WebGPU only** — wgpu (native) / WebGPU API (browser), no manual Vulkan/Metal/D3D12
- **Paths-are-paths** — text glyphs are paths, same pipeline for shapes and text

## Dependencies

- [lumen](https://github.com/almide-graphics/lumen) — vec/mat/color/quaternion math
- Almide compiler with `@gpu` → WGSL codegen (`almide/almide`)

## Roadmap

| Phase | Goal |
|-------|------|
| 0 | Triangle — prove `@gpu fn` → WGSL + auto bind group |
| 0.3 | Rotating cube — uniform buffer, lumen.mat4 in `@gpu` |
| 0.5 | 10K particles — compute → vertex, frame arena, double buffer |
| 0.7 | 1000 circles — flatten → tile → raster pipeline |
| 1 | Full 2D path renderer (bézier, stroke, fill, AA) |
| 2 | Font parser + text rendering (glyph outlines as paths) |
| 3 | nendo integration (VRM/glTF on snaidhm) |

## License

TBD

## Host

`src/web/gpu.almd` declares the `gpu` extern namespace; `host/gpu.js` is its
browser implementation. `host/MANIFEST` lists what a consumer copies into the
directory it serves.

```js
import { createGpuHost } from "./gpu.js";

const host = createGpuHost(canvas);
host.setFormat(navigator.gpu.getPreferredCanvasFormat());
const shaderIndex = host.registerShader(myWgsl);   // what create_shader resolves

const { instance } = await WebAssembly.instantiate(bytes, { gpu: host.imports, /* … */ });
host.setMemory(instance.exports.memory);

function frame() {
  host.beginFrame();      // resets per-frame clear ownership
  // … drive the module …
  requestAnimationFrame(frame);
}
```

`node test/host-contract.mjs` checks the host implements every extern the
contract declares — a missing one is a `LinkError` at instantiation, not a build
failure, so it is worth catching in CI.

## Fonts

`src/sfnt.almd` reads fonts from their bytes, in Almide on every target:
TrueType and OpenType, single fonts and collections (`.ttc`), glyf outlines
(`truetype.almd`) and CFF outlines, CID-keyed or not (`cff.almd`) — the form
Japanese and other CJK fonts such as Hiragino and Noto Sans CJK take — with
character maps in formats 12 and 4. A text layer (`text.almd`) draws with a
list of fonts, each character from the first that has it.

A host only reads the file. Natively `snaidhm.native.fonts.system()` finds a
Latin and a Japanese face by path; in the browser the page hands its fonts to
the module as WASI files, which the program reads with the same
`fs.read_bytes_raw`:

```js
import { createWasi } from "./wasi.js";

const wasi = createWasi({ files: { "/fonts/ui.otf": await (await fetch("ui.otf")).arrayBuffer() } });
const { instance } = await WebAssembly.instantiate(bytes, { wasi_snapshot_preview1: wasi.imports, /* … */ });
wasi.setMemory(instance.exports.memory);
```

`src/sfnt_test.almd` runs the reader over a font collection built to reach
every path of it (`test/font-oracle/make_fixture.py`), against outputs checked
with fontTools. `python3 test/font-oracle/oracle.py FONT [FACE] [--codes all]`
compares any installed font the same way.

## Wayland, without Rust

`snaidhm.wayland.window` is a desktop window over the Wayland protocol, written
in Almide down to the socket: the wire format (`wayland/wire.almd`), the
connection (`wayland/client.almd`, over `net`'s Unix-socket and shared-memory
primitives), the registry, xdg-shell toplevel, double-buffered wl_shm pixels,
pointer input, and the keyboard read through the XKB keymap the compositor
hands over (`wayland/xkb.almd`). Pixels are drawn on the CPU by
`snaidhm.cpu.canvas` — rectangles and text from the same glyph rasterizer the
GPU text layer uses — so nothing under the program is Rust but the language
runtime. `snaidhm.native.window` (winit + wgpu) stays the GPU path.

```sh
almide run examples/wayland/main.almd          # on a Wayland desktop
WAYLAND_DEBUG=1 almide run examples/wayland/main.almd   # trace every message
```

`test/wayland/run.sh` runs it on a headless sway in Docker, types into it,
drives a pointer over it (`examples/wayland/drive.almd`, a virtual pointer
written with the same client) and saves a screenshot.
