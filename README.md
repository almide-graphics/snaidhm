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
├── text/      Font parser (OTF/TTF → glyph outlines → Path)
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
