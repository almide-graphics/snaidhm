# Vello Architecture Study Notes

Reference: https://github.com/linebender/vello (cloned to /tmp/vello)

## Pipeline (13 compute passes + 1 render)

```
1. pathtag_reduce   → path tag stream reduction (monoid scan)
2. pathtag_scan     → path tag prefix sum (parallel scan)
3. bbox_clear       → clear bounding boxes
4. flatten          → cubic bezier → line segments (923 LOC!)
5. draw_reduce      → draw object reduction
6. draw_leaf        → draw object leaf processing
7. clip_reduce      → clip stack reduction
8. clip_leaf        → clip stack leaf processing
9. binning          → assign draw objects to tiles (coarse)
10. tile_alloc      → allocate tile segment lists
11. path_count      → count segments per tile
12. path_tiling     → write segment data per tile
13. backdrop        → PREFIX SUM of winding across tile rows ← KEY
14. coarse          → build per-tile command lists (PTCL)
15. fine            → per-pixel rasterization (1402 LOC)
```

## Key Insight: Backdrop (prefix sum)

The "horizontal lines" bug in snaidhm is caused by missing backdrop propagation.

### Problem
When a path edge crosses at tile column 5, all tiles at columns 0-4 in the same
row should have winding=1 (inside the path). But each tile computes winding
independently and doesn't know about edges in other tiles.

### Vello's Solution: `backdrop.wgsl`
Each tile stores a `backdrop` integer = the winding delta from segments crossing
the LEFT edge of that tile. Then a **prefix sum** across the tile row propagates
the winding from left to right.

```wgsl
// backdrop.wgsl — 30 lines
// workgroup prefix sum across tile columns
sh_backdrop[local_id.x] = tiles[ix].backdrop;
for (var i = 0u; i < firstTrailingBit(WG_SIZE); i += 1u) {
    workgroupBarrier();
    if local_id.x >= (1u << i) {
        backdrop += sh_backdrop[local_id.x - (1u << i)];
    }
    workgroupBarrier();
    sh_backdrop[local_id.x] = backdrop;
}
tiles[ix].backdrop = backdrop;
```

After this, each tile's `backdrop` is the accumulated winding at its left edge
from all paths. The fine rasterizer uses this as the starting winding number.

### What snaidhm needs
1. During coarse pass: compute per-tile backdrop delta (count how many times
   path edges cross the left boundary of each tile)
2. Add backdrop prefix sum pass (horizontal, per tile row)
3. In fine pass: initialize winding with tile's backdrop value

## Key Insight: Area-based AA (not distance-based)

Vello's fine rasterizer doesn't use `smoothstep(distance)` for AA.
It computes exact **area coverage** of each pixel.

For each line segment crossing a pixel column:
- Compute the signed area of the trapezoid formed by the segment and pixel edges
- This gives exact fractional coverage (0.0 to 1.0)
- No supersampling needed for analytical AA

The area calculation in fine.wgsl:
```
area[i] += y_edge * (1.0 - x_fract) + 0.5 * sign * abs(dy);
```

Where `y_edge` is the Y extent of the segment within the pixel column,
and `x_fract` is the fractional X position within the pixel.

## File Sizes (complexity indicator)

| File | Lines | Complexity |
|------|-------|-----------|
| fine.wgsl | 1402 | Very high — MSAA, blending, images, gradients, clips |
| flatten.wgsl | 923 | High — Euler spiral flattening, robust subdivision |
| coarse.wgsl | 469 | Medium — PTCL generation |
| draw_leaf.wgsl | 303 | Medium |
| path_count.wgsl | 202 | Medium |
| backdrop.wgsl | 35 | Simple — just a prefix sum |

## What to steal for snaidhm

### Immediate (fixes current bugs)
1. **Backdrop prefix sum** — fixes the "horizontal lines" artifact
2. **Area-based AA** — replaces our broken smoothstep approach

### Near-term
3. **Flatten on GPU** — Vello's flatten.wgsl uses Euler spiral approximation
   (more robust than Wang's formula for degenerate curves)
4. **PTCL (Per-Tile Command List)** — instead of storing segment indices per tile,
   store a command stream (fill, stroke, begin_clip, end_clip, solid, gradient, image)

### Long-term
5. **Monoid scan** — parallel prefix sum for path tags, enabling fully GPU-driven pipeline
6. **MSAA** — 8x/16x multisampled AA using lookup tables
7. **Blend modes** — full Porter-Duff + advanced blend
8. **Clip paths** — nested clip stack with reduction
