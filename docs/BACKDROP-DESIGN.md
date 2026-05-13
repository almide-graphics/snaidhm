# Backdrop Implementation — Vello Deep Dive

## How Vello Computes Backdrop (path_count.wgsl)

### Key insight
Backdrop delta is added in TWO situations:

#### 1. Segment entirely left of bbox (line 165-170)
When a segment's x-range is entirely left of the path's bounding box:
```
// The segment doesn't touch any tile, but it still affects winding.
// Add backdrop delta to ALL tile rows the segment's Y range covers.
for y in ymin..ymax:
    tile[row_start + 0].backdrop += delta  // leftmost tile of bbox
```

#### 2. Segment enters a tile from the top (line 185-187)
When a segment transitions from one tile column to the next (crosses a vertical tile boundary from the top):
```
if top_edge and x + 1 < bbox.z:
    tile[base + x + 1].backdrop += delta  // tile to the RIGHT
```

### Delta sign convention
```
delta = select(1, -1, is_down)
```
- **is_down** = segment goes downward (in pixel Y = screen coordinates)
- Downward segment crossing a tile boundary: delta = -1
- Upward segment: delta = +1

This is because Vello's winding convention counts **leftward** crossings (the ray goes to the left, or equivalently, the "inside" is to the right of a downward segment).

### After path_count: backdrop.wgsl
```
// Inclusive prefix sum across tile columns, per row.
// After this, each tile's backdrop = total winding from all segments
// to its left (that didn't get assigned to the tile).
```

### What this means for fine.wgsl
```
// In the fine rasterizer, each tile starts with:
let initial_winding = tile.backdrop;
// Then adds local winding from segments IN the tile.
// Total winding = backdrop + local_winding
```

## snaidhm Translation

### Current approach (correct but slow)
Assign segments to ALL tiles from column 0 to maxX.
This implicitly handles backdrop (every tile sees every segment to its right).

### Target approach (Vello-style, fast)
1. Assign segments to AABB-overlapping tiles only
2. When a segment crosses a tile's LEFT EDGE (vertical boundary):
   - Add backdrop delta to that tile
   - Delta sign: based on segment direction (up/down in NDC)
3. For segments entirely left of their path's tile range:
   - Add backdrop delta to all rows in the leftmost tile column
4. Prefix sum backdrop across tile columns per row
5. Fine pass: initialize winding = backdrop

### Key difference from our first attempt
Our first attempt checked if segments crossed tile boundaries during coarse assignment. The issue was sign convention — we need to match Vello's convention:

In pixel coordinates (Y down):
- Segment going DOWN (py increases): delta = -1
- Segment going UP (py decreases): delta = +1

In NDC (Y up, our internal format):
- Segment going UP (y increases): delta = +1
- Segment going DOWN (y decreases): delta = -1

The crossing detection should happen at tile LEFT EDGES (vertical lines at x = tile_col * TILE_SIZE in pixel coords).

### Implementation plan
1. During CPU coarse pass:
   - For each segment, iterate through tile columns in its AABB
   - At each tile column boundary (left edge), check if segment crosses it
   - A segment from (px0, py0) to (px1, py1) crosses x = tileLeftPx if:
     - min(px0, px1) < tileLeftPx <= max(px0, px1)
   - Crossing y = py0 + (tileLeftPx - px0) / (px1 - px0) * (py1 - py0)
   - Add delta to backdrop[crossingTileRow][tileCol]
2. Also handle segments entirely left of first overlapping tile
3. Prefix sum across columns per row
4. Upload backdrop buffer to GPU
5. Fine pass reads backdrop as initial winding

## Text Quality — Separate Issue

The gray horizontal lines in text are NOT caused by backdrop.
They appear even with the "all columns left" approach.
Root cause is likely:
- TTF glyph outlines have very thin features that create near-zero-area regions
- The winding number flickers between 0 and 1 at sub-pixel level
- 4x supersampling helps but 4 samples isn't enough for thin strokes

Solutions:
1. **8x/16x MSAA** (Vello's approach) — more samples per pixel
2. **SDF atlas** (Figma's approach) — pre-render glyphs as signed distance fields
3. **Analytical area coverage** (Vello fine.wgsl) — exact area instead of point sampling
