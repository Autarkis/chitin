# .phys Binary Format (v2)

Compact binary sidecar for convex collision hulls. Designed for fast loading in game engines and physics runtimes without parsing JSON or USD.

All values are little-endian.

## Header (32 bytes)

| Offset | Size | Type   | Field              |
|--------|------|--------|--------------------|
| 0      | 4    | bytes  | Magic: `PHYS`      |
| 4      | 2    | u16    | Version (currently 2) |
| 6      | 2    | u16    | Flags              |
| 8      | 4    | u32    | Hull count         |
| 12     | 4    | u32    | Total vertex count |
| 16     | 4    | u32    | Total index count  |
| 20     | 4    | u32    | Hull table offset  |
| 24     | 4    | u32    | Vertex data offset |
| 28     | 4    | u32    | Index data offset  |

### Flags

| Bit | Name              | Meaning                                    |
|-----|-------------------|--------------------------------------------|
| 0   | `HAS_BONES`       | Hull descriptors include a bone_index field |
| 1   | `HAS_BIND_POSES`  | Bind pose block appended after index data   |

## Hull Descriptor Table

Starts at `hull_table_offset` (always 32). One descriptor per hull.

**Without HAS_BONES (40 bytes each):**

| Offset | Size | Type     | Field        |
|--------|------|----------|--------------|
| 0      | 4    | u32      | vertex_offset (index into vertex array) |
| 4      | 4    | u32      | vertex_count |
| 8      | 4    | u32      | index_offset (index into index array) |
| 12     | 4    | u32      | index_count  |
| 16     | 12   | float[3] | aabb_min     |
| 28     | 12   | float[3] | aabb_max     |

**With HAS_BONES (44 bytes each):**

Same as above, plus:

| Offset | Size | Type | Field      |
|--------|------|------|------------|
| 40     | 4    | i32  | bone_index (-1 = unassigned) |

## Vertex Data

Starts at `vertex_data_offset`. Packed `int16[3]` per vertex (6 bytes each), quantized against the per-hull AABB.

**Quantization (writer):**
```
normalized = (vertex - aabb_min) / (aabb_max - aabb_min)
quantized  = clamp(normalized * 65535 - 32768, -32768, 32767)
```

**Dequantization (reader):**
```
vertex = (quantized + 32768) / 65535 * (aabb_max - aabb_min) + aabb_min
```

Zero-extent axes (aabb_min == aabb_max) use extent = 1.0 to avoid division by zero. Maximum quantization error per axis is `extent / 65535`.

## Index Data

Starts at `index_data_offset`. Packed `uint16` per index (2 bytes each). Every 3 indices form a triangle. Index values are relative to the hull's vertex_offset, not global.

## Bind Pose Block (optional, HAS_BIND_POSES)

Starts immediately after index data. Present only when flag bit 1 is set.

| Field           | Size          | Type         |
|-----------------|---------------|--------------|
| bone_count      | 4             | u32          |
| *per bone:*     |               |              |
| bind_transform  | 64            | float32[4x4] |
| name_length     | 2             | u16          |
| name            | name_length   | utf-8 bytes  |

Bind transforms are row-major float32 4x4 matrices representing the bone's world-space bind pose. To reconstruct world-space vertices from bone-local hull vertices: `world = local @ bind_transform`.

## Layout Invariants

These must hold for a valid file:

- `hull_table_offset == 32`
- `vertex_data_offset == hull_table_offset + hull_count * descriptor_size`
- `index_data_offset == vertex_data_offset + total_vertex_count * 6`
- Sum of all hull vertex_counts == total_vertex_count
- Sum of all hull index_counts == total_index_count
- All index values < their hull's vertex_count
- All index_counts divisible by 3
- aabb_min <= aabb_max per component

## Limits

- Vertex coordinates: int16 quantized (65536 levels per axis per hull)
- Indices: uint16 (max 65535 vertices per hull)
- Hull count, total vertices, total indices: uint32
- Bone names: utf-8, max 65535 bytes

## Versioning

Version 2 is the current and only supported version. Readers should reject files with version > 2. Future versions may extend the format by adding new flag bits or appending new blocks after existing data.
