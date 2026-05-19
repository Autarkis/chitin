# .phys Binary Format

Compact binary sidecar for convex collision hulls. Designed for fast loading in game engines and physics runtimes without parsing JSON or USD.

All values are little-endian.

## Header (32 bytes)

| Offset | Size | Type   | Field              |
|--------|------|--------|--------------------|
| 0      | 4    | bytes  | Magic: `PHYS`      |
| 4      | 2    | u16    | Version            |
| 6      | 2    | u16    | Flags              |
| 8      | 4    | u32    | Hull count         |
| 12     | 4    | u32    | Total vertex count |
| 16     | 4    | u32    | Total index count  |
| 20     | 4    | u32    | Hull table offset  |
| 24     | 4    | u32    | Vertex data offset |
| 28     | 4    | u32    | Index data offset  |

The header describes **LOD 0** (highest detail). All counts and offsets refer to LOD 0's data only. Flags determine which optional blocks are present after the core data.

### Flags

| Bit | Name              | Meaning                                    |
|-----|-------------------|--------------------------------------------|
| 0   | `HAS_BONES`       | Hull descriptors include a bone_index field |
| 1   | `HAS_BIND_POSES`  | Bind pose block appended after index data   |
| 2   | `HAS_LOD`         | LOD block appended after bind pose block    |

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

## LOD Block (optional, HAS_LOD)

Starts immediately after the bind pose block (or after index data if no bind poses). Present only when flag bit 2 is set.

LOD 0 is the main body described by the header -- the highest detail decomposition. The LOD block contains additional tiers at increasing concavity (decreasing detail). Each tier is a complete, independent decomposition -- not a decimation of the previous tier.

### LOD Block Header

| Field       | Size | Type |
|-------------|------|------|
| tier_count  | 4    | u32  |

### Per Tier (tier_count entries)

**Tier Header (24 bytes):**

| Offset | Size | Type | Field                                                |
|--------|------|------|------------------------------------------------------|
| 0      | 4    | f32  | Concavity threshold used to generate this tier       |
| 4      | 4    | u32  | Hull count                                           |
| 8      | 4    | u32  | Total vertex count                                   |
| 12     | 4    | u32  | Total index count                                    |
| 16     | 4    | u32  | Data size (bytes of descriptors + vertices + indices) |
| 20     | 4    | u32  | Reserved (0)                                         |

**Tier Data** follows immediately after each tier header:

| Section          | Size                                    |
|------------------|-----------------------------------------|
| Hull descriptors | hull_count * descriptor_size bytes      |
| Vertex data      | total_vertex_count * 6 bytes (int16[3]) |
| Index data       | total_index_count * 2 bytes (uint16)    |

Descriptor size follows the same rule as the main body (44 if `HAS_BONES`, 40 otherwise). Hull descriptors, vertex data, and index data use the same layout as LOD 0 -- offsets within each tier are relative to that tier's data, not global.

`data_size` must equal `hull_count * descriptor_size + total_vertex_count * 6 + total_index_count * 2`. This lets readers skip tiers without parsing their contents.

### LOD Ordering

Tiers are stored in order of decreasing detail (increasing concavity):

- **LOD 0** (main body): highest detail, lowest concavity, most hulls
- **LOD 1** (first tier in block): less detail
- **LOD N** (last tier): coarsest, fewest hulls

A reader that wants the cheapest option reads the last tier.

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

When `HAS_LOD` is set, the same invariants apply within each LOD tier (vertex/index sums match tier header counts, index bounds valid, etc.). Additionally:

- `data_size == hull_count * descriptor_size + total_vertex_count * 6 + total_index_count * 2` per tier
- Concavity values should be in ascending order across tiers

## Limits

- Vertex coordinates: int16 quantized (65536 levels per axis per hull)
- Indices: uint16 (max 65535 vertices per hull)
- Hull count, total vertices, total indices: uint32
- Bone names: utf-8, max 65535 bytes

## Block Order

Blocks appear in this order when present. Each block starts immediately after the previous one -- no padding.

1. Header (always, 32 bytes)
2. Hull descriptor table (always)
3. Vertex data (always)
4. Index data (always)
5. Bind pose block (if `HAS_BIND_POSES`)
6. LOD block (if `HAS_LOD`)
