// After bitonic sort, mark the first occurrence of each unique EdgeKey
// and write a pre-weld-index to welded-index mapping.
//
// Input:  sorted intersection records in arena (8 u32 each), key at offsets 4,5.
// Output: weld_map[i] = welded index for pre-weld intersection index i.
//         unique_flags[i] = 1 if record i starts a new unique key, else 0.
//
// After this kernel, a prefix sum on unique_flags gives the welded indices.
// A second pass (weld_assign) writes the actual weld_map values.

struct Params {
    ixn_base: u32,
    ixn_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> arena: array<u32>;
@group(0) @binding(1) var<storage, read_write> unique_flags: array<i32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let n = params.ixn_count;
    let base = params.ixn_base;

    if (tid >= n) { return; }

    if (tid == 0u) {
        unique_flags[0] = 1;
        return;
    }

    let cur_vlo = arena[base + tid * 8u + 4u];
    let cur_vhi = arena[base + tid * 8u + 5u];
    let prev_vlo = arena[base + (tid - 1u) * 8u + 4u];
    let prev_vhi = arena[base + (tid - 1u) * 8u + 5u];

    let is_new = (cur_vlo != prev_vlo) || (cur_vhi != prev_vhi);
    unique_flags[tid] = select(0, 1, is_new);
}
