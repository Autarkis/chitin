// Builds the pre-weld-index to welded-index map and writes
// welded intersection positions.
//
// Uses the corrected welded-index formula:
//   welded_idx = exclusive_scan[i] + unique_flag[i] - 1
//
// For the first occurrence of each unique EdgeKey (flag=1):
//   also copies the intersection position to the welded output.

struct Params {
    ixn_base: u32,
    ixn_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> arena: array<u32>;
@group(0) @binding(1) var<storage, read> excl_scan: array<i32>;
@group(0) @binding(2) var<storage, read> unique_flags: array<i32>;
@group(0) @binding(3) var<storage, read_write> weld_map: array<u32>;
@group(0) @binding(4) var<storage, read_write> welded_positions: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.ixn_count) { return; }

    let base = params.ixn_base;
    let flag = unique_flags[tid];
    let welded_idx = u32(excl_scan[tid]) + u32(flag) - 1u;

    let pre_weld_idx = arena[base + tid * 8u + 3u];
    weld_map[pre_weld_idx] = welded_idx;

    if (flag != 0) {
        let rb = base + tid * 8u;
        welded_positions[welded_idx * 4u] = arena[rb];
        welded_positions[welded_idx * 4u + 1u] = arena[rb + 1u];
        welded_positions[welded_idx * 4u + 2u] = arena[rb + 2u];
        welded_positions[welded_idx * 4u + 3u] = 0u;
    }
}
