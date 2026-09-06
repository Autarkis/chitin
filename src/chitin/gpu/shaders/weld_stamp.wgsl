// Stamps each intersection record slot with its pre-sort index
// in the record's pad field (offset 3).  Run BEFORE weld_sort_step.
//
// Covers the full padded range so zeroed (unoccupied) slots also
// carry a valid stamp for the sort.

struct Params {
    ixn_base: u32,
    ixn_count_padded: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> arena: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.ixn_count_padded) { return; }
    arena[params.ixn_base + tid * 8u + 3u] = tid;
}
