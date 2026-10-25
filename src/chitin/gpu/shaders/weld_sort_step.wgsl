// One compare-and-swap step of a bitonic sort over intersection records.
// Records are 8 u32 each in the arena.  Sort key is (v_lo, v_hi) at
// record offsets 4 and 5, compared lexicographically.
//
// The Python driver calls this O(log^2 N) times with varying (k, j).

struct Params {
    ixn_base: u32,
    ixn_count_padded: u32,
    k: u32,
    j: u32,
}

@group(0) @binding(0) var<storage, read_write> arena: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let n = params.ixn_count_padded;
    let k = params.k;
    let j = params.j;
    let base = params.ixn_base;

    let block_size = 2u * j;
    let block_idx = tid / j;
    let in_offset = tid % j;
    let l = block_idx * block_size + in_offset;
    let r = l + j;

    if (r >= n) { return; }

    let ascending = ((l & k) == 0u);

    let l_vlo = arena[base + l * 8u + 4u];
    let l_vhi = arena[base + l * 8u + 5u];
    let r_vlo = arena[base + r * 8u + 4u];
    let r_vhi = arena[base + r * 8u + 5u];

    let l_gt_r = (l_vlo > r_vlo) || (l_vlo == r_vlo && l_vhi > r_vhi);
    let l_lt_r = (l_vlo < r_vlo) || (l_vlo == r_vlo && l_vhi < r_vhi);

    let should_swap = select(l_lt_r, l_gt_r, ascending);
    if (should_swap) {
        for (var i = 0u; i < 8u; i = i + 1u) {
            let tl = arena[base + l * 8u + i];
            let tr = arena[base + r * 8u + i];
            arena[base + l * 8u + i] = tr;
            arena[base + r * 8u + i] = tl;
        }
    }
}
