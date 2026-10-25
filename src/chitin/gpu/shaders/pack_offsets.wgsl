// Packs 4 separate i32 scan-offset buffers into one interleaved buffer.
// Output layout: [pos_0..pos_{n-1}, neg_0..neg_{n-1}, ixn_0..ixn_{n-1}, bnd_0..bnd_{n-1}]

struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> src_pos: array<i32>;
@group(0) @binding(1) var<storage, read> src_neg: array<i32>;
@group(0) @binding(2) var<storage, read> src_ixn: array<i32>;
@group(0) @binding(3) var<storage, read> src_bnd: array<i32>;
@group(0) @binding(4) var<storage, read_write> dst: array<i32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let n = params.n;
    if (tid >= n) { return; }
    dst[tid] = src_pos[tid];
    dst[n + tid] = src_neg[tid];
    dst[2u * n + tid] = src_ixn[tid];
    dst[3u * n + tid] = src_bnd[tid];
}
