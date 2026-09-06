// Per-triangle clipping count kernel. Reads vertex signs from the classify
// stage, determines output geometry counts per triangle for the scan stage.

struct Triangle {
    i0: u32,
    i1: u32,
    i2: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read> faces: array<Triangle>;
@group(0) @binding(1) var<storage, read> signs: array<i32>;
@group(0) @binding(2) var<storage, read_write> pos_counts: array<i32>;
@group(0) @binding(3) var<storage, read_write> neg_counts: array<i32>;
@group(0) @binding(4) var<storage, read_write> ixn_counts: array<i32>;
@group(0) @binding(5) var<storage, read_write> bnd_counts: array<i32>;
@group(0) @binding(6) var<uniform> params: vec4<u32>; // x = face_count

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = wgid.x * 256u + lid.x;
    let face_count = params.x;
    if (idx >= face_count) {
        return;
    }

    let tri = faces[idx];
    let s0 = signs[tri.i0];
    let s1 = signs[tri.i1];
    let s2 = signs[tri.i2];

    let cut_e = u32(s0 * s1 < 0) + u32(s1 * s2 < 0) + u32(s2 * s0 < 0);
    let num_pos = u32(s0 > 0) + u32(s1 > 0) + u32(s2 > 0);
    let has_pos = s0 > 0 || s1 > 0 || s2 > 0;
    let has_neg = s0 < 0 || s1 < 0 || s2 < 0;

    var pos: i32 = 0;
    var neg: i32 = 0;
    var ixn: i32 = 0;
    var bnd: i32 = 0;

    if (cut_e == 0u) {
        // Entirely on one side (coplanar 0,0,0 counts as both).
        if (!has_neg) {
            pos = 1;
        }
        if (!has_pos) {
            neg = 1;
        }
        // On-plane edges are boundary edges (lie on the clipping plane).
        // Guard against degenerate (zero-area) triangles with a repeated
        // vertex index -- a self-pair is not a real edge.
        let on01 = u32(s0 == 0 && s1 == 0 && tri.i0 != tri.i1);
        let on12 = u32(s1 == 0 && s2 == 0 && tri.i1 != tri.i2);
        let on20 = u32(s2 == 0 && s0 == 0 && tri.i2 != tri.i0);
        bnd = i32(on01 + on12 + on20);
    } else if (cut_e == 2u) {
        // Two vertices on one side, one on the other.
        if (num_pos >= 2u) {
            pos = 2;
            neg = 1;
        } else {
            pos = 1;
            neg = 2;
        }
        ixn = 2;
        bnd = 1;
    } else {
        // cut_e == 1: one vertex lies exactly on the plane.
        pos = 1;
        neg = 1;
        ixn = 1;
        bnd = 1;
    }

    pos_counts[idx] = pos;
    neg_counts[idx] = neg;
    ixn_counts[idx] = ixn;
    bnd_counts[idx] = bnd;
}
