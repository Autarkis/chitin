// Per-triangle clipping emit kernel. Reads scanned offsets and writes
// clipped geometry (positive/negative faces, intersections, boundary edges,
// ancestry) into a packed output arena.

struct Point {
    x: f32,
    y: f32,
    z: f32,
    _pad0: f32,
}

struct Triangle {
    i0: u32,
    i1: u32,
    i2: u32,
    _pad0: u32,
}

struct EmitParams {
    face_count: u32,
    vertex_count: u32,
    pos_face_base: u32,
    neg_face_base: u32,
    ixn_base: u32,
    bnd_base: u32,
    pos_ancestry_base: u32,
    neg_ancestry_base: u32,
}

@group(0) @binding(0) var<storage, read> vertices: array<Point>;
@group(0) @binding(1) var<storage, read> faces: array<Triangle>;
@group(0) @binding(2) var<storage, read> signs: array<i32>;
@group(0) @binding(3) var<storage, read> world_dots: array<f32>;
@group(0) @binding(4) var<storage, read> offsets: array<i32>;
@group(0) @binding(5) var<storage, read_write> out_arena: array<u32>;
@group(0) @binding(6) var<uniform> params: EmitParams;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let gidx = wgid.x * 256u + lid.x;
    let fc = params.face_count;
    let vc = params.vertex_count;
    if (gidx >= fc) {
        return;
    }

    let idx = gidx;
    let tri = faces[idx];
    let vi0 = tri.i0;
    let vi1 = tri.i1;
    let vi2 = tri.i2;
    let s0 = signs[vi0];
    let s1 = signs[vi1];
    let s2 = signs[vi2];

    let pos_off = offsets[idx];
    let neg_off = offsets[fc + idx];
    let ixn_off = offsets[2u * fc + idx];
    let bnd_off = offsets[3u * fc + idx];

    let cut01 = s0 * s1 < 0;
    let cut12 = s1 * s2 < 0;
    let cut20 = s2 * s0 < 0;

    if (!cut01 && !cut12 && !cut20) {
        // No cut edges: triangle lies entirely on one side (coplanar 0,0,0
        // lies on both and is written to both sides).
        let has_pos = s0 > 0 || s1 > 0 || s2 > 0;
        let has_neg = s0 < 0 || s1 < 0 || s2 < 0;

        if (!has_neg) {
            let base = params.pos_face_base + u32(pos_off) * 4u;
            out_arena[base] = vi0;
            out_arena[base + 1u] = vi1;
            out_arena[base + 2u] = vi2;
            out_arena[base + 3u] = 0u;
            out_arena[params.pos_ancestry_base + u32(pos_off)] = idx;
        }
        if (!has_pos) {
            let base = params.neg_face_base + u32(neg_off) * 4u;
            out_arena[base] = vi0;
            out_arena[base + 1u] = vi1;
            out_arena[base + 2u] = vi2;
            out_arena[base + 3u] = 0u;
            out_arena[params.neg_ancestry_base + u32(neg_off)] = idx;
        }

        // On-plane edges are boundary edges (lie on the clipping plane).
        // Guard against degenerate (zero-area) triangles with a repeated
        // vertex index -- a self-pair is not a real edge.
        var bw: u32 = 0u;
        if (s0 == 0 && s1 == 0 && vi0 != vi1) {
            let bbase = params.bnd_base + (u32(bnd_off) + bw) * 2u;
            out_arena[bbase] = min(vi0, vi1);
            out_arena[bbase + 1u] = max(vi0, vi1);
            bw++;
        }
        if (s1 == 0 && s2 == 0 && vi1 != vi2) {
            let bbase = params.bnd_base + (u32(bnd_off) + bw) * 2u;
            out_arena[bbase] = min(vi1, vi2);
            out_arena[bbase + 1u] = max(vi1, vi2);
            bw++;
        }
        if (s2 == 0 && s0 == 0 && vi2 != vi0) {
            let bbase = params.bnd_base + (u32(bnd_off) + bw) * 2u;
            out_arena[bbase] = min(vi2, vi0);
            out_arena[bbase + 1u] = max(vi2, vi0);
            bw++;
        }
        return;
    }

    // Mixed case: walk the 3 edges in order, building the positive and
    // negative polygons simultaneously and emitting an intersection record
    // for each cut edge.
    var pos_poly: array<u32, 4>;
    var neg_poly: array<u32, 4>;
    var pc: u32 = 0u;
    var nc: u32 = 0u;
    var iw: u32 = 0u;

    // Edge (vi0, vi1)
    {
        let vi = vi0;
        let vj = vi1;
        let si = s0;
        let sj = s1;
        if (si >= 0) {
            pos_poly[pc] = vi;
            pc++;
        }
        if (si <= 0) {
            neg_poly[nc] = vi;
            nc++;
        }
        if (si * sj < 0) {
            let da = world_dots[vi];
            let db = world_dots[vj];
            let t = da / (da - db);
            let va = vertices[vi];
            let vb = vertices[vj];
            let ix = va.x + t * (vb.x - va.x);
            let iy = va.y + t * (vb.y - va.y);
            let iz = va.z + t * (vb.z - va.z);

            let ixn_global = vc + u32(ixn_off) + iw;
            pos_poly[pc] = ixn_global;
            pc++;
            neg_poly[nc] = ixn_global;
            nc++;

            let ixn_base_u32 = params.ixn_base + (u32(ixn_off) + iw) * 8u;
            out_arena[ixn_base_u32] = bitcast<u32>(ix);
            out_arena[ixn_base_u32 + 1u] = bitcast<u32>(iy);
            out_arena[ixn_base_u32 + 2u] = bitcast<u32>(iz);
            out_arena[ixn_base_u32 + 3u] = 0u;
            out_arena[ixn_base_u32 + 4u] = min(vi, vj);
            out_arena[ixn_base_u32 + 5u] = max(vi, vj);
            out_arena[ixn_base_u32 + 6u] = 0u;
            out_arena[ixn_base_u32 + 7u] = 0u;

            iw++;
        }
    }

    // Edge (vi1, vi2)
    {
        let vi = vi1;
        let vj = vi2;
        let si = s1;
        let sj = s2;
        if (si >= 0) {
            pos_poly[pc] = vi;
            pc++;
        }
        if (si <= 0) {
            neg_poly[nc] = vi;
            nc++;
        }
        if (si * sj < 0) {
            let da = world_dots[vi];
            let db = world_dots[vj];
            let t = da / (da - db);
            let va = vertices[vi];
            let vb = vertices[vj];
            let ix = va.x + t * (vb.x - va.x);
            let iy = va.y + t * (vb.y - va.y);
            let iz = va.z + t * (vb.z - va.z);

            let ixn_global = vc + u32(ixn_off) + iw;
            pos_poly[pc] = ixn_global;
            pc++;
            neg_poly[nc] = ixn_global;
            nc++;

            let ixn_base_u32 = params.ixn_base + (u32(ixn_off) + iw) * 8u;
            out_arena[ixn_base_u32] = bitcast<u32>(ix);
            out_arena[ixn_base_u32 + 1u] = bitcast<u32>(iy);
            out_arena[ixn_base_u32 + 2u] = bitcast<u32>(iz);
            out_arena[ixn_base_u32 + 3u] = 0u;
            out_arena[ixn_base_u32 + 4u] = min(vi, vj);
            out_arena[ixn_base_u32 + 5u] = max(vi, vj);
            out_arena[ixn_base_u32 + 6u] = 0u;
            out_arena[ixn_base_u32 + 7u] = 0u;

            iw++;
        }
    }

    // Edge (vi2, vi0)
    {
        let vi = vi2;
        let vj = vi0;
        let si = s2;
        let sj = s0;
        if (si >= 0) {
            pos_poly[pc] = vi;
            pc++;
        }
        if (si <= 0) {
            neg_poly[nc] = vi;
            nc++;
        }
        if (si * sj < 0) {
            let da = world_dots[vi];
            let db = world_dots[vj];
            let t = da / (da - db);
            let va = vertices[vi];
            let vb = vertices[vj];
            let ix = va.x + t * (vb.x - va.x);
            let iy = va.y + t * (vb.y - va.y);
            let iz = va.z + t * (vb.z - va.z);

            let ixn_global = vc + u32(ixn_off) + iw;
            pos_poly[pc] = ixn_global;
            pc++;
            neg_poly[nc] = ixn_global;
            nc++;

            let ixn_base_u32 = params.ixn_base + (u32(ixn_off) + iw) * 8u;
            out_arena[ixn_base_u32] = bitcast<u32>(ix);
            out_arena[ixn_base_u32 + 1u] = bitcast<u32>(iy);
            out_arena[ixn_base_u32 + 2u] = bitcast<u32>(iz);
            out_arena[ixn_base_u32 + 3u] = 0u;
            out_arena[ixn_base_u32 + 4u] = min(vi, vj);
            out_arena[ixn_base_u32 + 5u] = max(vi, vj);
            out_arena[ixn_base_u32 + 6u] = 0u;
            out_arena[ixn_base_u32 + 7u] = 0u;

            iw++;
        }
    }

    // Fan-triangulate the positive polygon and write faces + ancestry.
    var pf: u32 = 0u;
    for (var i = 1u; i < pc - 1u; i++) {
        let base = params.pos_face_base + (u32(pos_off) + pf) * 4u;
        out_arena[base] = pos_poly[0];
        out_arena[base + 1u] = pos_poly[i];
        out_arena[base + 2u] = pos_poly[i + 1u];
        out_arena[base + 3u] = 0u;
        out_arena[params.pos_ancestry_base + u32(pos_off) + pf] = idx;
        pf++;
    }

    // Fan-triangulate the negative polygon and write faces + ancestry.
    var nf: u32 = 0u;
    for (var i = 1u; i < nc - 1u; i++) {
        let base = params.neg_face_base + (u32(neg_off) + nf) * 4u;
        out_arena[base] = neg_poly[0];
        out_arena[base + 1u] = neg_poly[i];
        out_arena[base + 2u] = neg_poly[i + 1u];
        out_arena[base + 3u] = 0u;
        out_arena[params.neg_ancestry_base + u32(neg_off) + nf] = idx;
        nf++;
    }

    // Boundary edge between the cut points (or between an on-plane vertex
    // and the single cut point).
    let num_cuts = u32(cut01) + u32(cut12) + u32(cut20);
    if (num_cuts == 2u) {
        let i0g = vc + u32(ixn_off);
        let i1g = vc + u32(ixn_off) + 1u;
        let bbase = params.bnd_base + u32(bnd_off) * 2u;
        out_arena[bbase] = min(i0g, i1g);
        out_arena[bbase + 1u] = max(i0g, i1g);
    } else if (num_cuts == 1u) {
        var on_plane_v: u32 = vi0;
        if (s1 == 0) {
            on_plane_v = vi1;
        } else if (s2 == 0) {
            on_plane_v = vi2;
        }
        let i0g = vc + u32(ixn_off);
        let bbase = params.bnd_base + u32(bnd_off) * 2u;
        out_arena[bbase] = min(on_plane_v, i0g);
        out_arena[bbase + 1u] = max(on_plane_v, i0g);
    }
}
