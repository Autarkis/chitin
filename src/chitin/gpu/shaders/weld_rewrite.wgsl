// Rewrites face and boundary-edge indices from pre-weld intersection
// IDs to welded IDs, using the weld_map built by weld_finalize.
//
// Dispatched with enough threads for all positive faces + negative
// faces + boundary edges (worst-case counts are fine; unused arena
// slots contain zeros, which remap to zero — a benign no-op).

struct Params {
    vertex_count: u32,
    pos_face_base: u32,
    pos_face_count: u32,
    neg_face_base: u32,
    neg_face_count: u32,
    bnd_base: u32,
    bnd_count: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> arena: array<u32>;
@group(0) @binding(1) var<storage, read> weld_map: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn remap(v: u32) -> u32 {
    if (v < params.vertex_count) { return v; }
    let pre_weld_idx = v - params.vertex_count;
    return params.vertex_count + weld_map[pre_weld_idx];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let total_faces = params.pos_face_count + params.neg_face_count;

    if (tid < total_faces) {
        var base: u32;
        if (tid < params.pos_face_count) {
            base = params.pos_face_base + tid * 4u;
        } else {
            base = params.neg_face_base + (tid - params.pos_face_count) * 4u;
        }
        arena[base] = remap(arena[base]);
        arena[base + 1u] = remap(arena[base + 1u]);
        arena[base + 2u] = remap(arena[base + 2u]);
        return;
    }

    let bnd_tid = tid - total_faces;
    if (bnd_tid < params.bnd_count) {
        let base = params.bnd_base + bnd_tid * 2u;
        arena[base] = remap(arena[base]);
        arena[base + 1u] = remap(arena[base + 1u]);
    }
}
