// GPU port of `_to_grid_frame` + `QuantizationPolicy.normalize_to_grid` (CPU classify
// kernel). Computes centroid, scale factor, and quantized plane for one (mesh, plane)
// pair; output feeds classify.wgsl's grid-quantized dot product.
//
// Single-pair dispatch (1,1,1): mesh_headers[0] and planes[0] supply the pair,
// norm_params[0] is the output slot. Multi-pair indexing via workgroup_id is not
// implemented, since only one pair is dispatched today.

struct NormParams {
    centroid_x: f32,
    centroid_y: f32,
    centroid_z: f32,
    scale_factor: f32,
    grid_plane_x: i32,
    grid_plane_y: i32,
    grid_plane_z: i32,
    ambiguity_bound: f32,
    grid_normal_x: f32,
    grid_normal_y: f32,
    grid_normal_z: f32,
    grid_bits: u32,
    vertex_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct ClipPlane {
    point_x: f32,
    point_y: f32,
    point_z: f32,
    _pad0: u32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    _pad1: u32,
}

struct MeshHeader {
    vertex_offset: u32,
    vertex_count: u32,
    face_offset: u32,
    face_count: u32,
}

struct Point {
    x: f32,
    y: f32,
    z: f32,
    _pad0: f32,
}

@group(0) @binding(0) var<storage, read> vertices: array<Point>;
@group(0) @binding(1) var<storage, read> mesh_headers: array<MeshHeader>;
@group(0) @binding(2) var<storage, read> planes: array<ClipPlane>;
@group(0) @binding(3) var<storage, read_write> norm_params: array<NormParams>;
@group(0) @binding(4) var<uniform> params: vec4<u32>; // x = grid_bits

var<workgroup> shared_x: array<f32, 256>;
var<workgroup> shared_y: array<f32, 256>;
var<workgroup> shared_z: array<f32, 256>;

// Centroid broadcast: phase 2 needs it on every thread, not just thread 0.
var<workgroup> centroid_x: f32;
var<workgroup> centroid_y: f32;
var<workgroup> centroid_z: f32;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let header = mesh_headers[0];
    let plane = planes[0];
    let vertex_count = header.vertex_count;

    // Combined point set is V mesh vertices + 1 plane point; the plane point sits
    // at combined index vertex_count, not vertex_count - 1. Grid-stride the loop
    // so it covers both the single-point-per-thread case (vertex_count < 256) and
    // the multi-point case (vertex_count >= 256) with one code path.
    var sum_x: f32 = 0.0;
    var sum_y: f32 = 0.0;
    var sum_z: f32 = 0.0;
    for (var j = tid; j <= vertex_count; j += 256u) {
        if (j < vertex_count) {
            let p = vertices[header.vertex_offset + j];
            sum_x += p.x;
            sum_y += p.y;
            sum_z += p.z;
        } else {
            sum_x += plane.point_x;
            sum_y += plane.point_y;
            sum_z += plane.point_z;
        }
    }
    shared_x[tid] = sum_x;
    shared_y[tid] = sum_y;
    shared_z[tid] = sum_z;
    workgroupBarrier();

    for (var stride = 128u; stride >= 1u; stride /= 2u) {
        if (tid < stride) {
            shared_x[tid] += shared_x[tid + stride];
            shared_y[tid] += shared_y[tid + stride];
            shared_z[tid] += shared_z[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        // +1 for the plane point folded into the same centroid.
        let denom = f32(vertex_count + 1u);
        centroid_x = shared_x[0] / denom;
        centroid_y = shared_y[0] / denom;
        centroid_z = shared_z[0] / denom;
    }
    workgroupBarrier();

    var local_max: f32 = 0.0;
    for (var j = tid; j <= vertex_count; j += 256u) {
        var px: f32;
        var py: f32;
        var pz: f32;
        if (j < vertex_count) {
            let p = vertices[header.vertex_offset + j];
            px = p.x;
            py = p.y;
            pz = p.z;
        } else {
            px = plane.point_x;
            py = plane.point_y;
            pz = plane.point_z;
        }
        let d = max(max(abs(px - centroid_x), abs(py - centroid_y)), abs(pz - centroid_z));
        local_max = max(local_max, d);
    }
    // shared_x's phase-1 sum is fully consumed by now; reuse it for the max reduction.
    shared_x[tid] = local_max;
    workgroupBarrier();

    for (var stride = 128u; stride >= 1u; stride /= 2u) {
        if (tid < stride) {
            shared_x[tid] = max(shared_x[tid], shared_x[tid + stride]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        // Floor guards a degenerate zero-extent mesh/plane from a div-by-zero scale.
        let extent = max(shared_x[0], 1e-30);
        let scale_factor = 1.0 / (2.0 * extent);

        let unit_plane_x = (plane.point_x - centroid_x) * scale_factor;
        let unit_plane_y = (plane.point_y - centroid_y) * scale_factor;
        let unit_plane_z = (plane.point_z - centroid_z) * scale_factor;

        // WGSL `<<` only operates on u32; shift there, then convert to f32.
        let grid_scale = f32(1u << params.x);

        // round() is round-half-to-even, matching np.rint / f32::round_ties_even.
        let grid_plane_x = i32(round(unit_plane_x * grid_scale));
        let grid_plane_y = i32(round(unit_plane_y * grid_scale));
        let grid_plane_z = i32(round(unit_plane_z * grid_scale));

        let grid_normal_x = plane.normal_x * scale_factor;
        let grid_normal_y = plane.normal_y * scale_factor;
        let grid_normal_z = plane.normal_z * scale_factor;

        let ambiguity_bound = abs(grid_normal_x) + abs(grid_normal_y) + abs(grid_normal_z);

        var out: NormParams;
        out.centroid_x = centroid_x;
        out.centroid_y = centroid_y;
        out.centroid_z = centroid_z;
        out.scale_factor = scale_factor;
        out.grid_plane_x = grid_plane_x;
        out.grid_plane_y = grid_plane_y;
        out.grid_plane_z = grid_plane_z;
        out.ambiguity_bound = ambiguity_bound;
        out.grid_normal_x = grid_normal_x;
        out.grid_normal_y = grid_normal_y;
        out.grid_normal_z = grid_normal_z;
        out.grid_bits = params.x;
        out.vertex_count = vertex_count;
        out._pad0 = 0u;
        out._pad1 = 0u;
        out._pad2 = 0u;
        norm_params[0] = out;
    }
}
