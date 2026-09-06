// Policy 0.3 vertex classification against a clipping plane.
// Grid-quantized dot product decides sign; world-frame f32 dot is the
// tie-break only when the grid dot falls inside the ambiguity bound
// (near-coplanar vertices, where quantization noise could flip the sign).

struct Point {
    x: f32,
    y: f32,
    z: f32,
    _pad0: f32,
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

@group(0) @binding(0) var<storage, read> vertices: array<Point>;
@group(0) @binding(1) var<storage, read> planes: array<ClipPlane>;
@group(0) @binding(2) var<storage, read> norm: array<NormParams>;
@group(0) @binding(3) var<storage, read_write> signs: array<i32>;
@group(0) @binding(4) var<storage, read_write> world_dots: array<f32>;
@group(0) @binding(5) var<uniform> params: vec4<u32>; // x = vertex_count, y = pair_index

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = wgid.x * 256u + lid.x;
    let vertex_count = params.x;
    if (idx >= vertex_count) {
        return;
    }

    let pair = params.y;
    let np = norm[pair];
    let plane = planes[pair];
    let v = vertices[idx];

    let centered_x = v.x - np.centroid_x;
    let centered_y = v.y - np.centroid_y;
    let centered_z = v.z - np.centroid_z;

    let unit_x = centered_x * np.scale_factor;
    let unit_y = centered_y * np.scale_factor;
    let unit_z = centered_z * np.scale_factor;

    let grid_scale = f32(1u << np.grid_bits);

    let grid_x = i32(round(unit_x * grid_scale));
    let grid_y = i32(round(unit_y * grid_scale));
    let grid_z = i32(round(unit_z * grid_scale));

    let dot = f32(grid_x - np.grid_plane_x) * np.grid_normal_x
        + f32(grid_y - np.grid_plane_y) * np.grid_normal_y
        + f32(grid_z - np.grid_plane_z) * np.grid_normal_z;

    // World dot is needed by every vertex downstream (emit-stage intersection
    // t-parameter), not only the ambiguous ones, so compute it unconditionally.
    let world_dot = (v.x - plane.point_x) * plane.normal_x
        + (v.y - plane.point_y) * plane.normal_y
        + (v.z - plane.point_z) * plane.normal_z;

    var sign: i32;
    if (abs(dot) > np.ambiguity_bound) {
        sign = select(-1, 1, dot > 0.0);
    } else {
        sign = select(select(-1, 1, world_dot > 0.0), 0, world_dot == 0.0);
    }

    signs[idx] = sign;
    world_dots[idx] = world_dot;
}
