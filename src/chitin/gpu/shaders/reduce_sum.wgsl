// Tree reduction sum. Single workgroup, max 256 elements.
// Uses pairwise tree structure matching CPU reference.

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>; // [0] = sum
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = count

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = params.x;

    if (tid < n) {
        shared_data[tid] = input_data[tid];
    } else {
        shared_data[tid] = 0.0;
    }
    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride >= 1u; stride /= 2u) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output_data[0] = shared_data[0];
    }
}
