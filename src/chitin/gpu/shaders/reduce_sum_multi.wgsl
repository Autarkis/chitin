// Tree reduction sum, multi-workgroup first pass. Each workgroup reduces its
// own 256-element slice into partial_sums[wid.x]; a second pass over
// reduce_sum.wgsl reduces the partials to a final scalar.

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> partial_sums: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = total element count

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let n = params.x;
    let idx = wid.x * 256u + tid;

    if (idx < n) {
        shared_data[tid] = input_data[idx];
    } else {
        shared_data[tid] = 0.0;
    }
    workgroupBarrier();

    for (var stride = 128u; stride >= 1u; stride /= 2u) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        partial_sums[wid.x] = shared_data[0];
    }
}
