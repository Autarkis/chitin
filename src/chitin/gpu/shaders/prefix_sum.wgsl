// Exclusive prefix sum (Blelloch). Single workgroup, max 256 elements.
// Input: array<i32>, Output: array<i32> (exclusive scan), element [n] = total

@group(0) @binding(0) var<storage, read> input_data: array<i32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<i32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = count

var<workgroup> shared_data: array<i32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = params.x;

    // Load
    if (tid < n) {
        shared_data[tid] = input_data[tid];
    } else {
        shared_data[tid] = 0;
    }
    workgroupBarrier();

    // Up-sweep (reduce)
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        let idx = (tid + 1u) * stride * 2u - 1u;
        if (idx < 256u) {
            shared_data[idx] += shared_data[idx - stride];
        }
        workgroupBarrier();
    }

    // Store total before clearing
    if (tid == 0u) {
        output_data[n] = shared_data[255u]; // total in last output slot
        shared_data[255u] = 0;
    }
    workgroupBarrier();

    // Down-sweep
    for (var stride = 128u; stride >= 1u; stride /= 2u) {
        let idx = (tid + 1u) * stride * 2u - 1u;
        if (idx < 256u) {
            let temp = shared_data[idx - stride];
            shared_data[idx - stride] = shared_data[idx];
            shared_data[idx] += temp;
        }
        workgroupBarrier();
    }

    // Write result
    if (tid < n) {
        output_data[tid] = shared_data[tid];
    }
}
