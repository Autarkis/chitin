// Exclusive prefix sum (Blelloch), pass 1 of 3. One 256-element tile per workgroup.
// Input: array<i32>, Output: array<i32> (exclusive scan per tile), block_sums[wid] = tile total

@group(0) @binding(0) var<storage, read> input_data: array<i32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<i32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<i32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = total element count

var<workgroup> shared_data: array<i32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = params.x;
    let base = wid.x * 256u;
    let global_idx = base + tid;

    if (global_idx < n) {
        shared_data[tid] = input_data[global_idx];
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

    // Store tile total before clearing for down-sweep
    if (tid == 0u) {
        block_sums[wid.x] = shared_data[255u];
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
    if (global_idx < n) {
        output_data[global_idx] = shared_data[tid];
    }
}
