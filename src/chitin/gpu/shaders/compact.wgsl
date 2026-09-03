// Stream compaction. Keeps elements where mask != 0.
// Uses exclusive scan of mask for scatter addresses.

@group(0) @binding(0) var<storage, read> values: array<i32>;
@group(0) @binding(1) var<storage, read> mask: array<i32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<i32>;
@group(0) @binding(3) var<storage, read_write> output_count: array<u32>; // [0] = count
@group(0) @binding(4) var<uniform> params: vec4<u32>; // x = count

var<workgroup> scan: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = params.x;

    // Load mask into shared
    if (tid < n && mask[tid] != 0) {
        scan[tid] = 1u;
    } else {
        scan[tid] = 0u;
    }
    workgroupBarrier();

    // Blelloch exclusive scan on scan[]
    // Up-sweep
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        let idx = (tid + 1u) * stride * 2u - 1u;
        if (idx < 256u) {
            scan[idx] += scan[idx - stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        scan[255u] = 0u;
    }
    workgroupBarrier();

    // Down-sweep
    for (var stride = 128u; stride >= 1u; stride /= 2u) {
        let idx = (tid + 1u) * stride * 2u - 1u;
        if (idx < 256u) {
            let temp = scan[idx - stride];
            scan[idx - stride] = scan[idx];
            scan[idx] += temp;
        }
        workgroupBarrier();
    }

    // Scatter
    if (tid < n && mask[tid] != 0) {
        output_values[scan[tid]] = values[tid];
    }

    // Write count (total from last scan element + last mask)
    if (tid == 0u) {
        var count = scan[n - 1u];
        if (mask[n - 1u] != 0) {
            count += 1u;
        }
        output_count[0] = count;
    }
}
