// Exclusive prefix sum (Blelloch), pass 3 of 3. Adds each tile's scanned block offset
// (produced by scanning block_sums, itself a small prefix_sum_local.wgsl dispatch) to
// every element of that tile.

@group(0) @binding(0) var<storage, read_write> data: array<i32>;
@group(0) @binding(1) var<storage, read> block_offsets: array<i32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = total element count

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = params.x;
    let global_idx = wid.x * 256u + tid;

    if (global_idx < n) {
        data[global_idx] += block_offsets[wid.x];
    }
}
