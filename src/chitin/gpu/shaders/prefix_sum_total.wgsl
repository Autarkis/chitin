// Exclusive prefix sum, grand-total extraction. Copies the scan's total (the
// element one past the scanned range, written by prefix_sum.wgsl) into a
// dedicated 4-byte buffer so callers never need a CPU readback to find it.

@group(0) @binding(0) var<storage, read> scanned: array<i32>;
@group(0) @binding(1) var<storage, read_write> total: array<i32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = number of scanned entries

@compute @workgroup_size(1)
fn main() {
    total[0] = scanned[params.x];
}
