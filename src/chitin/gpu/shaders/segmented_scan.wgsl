// Segmented exclusive prefix sum. Each segment restarts accumulator.
// segment_ids must be non-decreasing.

@group(0) @binding(0) var<storage, read> values: array<i32>;
@group(0) @binding(1) var<storage, read> segment_ids: array<i32>;
@group(0) @binding(2) var<storage, read_write> output_data: array<i32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = count

// Sequential kernel — one thread does the work (matching CPU reference exactly).
// GPU parallelism comes from running many clips concurrently, not from parallelizing
// one small scan.
@compute @workgroup_size(1)
fn main() {
    let n = params.x;
    var acc: i32 = 0;
    var current_seg: i32 = segment_ids[0];

    for (var i = 0u; i < n; i++) {
        if (segment_ids[i] != current_seg) {
            acc = 0;
            current_seg = segment_ids[i];
        }
        output_data[i] = acc;
        acc += values[i];
    }
}
