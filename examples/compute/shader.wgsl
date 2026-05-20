@group(0) @binding(0)
var<storage, read_write> values: array<u32, 4>;

fn collatz_iterations(n_base: u32) -> u32 {
    var n = n_base;
    var i = 0u;
    loop {
        if (n <= 1u) {
            break;
        }
        if ((n % 2u) == 0u) {
            n = n / 2u;
        } else {
            if (n >= 1431655765u) {
                return 4294967295u;
            }
            n = 3u * n + 1u;
        }
        i = i + 1u;
    }
    return i;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    values[global_id.x] = collatz_iterations(values[global_id.x]);
}
