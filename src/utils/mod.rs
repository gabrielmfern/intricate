pub mod matrix_operations;
pub mod vector_operations;
pub mod approx_eq;

pub fn gcd(n: usize, m: usize) -> usize {
    let mut largest_divisor: usize = 1;
    let middle_point: usize = (n.max(m) as f32).sqrt().floor() as usize;

    for k in (0..middle_point).rev() {
        if n % k == 0 && m % k == 0 {
            largest_divisor = k;
            break;
        }
    }

    largest_divisor
}