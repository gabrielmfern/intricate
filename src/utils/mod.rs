//! Just a module with a few utilities that make writing code easier through out Intricate

pub(crate) mod approx_eq;

pub use savefile;
pub use opencl3;

pub mod opencl;
pub use opencl::{
    BufferOperations,
    setup_opencl,
    OpenCLState
};

pub(crate) fn find_divsor_of_n_closest_to_m(n: usize, m: usize) -> usize {
    let middle = (n as f32).sqrt().floor() as usize;
    let (mut closest_up_to_now, mut distance) = (1, m - 1);
    for k in 2..=middle.min(m) {
        if n % k == 0 {
            let inverse = n / k;

            let k_inverse_distance = (inverse as i32 - m as i32).abs() as usize;
            if k_inverse_distance < distance && inverse < m {
                closest_up_to_now = inverse;
                distance = k_inverse_distance;
            }

            let k_distance = (k as i32 - m as i32).abs() as usize;
            if k_distance < distance {
                closest_up_to_now = k;
                distance = k_distance;
            }
        }
    }
    closest_up_to_now
}

// commented while it has no use
// pub(crate) fn find_multiple_of_n_closest_to_m(n: usize, m: usize) -> usize {
//     let floor_multiple = (m as f32 / n as f32).floor() * n as f32;
//     let ceil_multiple = (m as f32 / n as f32).ceil() * n as f32;

//     if (floor_multiple - m as f32).abs() < (ceil_multiple - m as f32).abs() {
//         floor_multiple as usize
//     } else {
//         ceil_multiple as usize
//     }
// }

/// Finds the gratest common divisor (gcd) of two numbers **n** and **m** independent of their
/// ordering.
///
/// Worst case scenaryio for time complexity in this function is going to be
/// O(sqrt(max(n,m)) / 2).
pub fn gcd(n: usize, m: usize) -> usize {
    let mut largest_divisor: usize = 1;
    let max_point: usize = n.max(m);
    let middle: usize = (max_point as f32).sqrt() as usize;

    let mut common_divisors_on_other_half: Vec<usize> = Vec::with_capacity(middle);

    for k in ((middle).max(1)..=max_point).rev() {
        if n % k == 0 && m % k == 0 {
            largest_divisor = k;
            break;
        }

        if k < max_point {
            let inverse_k = max_point - k;
            if n % inverse_k == 0 && m % inverse_k == 0 {
                common_divisors_on_other_half.push(inverse_k);
            }
        }
    }

    // If there was no change to the largest divisor
    if largest_divisor == 1 && middle > 1 {
        // This is always an Ok() because the other half will always exist
        // if there were no divisors found
        largest_divisor = *common_divisors_on_other_half.last().unwrap();
    }

    largest_divisor
}

#[test]
fn gcd_should_compute_greatest_common_divisor() {
    let n = 115;
    let m = 35;
    let expected_result = 5;
    assert_eq!(gcd(n, m), expected_result);
}

#[test]
fn gcd_should_work_small_numbers() {
    let n = 5;
    let m = 2;
    let expected_result = 1;
    assert_eq!(gcd(n, m), expected_result);
}

#[test]
fn gcd_should_work_with_very_small_numbers() {
    let n = 2;
    let m = 1;
    let expected_result = 1;
    assert_eq!(gcd(n, m), expected_result);
    let n = 3;
    let m = 2;
    let expected_result = 1;
    assert_eq!(gcd(n, m), expected_result);
}