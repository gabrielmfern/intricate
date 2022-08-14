//! Just a module with a few utilities that make writing code easier through out Intricate

pub mod approx_eq;

pub mod opencl;
pub use opencl::{
    OpenCLSummable,
    setup_opencl,
    OpenCLState
};

/// Finds the gratest common divisor (gcd) of two numbers **n** and **m** independent of their
/// ordering.
///
/// Worst case scenaryio for time complexity in this function is going to be
/// O(sqrt(max(n,m)) / 2).
pub fn gcd(n: usize, m: usize) -> usize {
    let mut largest_divisor: usize = 1;
    let max_point: usize = n.max(m);
    let middle: usize = max_point / 2;

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

