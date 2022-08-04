use super::vector_operations::VectorOperations;

/// same things as the other method but for matrices
pub fn assert_approx_equal_matrix(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>, decimal_place: u32) -> () {
    assert_eq!(a.len(), b.len());
    for (arr1, arr2) in a.iter().zip(b) {
        assert_approx_equal(arr1, arr2, decimal_place);
    }
}

/// just used for comparing floating point numbers in tests
/// so that you don't need to compare up to the last precision
/// and sometimes fail in tests that should have passed
pub fn assert_approx_equal(a: &Vec<f32>, b: &Vec<f32>, decimal_place: u32) -> () {
    assert_eq!(a.len(), b.len());
    let power_ten = &10.0_f32.powf(decimal_place as f32);
    let approximate_a = a.multiply_number(power_ten)
                         .floor()
                         .divide_number(power_ten);
    let approximate_b = b.multiply_number(power_ten)
                         .floor()
                         .divide_number(power_ten);
    
    assert_eq!(approximate_a, approximate_b);
}

pub fn assert_approx_equal_distance(a: &Vec<f32>, b: &Vec<f32>, max_dist: f32) -> () {
    assert_eq!(a.len(), b.len());

    a.iter().zip(b).for_each(|(x, y)| {
        println!("x:{}\ny:{}", x, y);
        assert!((x - y).abs() <= max_dist);
    });
}