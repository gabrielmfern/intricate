#![allow(dead_code)]

/// Asserts two matrices are approximately equal using the **assert_approx_equal**
/// function in every single vector of both matrices.
///
/// # Panics
///
/// Panics if the length of both matrices are not euqal, or
/// the length of vectors being compared are not equal.
pub(crate) fn assert_approx_equal_matrix(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>, decimal_place: u32) -> () {
    assert_eq!(a.len(), b.len());
    for (arr1, arr2) in a.iter().zip(b) {
        assert_approx_equal(arr1, arr2, decimal_place);
    }
}

/// Asserts that two vectors are approximately equal comparing all of their numbers
/// up to a certain **decimal_place**.
///
/// # Panics
///
/// Panics if the length of both vectors are not equal.
pub(crate) fn assert_approx_equal(a: &Vec<f32>, b: &Vec<f32>, decimal_place: u32) -> () {
    assert_eq!(a.len(), b.len());

    let power_ten = &10.0_f32.powf(decimal_place as f32);
    let approximate_a: Vec<f32> = a
        .iter()
        .map(|x| (x * power_ten).floor() / power_ten)
        .collect();
    let approximate_b: Vec<f32> = b
        .iter()
        .map(|x| (x * power_ten).floor() / power_ten)
        .collect();

    assert_eq!(approximate_a, approximate_b);
}

/// Asserts if the vectors **a** and **b** are approximately equal
/// being at most **max_dist** of a difference.
///
/// # Panics
///
/// Panics if the length of both vectors are not equal.
pub(crate) fn assert_approx_equal_distance(a: &Vec<f32>, b: &Vec<f32>, max_dist: f32) -> () {
    assert_eq!(a.len(), b.len());

    a.iter().zip(b).for_each(|(x, y)| {
        println!("x:{}\ny:{}", x, y);
        assert!((x - y).abs() <= max_dist);
    });
}