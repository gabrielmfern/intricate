/// A trait implemented by Intricate to make it easier to do computations between
/// matrices, in a way that if we wanted to compute the prediction of a Dense layer using this
/// trait we can just do 
///
/// ```
/// weights.dot_product(inputs).add(biases)
/// ```
pub trait MatrixOperations
where 
    Self: Sized 
{
    type Item;

    fn dot_product(&self, against: &Vec<Self::Item>) -> Option< Vec<Self::Item>>;

    fn add(&self, against: &Self) -> Self;

    fn subtract(&self, against: &Self) -> Self;

    fn multiply_by_other(&self, against: &Self) -> Self;

    fn multiply(&self, against: Self::Item) -> Self;

    fn transpose(&self) -> Option<Self>;

    fn get_width(&self) -> Option<usize>;

    fn get_height(&self) -> usize;
}

impl MatrixOperations for Vec<Vec<f32>> {
    type Item = f32;
    fn dot_product(&self, against: &Vec<Self::Item>) -> Option<Vec<Self::Item>> {
        let width = self.get_width()?;
        let height = self.get_height();
        assert_eq!(height, against.len());

        let mut result: Vec<Self::Item> = vec![0.0; width];

        for col in 0..width {
            for row in 0..height {
                result[col] += against[row] * self[row][col];
            }
        }

        Some(result)
    }

    fn add(&self, against: &Self) -> Self {
        self.iter()
            .zip(against)
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + y).collect())
            .collect()
    }

    fn subtract(&self, against: &Self) -> Self {
        self.iter()
            .zip(against)
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x - y).collect())
            .collect()
    }

    fn multiply_by_other(&self, against: &Self) -> Self {
        self.iter()
            .zip(against)
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x * y).collect())
            .collect()
    }

    fn multiply(&self, by: Self::Item) -> Self {
        self.iter()
            .map(|row| row.iter().map(|x| x * by).collect())
            .collect()
    }

    fn transpose(&self) -> Option<Self> {
        let width = self.get_width()?;
        let height = self.get_height();
        let mut transposed: Self = vec![vec![0.0; height]; width];

        for (i, row) in self.iter().enumerate() {
            assert_eq!(width, row.len());
            for (j, col) in row.iter().enumerate() {
                transposed[j][i] = *col;
            }
        }

        Some(transposed)
    }

    fn get_width(&self) -> Option<usize> {
        if let Some(first_vec) = self.get(0) {
            Some(first_vec.len())
        } else {
            None
        }
    }

    fn get_height(&self) -> usize {
        self.len()
    }
}

#[test]
fn should_multiply_equally_sized_matrices_correctly() {
    let matrix1: Vec<Vec<f32>> = vec![
        vec![0.5, 0.3, 0.4],
        vec![15.3, 25.3, 99.1],
        vec![111.0, 123.5, 1.3],
    ];
    let matrix2: Vec<Vec<f32>> = vec![
        vec![123.2, 15.3, 93.4],
        vec![15.3, 25.3, 43.1],
        vec![12.2, 253.5, 1.3],
    ];
    let expected_result: Vec<Vec<f32>> = vec![
        vec![0.5 * 123.2, 0.3 * 15.3, 0.4 * 93.4],
        vec![15.3 * 15.3, 25.3 * 25.3, 99.1 * 43.1],
        vec![111.0 * 12.2, 123.5 * 253.5 , 1.3 * 1.3],
    ];
    assert_eq!(matrix1.multiply_by_other(&matrix2), expected_result);
}

#[test]
fn should_subtract_matrices_correctly() {
    let matrix1: Vec<Vec<f32>> = vec![
        vec![0.5, 0.3, 0.4],
        vec![15.3, 25.3, 99.1],
        vec![111.0, 123.5, 1.3],
    ];
    let matrix2: Vec<Vec<f32>> = vec![
        vec![123.2, 15.3, 93.4],
        vec![15.3, 25.3, 43.1],
        vec![12.2, 253.5, 1.3],
    ];
    let expected_result: Vec<Vec<f32>> = vec![
        vec![0.5 - 123.2, 0.3 - 15.3, 0.4 - 93.4],
        vec![15.3 - 15.3, 25.3 - 25.3, 99.1 - 43.1],
        vec![111.0 - 12.2, 123.5 - 253.5 , 1.3 - 1.3],
    ];
    assert_eq!(matrix1.subtract(&matrix2), expected_result);
}

#[test]
fn should_multiply_matrix_by_number_correctly() {
    let matrix1: Vec<Vec<f32>> = vec![
        vec![0.5, 0.3, 0.4],
        vec![15.3, 25.3, 99.1],
        vec![111.0, 123.5, 1.3],
    ];
    let num: f32 = 3.2;
    let expected_result: Vec<Vec<f32>> = vec![
        vec![0.5 * 3.2, 0.3 * 3.2, 0.4 * 3.2],
        vec![15.3 * 3.2, 25.3 * 3.2, 99.1 * 3.2],
        vec![111.0 * 3.2, 123.5 * 3.2, 1.3 * 3.2],
    ];
    assert_eq!(matrix1.multiply(num), expected_result);
}

#[test]
fn should_add_matrices_correctly() {
    let matrix1: Vec<Vec<f32>> = vec![
        vec![0.5, 0.3, 0.4],
        vec![15.3, 25.3, 99.1],
        vec![111.0, 123.5, 1.3],
    ];
    let matrix2: Vec<Vec<f32>> = vec![
        vec![123.2, 15.3, 93.4],
        vec![15.3, 25.3, 43.1],
        vec![12.2, 253.5, 1.3],
    ];
    let expected_result: Vec<Vec<f32>> = vec![
        vec![0.5 + 123.2, 15.3 + 0.3, 93.4 + 0.4],
        vec![15.3 + 15.3, 25.3 + 25.3, 43.1 + 99.1],
        vec![111.0 + 12.2, 253.5 + 123.5, 1.3 + 1.3],
    ];
    assert_eq!(matrix1.add(&matrix2), expected_result);
}

#[test]
fn should_correctly_take_dot_product() {
    let matrix: Vec<Vec<f32>> = Vec::from([
        Vec::from([0.2, 0.4]),
        Vec::from([3.1, 9.2]),
        Vec::from([0.9, 4.4]),
    ]);
    let vector: Vec<f32> = Vec::from([0.5, 0.4, 0.3]);

    let expected_result: Vec<f32> = Vec::from([
        vector[0] * matrix[0][0] + vector[1] * matrix[1][0] + vector[2] * matrix[2][0],
        vector[0] * matrix[0][1] + vector[1] * matrix[1][1] + vector[2] * matrix[2][1]
    ]);

    let actual_result: Vec<f32> = matrix.dot_product(&vector).unwrap();

    assert_eq!(actual_result, expected_result);
}

#[test]
fn should_transpose_matrix_correctly() {
    let matrix: Vec<Vec<f32>> = Vec::from([
        Vec::from([0.2, 0.4]),
        Vec::from([3.1, 9.2]),
        Vec::from([0.9, 4.4]),
    ]);
    let correctly_transposed_matrix: Vec<Vec<f32>> =
        Vec::from([Vec::from([0.2, 3.1, 0.9]), Vec::from([0.4, 9.2, 4.4])]);
    let transposed_matrix = matrix.transpose().unwrap();

    assert_eq!(correctly_transposed_matrix, transposed_matrix);
}