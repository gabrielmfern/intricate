use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

pub trait MatrixOperations {
    type Item;

    fn dot_product(&self, against: &Vec<Self::Item>) -> Vec<Self::Item>;

    fn add(&self, against: &Self) -> Self;

    fn subtract(&self, against: &Self) -> Self;

    fn multiply_by_other(&self, against: Self) -> Self;

    fn multiply(&self, against: Self::Item) -> Self;

    fn transpose(&self) -> Self;

    fn get_width(&self) -> usize;

    fn get_height(&self) -> usize;
}

impl MatrixOperations for Vec<Vec<f64>> {
    type Item = f64;
    fn dot_product(&self, against: &Vec<Self::Item>) -> Vec<Self::Item> {
        let width = self.get_width();
        let height = self.get_height();
        assert_eq!(height, against.len());

        let mut result = vec![0.0_f64; width];

        for col in 0..width {
            for row in 0..height {
                result[col] += against[row] * self[row][col];
            }
        }

        result
    }

    fn add(&self, against: &Self) -> Self {
        self.par_iter()
            .zip(against)
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + y).collect())
            .collect()
    }

    fn subtract(&self, against: &Self) -> Self {
        self.par_iter()
            .zip(against)
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x - y).collect())
            .collect()
    }

    fn multiply_by_other(&self, against: Self) -> Self {
        self.par_iter()
            .zip(against)
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x * y).collect())
            .collect()
    }

    fn multiply(&self, by: Self::Item) -> Self {
        self.par_iter()
            .map(|row| row.iter().map(|x| x * by).collect::<Vec<f64>>())
            .collect::<Vec<Vec<f64>>>()
    }

    fn transpose(&self) -> Self {
        let width = self.get_width();
        let height = self.get_height();
        let mut transposed: Vec<Vec<f64>> = vec![vec![0.0_f64; height]; width];

        for (i, row) in self.iter().enumerate() {
            assert_eq!(width, row.len());
            for (j, col) in row.iter().enumerate() {
                transposed[j][i] = *col;
            }
        }

        transposed
    }

    fn get_width(&self) -> usize {
        self[0].len()
    }

    fn get_height(&self) -> usize {
        self.len()
    }
}

#[test]
fn should_correctly_multiply_matrix_and_vector() {
    let matrix: Vec<Vec<f64>> = Vec::from([
        Vec::from([0.2, 0.4]),
        Vec::from([3.1, 9.2]),
        Vec::from([0.9, 4.4]),
    ]);
    let vector: Vec<f64> = Vec::from([0.5, 0.4, 0.3]);

    let expected_result: Vec<f64> = Vec::from([
        vector[0] * matrix[0][0] + vector[1] * matrix[1][0] + vector[2] * matrix[2][0],
        vector[0] * matrix[0][1] + vector[1] * matrix[1][1] + vector[2] * matrix[2][1]
    ]);

    let actual_result: Vec<f64> = matrix.dot_product(&vector);

    assert_eq!(actual_result, expected_result);
}

#[test]
fn should_transpose_matrix_correctly() {
    let matrix: Vec<Vec<f64>> = Vec::from([
        Vec::from([0.2, 0.4]),
        Vec::from([3.1, 9.2]),
        Vec::from([0.9, 4.4]),
    ]);
    let correctly_transposed_matrix: Vec<Vec<f64>> =
        Vec::from([Vec::from([0.2, 3.1, 0.9]), Vec::from([0.4, 9.2, 4.4])]);
    let transposed_matrix = matrix.transpose();

    assert_eq!(correctly_transposed_matrix, transposed_matrix);
}
