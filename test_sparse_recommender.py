#test_sparse_recommender
import pytest
from sparse_recommender import SparseMatrix

# Initialize the SparseMatrix class.
def test_initialization():
    matrix = SparseMatrix()
    assert isinstance(matrix, SparseMatrix)

# Test the set and get methods.
def test_set_and_get():
    matrix = SparseMatrix()
    matrix.set(0, 0, 4)
    assert matrix.get(0, 0) == 4

# Test recommendations using a sample user vector.
def test_recommendations():
    matrix = SparseMatrix()
    matrix.set(0, 0, 4)
    matrix.set(1, 1, 2)
    user_vector = [1, 0]
    recommendations = matrix.recommend(user_vector)
    assert recommendations == [4, 0]

# Test adding another sparse matrix.
def test_add_movie():
    matrix1 = SparseMatrix()
    matrix1.set(0, 0, 4)
    matrix2 = SparseMatrix()
    matrix2.set(1, 1, 2)
    result = matrix1.add_movie(matrix2)
    assert result.get(0, 0) == 4
    assert result.get(1, 1) == 2

# Test converting to a dense matrix.
def test_to_dense():
    matrix = SparseMatrix()
    matrix.set(0, 0, 4)
    matrix.set(1, 1, 2)
    dense_matrix = matrix.to_dense()
    assert dense_matrix == [[4, 0], [0, 2]]

#error handling cases

def test_set_negative_indices():
    matrix = SparseMatrix()
    with pytest.raises(IndexError):
        matrix.set(-1, 0, 2)

def test_get_negative_indices():
    matrix = SparseMatrix()
    with pytest.raises(IndexError):
        matrix.get(0, -1)

def test_to_dense_missing_rows_columns():
    matrix = SparseMatrix()
    matrix.set(2, 3, 2)
    dense_matrix = matrix.to_dense()
    assert dense_matrix == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]]
    
def test_recommend_empty_matrix():
    matrix = SparseMatrix()
    user_vector = [1, 0]
    recommendations = matrix.recommend(user_vector)
    assert recommendations == []


# Test adding two matrices with no overlapping values.
def test_add_matrices_no_overlap():
    matrix1 = SparseMatrix()
    matrix1.set(0, 0, 3)
    matrix1.set(1, 1, 4)

    matrix2 = SparseMatrix()
    matrix2.set(0, 1, 2)
    matrix2.set(1, 0, 5)

    result = matrix1.add_movie(matrix2)
    assert result.get(0, 0) == 3
    assert result.get(0, 1) == 2
    assert result.get(1, 0) == 5
    assert result.get(1, 1) == 4

# Test adding an empty matrix to a non-empty matrix.
def test_add_empty_matrix_to_non_empty():
    matrix1 = SparseMatrix()
    matrix1.set(0, 0, 3)

    matrix2 = SparseMatrix()

    result = matrix1.add_movie(matrix2)
    assert result.get(0, 0) == 3




if __name__ == "__main__":
    pytest.main()
