#sparse_recommender
class SparseMatrix:
    def __init__(self):
        # Initializing empty dictionary to represent the sparse matrix
        self.matrix = {}

    def set(self, row, col, value):
        if row < 0 or col < 0:
            raise IndexError("Row and column indices must be non-negative")
         # If the row not exist in the matrix, create it as a dictionary.
        if row not in self.matrix:
            self.matrix[row] = {}
         # Set the value at the specified row and column
        self.matrix[row][col] = value
    
    
    def get(self, row, col):
        if row in self.matrix and col in self.matrix[row]:
            return self.matrix[row][col]
        else:
            raise IndexError("Row and column indices must be non-negative")
        return 0

    def recommend(self, vector):
        recommendations = []
        for row in self.matrix:
            dot_product = sum(self.matrix[row].get(col, 0) * vector[col] for col in range(len(vector)))
            recommendations.append(dot_product)
        return recommendations

    def add_movie(self, matrix):
        result = SparseMatrix()
        for row in self.matrix:
            for col in self.matrix[row]:
                result.set(row, col, self.matrix[row][col])
        for row in matrix.matrix:
            for col in matrix.matrix[row]:
                result.set(row, col, matrix.matrix[row][col])
        return result
    #Convert the sparse matrix to a dense matrix.
    def to_dense(self):
        max_row = max(self.matrix.keys())
        max_col = max(max(self.matrix[row].keys()) for row in self.matrix)
        dense_matrix = [[0] * (max_col + 1) for _ in range(max_row + 1)]
        for row in self.matrix:
            for col in self.matrix[row]:
                dense_matrix[row][col] = self.matrix[row][col]
        return dense_matrix
