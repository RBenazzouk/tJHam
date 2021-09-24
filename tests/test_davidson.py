import unittest
import numpy as np
import math as mp
import davidson as dav

class TestDavidsonMethods(unittest.TestCase):
    def test_lowests_eigs(self):
        eigvals = [10, 3, 0, 0, 5, 7, 9, 1, 2, 7, -5, -1]
        eigvects = np.eye(len(eigvals))
        val_sorted, vect_sorted = dav.get_lowests_eig(eigvals, eigvects, 11)
        
        expected_val_sorted = [-5, -1, 0, 0, 1, 2, 3, 5, 7, 7, 9]
        expected_vect_sorted = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], \
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], \
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], \
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        
        self.assertListEqual(val_sorted, expected_val_sorted)
        self.assertListEqual(vect_sorted.tolist(), expected_vect_sorted)

    def test_modified_davidson_algorithm(self):
        true_eigvals = [np.random.randint(100) for _ in range(300)]
        true_lowests_eigvals,_ = dav.get_lowests_eig(true_eigvals, np.zeros((len(true_eigvals), len(true_eigvals))), 15)
        D = np.diag(true_eigvals)
        V = dav.perform_modified_gram_schmidt(np.random.random((D.shape)))
        M = np.matmul(V.T, np.matmul(D, V))

        eigenstates = dav.modified_davidson_algorithm(M, 15, initial_guess=np.random.random((D.shape[0], 15)), stp=1000, tol=1e-5, dim_max=20, verbose=False)
        self.assertListEqual([round(eig, 6) for eig in eigenstates.eigvals], true_lowests_eigvals)

    def test_modified_davidson_algorithm_opt(self):
        true_eigvals = [np.random.randint(100) for _ in range(30)]
        true_lowests_eigvals,_ = dav.get_lowests_eig(true_eigvals, np.zeros((len(true_eigvals), len(true_eigvals))), 15)
        D = np.diag(true_eigvals)
        V = dav.perform_modified_gram_schmidt(np.random.random((D.shape)))
        M = np.matmul(V.T, np.matmul(D, V))

        eigenstates = dav.modified_davidson_algorithm_opt(M, 15, 10)
        self.assertListEqual([round(eig, 6) for eig in eigenstates.eigvals], true_lowests_eigvals)

if __name__ == '__main__':
    unittest.main()