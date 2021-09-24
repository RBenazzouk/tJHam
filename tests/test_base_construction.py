import unittest
import numpy as np
import math as mp
import latticeFunctions as lF
import baseConstruction as bC

class TestBasisMethods(unittest.TestCase):
    def test_reorder_basis_element(self):
        bond_matrix_1888 = lF.build_rectangular_bond_matrix(2, 1, 3)
        configuration_1888 = bC.Configuration(bond_matrix_1888, 2, 8, [], 'periodic', 2, 1, 3)
        det = [5, 1, 6, 0, 4, 12, 17, 9, 2, 8, 15, 14, 3, 16, 13, 7, 10, 11]
        self.assertListEqual([1, 5, 0, 2, 4, 6, 8, 9, 12, 17, 3, 7, 10, 11, 13, 14, 15, 16], bC.reorder_basis_element(configuration_1888, det))

    def test_index_in_basis(self):
        bond_matrix_4 = lF.build_rectangular_bond_matrix(1, 1, 2)
        configuration_421 = bC.Configuration(bond_matrix_4, 1, 1, [], 'periodic', 1, 1, 2)
        basis_421 = bC.construct_basis(configuration_421)

        for ind_421, det_421 in enumerate(basis_421.determinants):
            self.assertEqual(ind_421, bC.get_index_in_basis(configuration_421, det_421))
        
        bond_matrix_9 = lF.build_rectangular_bond_matrix(1, 1, 3)
        configuration_944 = bC.Configuration(bond_matrix_9, 1, 4, [], 'periodic', 1, 1, 3)
        basis_944 = bC.construct_basis(configuration_944)

        for ind_944, det_944 in enumerate(basis_944.determinants):
            self.assertEqual(ind_944, bC.get_index_in_basis(configuration_944, det_944))

    def test_hash_basis(self):
        bond_matrix = lF.build_rectangular_bond_matrix(2, 1, 3)
        configuration = bC.Configuration(bond_matrix, 2, 8, [], 'periodic', 2, 1, 3)

        list_indices = [np.random.randint(configuration.nb_conf) for _ in range(20)]

        for ind in list_indices:
            self.assertEqual(ind, bC.get_index_in_basis(configuration, bC.get_hash_basis(2, 8, 8, ind)))

    def test_extend_determinant(self):
        bond_matrix_9 = lF.build_rectangular_bond_matrix(1, 1, 3)
        configuration_953 = bC.Configuration(bond_matrix_9, 1, 3, [], 'periodic', 1, 1, 3)
        configuration_935 = bC.Configuration(bond_matrix_9, 1, 5, [], 'periodic', 1, 1, 3)

        det_953 = [4, 0, 2, 6, 1, 3, 5, 7, 8]
        det_935 = [3, 1, 4, 5, 6, 8, 0, 2, 7]

        det_extended = [4, 12, 0, 2, 6, 10, 13, 14, 15, 17, 1, 3, 5, 7, 8, 9, 11, 16]
        self.assertListEqual(det_extended, bC.extend_determinant(configuration_953, configuration_935, det_953, det_935))

class TestReverseSpinsMethods(unittest.TestCase):
    def test_reverse_spins(self):
        bond_matrix_9 = lF.build_rectangular_bond_matrix(1, 1, 3)

        configuration_953 = bC.Configuration(bond_matrix_9, 1, 3, [], 'periodic', 1, 1, 3)
        det = [4, 1, 3, 5, 0, 2, 6, 7, 8]

        self.assertListEqual(bC.reverse_spins(configuration_953, det)[1], [4, 0, 2, 6, 7, 8, 1, 3, 5])

    def test_basis_flip_953(self):
        bond_matrix_9 = lF.build_rectangular_bond_matrix(1, 1, 3)

        configuration_953 = bC.Configuration(bond_matrix_9, 1, 3, [], 'periodic', 1, 1, 3)
        basis_953 = bC.construct_basis(configuration_953)    
        basis_flip = bC.reverse_spins_basis(basis_953)

        configuration_935 = bC.Configuration(bond_matrix_9, 1, 5, [], 'periodic', 1, 1, 3)
        basis_935 = bC.construct_basis(configuration_935)      

        for det in basis_flip.determinants:
            self.assertIn(det, basis_935.determinants)

    def test_reverse_spins_vect(self):
        bond_matrix_4 = lF.build_rectangular_bond_matrix(1, 1, 2)
        configuration_421 = bC.Configuration(bond_matrix_4, 1, 1, [], 'periodic', 1, 1, 2)
        basis_421 = bC.construct_basis(configuration_421)
        vect = np.array([1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1])
        
        vect_reverse = np.array([0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0])

        self.assertListEqual(vect_reverse.tolist(), bC.reverse_spins_vect(basis_421, vect).tolist())

class TestSitesMethods(unittest.TestCase):
    def test_spin_site(self):
        bond_matrix_9 = lF.build_rectangular_bond_matrix(1, 1, 3)
        configuration_944 = bC.Configuration(bond_matrix_9, 1, 4, [], 'periodic', 1, 1, 3)

        spins = {
            0 : [4],
            0.5 : [0, 2, 6, 7],
            -0.5 : [1, 3, 5, 8]
        }
        det = spins[0] + spins[-0.5] + spins[0.5]
        for site in range(9):
            for spin, sites in spins.items():
                if site in sites :
                    spin_site = spin
            self.assertEqual(spin_site, bC.get_spin_site(configuration_944, det, site))

    def test_bond_linear(self): # Must be tested on weirder lattices than 9 or 18
        # Periodic indexing
        bond_matrix_18 = lF.build_rectangular_bond_matrix(2, 1, 3)
        configuration_18_periodic = bC.Configuration(bond_matrix_18, 2, 8, [], 'periodic', 2, 1, 3)

        bonds = {
            (0, 1, 2) : True,
            (0, 2, 1) : True,
            (0, 1, 6) : False,
            (0, 1, 3) : ValueError,
            (0, 1, 7) : False,
            (17, 11, 5) : True,
            (8, 7, 6) : True
        }
        for bond, result in bonds.items():
            if isinstance(result, bool):
                self.assertIs(bC.is_bond_linear(configuration_18_periodic, bond[0], bond[1], bond[2]), result)
            else:
                with self.assertRaises(ValueError):
                    bC.is_bond_linear(configuration_18_periodic, bond[0], bond[1], bond[2])

        # Peripheral indexing
        bond_matrix_9_neel = lF.build_bond_matrix_neel_embedding(1, 1, 3)
        configuration_9_neel = bC.Configuration(bond_matrix_9_neel, 1, 12, [], 'peripheral', 1, 1, 3)

        bonds_neel = {
            (0, 1, 2) : True,
            (0, 2, 1) : True,
            (1, 2, 12) : True,
            (9, 0, 1) : False,
            (9, 0, 3) : True,
            (14, 8, 7) : True,
            (19, 3, 6) : False,
            (4, 7, 16) : True,
            (12, 13, 14) : True,
            (13, 14, 15) : ValueError,
            (11, 2, 12) : False,
            (16, 7, 6) : False
        }
        for bond, result in bonds_neel.items():
            if isinstance(result, bool):
                self.assertIs(bC.is_bond_linear(configuration_9_neel, bond[0], bond[1], bond[2]), result)
            else:
                with self.assertRaises(ValueError):
                    bC.is_bond_linear(configuration_9_neel, bond[0], bond[1], bond[2])  
        
        # Clusters indexing
        bond_matrix_9_emb = lF.build_bond_matrix_block_embedding(1, 1, 3)
        configuration_9_emb = bC.Configuration(bond_matrix_9_emb, 2, 8, [], 'clusters', 2, 1, 3)

        bonds_emb = {
            (0, 1, 2) : True,
            (0, 2, 1) : True,
            (4, 5, 12) : True,
            (2, 9, 10) : True,
            (2, 9, 12) : False,
            (7, 8, 12) : ValueError,
            (1, 2, 10) : ValueError
        }
        for bond, result in bonds_emb.items():
            if isinstance(result, bool):
                self.assertIs(bC.is_bond_linear(configuration_9_emb, bond[0], bond[1], bond[2]), result)
            else:
                with self.assertRaises(ValueError):
                    bC.is_bond_linear(configuration_9_emb, bond[0], bond[1], bond[2]            )
        
class TestProbabilitiesMethods(unittest.TestCase):
    def test_probability_spin_lattice(self):
        bond_matrix = lF.build_rectangular_bond_matrix(1, 1, 3)
        configuration = bC.Configuration(bond_matrix, 1, 4, [], 'periodic', 1, 1, 3)
        basis = bC.construct_basis(configuration)

        det_1 = [4, 2, 3, 7, 8, 0, 1, 5, 6]
        det_2 = [4, 0, 5, 6, 7, 1, 2, 3, 8]
        det_3 = [6, 3, 5, 7, 8, 0, 1, 2, 4]
        det_4 = [1, 3, 4, 5, 7, 0, 2, 6, 8]
        det_5 = [7, 0, 2, 4, 6, 1, 3, 5, 8]

        vect = np.zeros((configuration.nb_conf, 1))
        vect[bC.get_index_in_basis(configuration, det_1), 0] = 1/2
        vect[bC.get_index_in_basis(configuration, det_2), 0] = -1/2
        vect[bC.get_index_in_basis(configuration, det_3), 0] = 1/mp.sqrt(6)
        vect[bC.get_index_in_basis(configuration, det_4), 0] = 1/mp.sqrt(6)
        vect[bC.get_index_in_basis(configuration, det_5), 0] = -1/mp.sqrt(6)

        probabilities_some_sites = {
            (4, 0) : 1/2,
            (0, 1) : 7/12,
            (5, 1) : 5/12,
            (8, -1) : 5/12,
            (7, -1) : 5/6,
            (1, 0) : 1/6,
            (1, -1) : 0,
            (7, 1) : 0
        }
        probabilities_spin_flip_some_sites = {
            (4, 0) : 1/2,
            (0, -1) : 7/12,
            (5, -1) : 5/12,
            (8, 1) : 5/12,
            (7, 1) : 5/6,
            (1, 0) : 1/6,
            (1, 1) : 0,
            (7, -1) : 0
        }
        for (site, spin), proba in probabilities_some_sites.items():
            self.assertAlmostEqual(proba, bC.get_probability_spin_lattice(basis, vect, spin, reverse_spins=False)[site], 6)
        for (site, spin), proba in probabilities_spin_flip_some_sites.items():
            self.assertAlmostEqual(proba, bC.get_probability_spin_lattice(basis, vect, spin, reverse_spins=True)[site], 6)

    def test_most_dominant_determinants(self):
        bond_matrix = lF.build_rectangular_bond_matrix(1, 1, 3)
        configuration = bC.Configuration(bond_matrix, 1, 4, [], 'periodic', 1, 1, 3)
        basis = bC.construct_basis(configuration)

        vect = np.zeros((configuration.nb_conf, 1))
        vect[32, 0] = mp.sqrt(5/15)
        vect[456, 0] = -mp.sqrt(4/15)
        vect[168, 0] = -mp.sqrt(3/15)
        vect[75, 0] = mp.sqrt(2/15)
        vect[518, 0] = -mp.sqrt(1/15)
        list_indices = [32, 456, 168, 75, 518]
        list_coeffs = [mp.sqrt(5/15), -mp.sqrt(4/15), -mp.sqrt(3/15), mp.sqrt(2/15), -mp.sqrt(1/15)]

        coeffs, most_dominant_dets = bC.get_dominant_determinants(basis, vect, 5)
        for ind, det in enumerate(most_dominant_dets):
            self.assertEqual(det, bC.get_hash_basis(1, 4, 4, list_indices[ind]))
            self.assertAlmostEqual(coeffs[ind], list_coeffs[ind], 10)

    def test_rotate_determinant(self):
        bond_matrix = lF.build_rectangular_bond_matrix(1, 1, 3)
        configuration = bC.Configuration(bond_matrix, 1, 4, [], 'periodic', 1, 1, 3)

        det = [0, 2, 4, 6, 7, 1, 3, 5, 8]
        dets_rotated = {
            0 : det,
            1 : [6, 0, 4, 5, 8, 1, 2, 3, 7],
            2 : [8, 1, 2, 4, 6, 0, 3, 5, 7],
            3 : [2, 0, 3, 4, 8, 1, 5, 6, 7]
        }
        for rot, det_rotated in dets_rotated.items():
            self.assertEqual(det_rotated, bC.rotate_determinant(rot, configuration, det))

if __name__ == '__main__':
    unittest.main()