import unittest
import latticeFunctions as lF
import baseConstruction as bC

class TestNeighborsMethods(unittest.TestCase):

    def test_neighbors_block_9(self):
        bond_matrix = lF.build_rectangular_bond_matrix(1, 1, 3)
        neighborhood = {
            0 : [1, 3],
            1 : [0, 2, 4],
            2 : [1, 5],
            3 : [0, 4, 6],
            4 : [1, 3, 5, 7],
            5 : [2, 4, 8],
            6 : [3, 7],
            7 : [6, 4, 8],
            8 : [5, 7]
        }
        for site, neighbors in neighborhood.items():
            self.assertListEqual(sorted(neighbors), sorted(lF.get_neighbors(site, bond_matrix)))
    
    def test_neighbors_block_9_sites_restriction(self):
        bond_matrix = lF.build_rectangular_bond_matrix(1, 1, 3)
        sites_restriction = [0, 1, 2, 3, 4]
        neighborhood = {
            0 : [1, 3],
            1 : [0, 2, 4],
            2 : [1],
            3 : [0, 4],
            4 : [1, 3],
            5 : [2, 4],
            6 : [3],
            7 : [4],
            8 : []
        }
        for site, neighbors in neighborhood.items():
            self.assertListEqual(sorted(neighbors), sorted(lF.get_neighbors(site, bond_matrix, sites_restriction)))
    
    def test_neighbors_block_9_embedded(self):
        # Néel peripheral embedding
        bond_matrix_neel = lF.build_bond_matrix_neel_embedding(1, 1, 3)
        neighborhood_neel = {
            0 : [1, 3, 9, 20],
            4 : [1, 3, 5, 7],
            5 : [2, 4, 8, 13],
            12 : [2, 13],
            16 : [7, 15, 17]
        }
        for site, neighbors in neighborhood_neel.items():
            self.assertListEqual(sorted(neighbors), sorted(lF.get_neighbors(site, bond_matrix_neel)))

        # Block embedding
        bond_matrix_blocks = lF.build_bond_matrix_block_embedding(1, 1, 3)
        neighborhood_blocks = {
            0 : [1, 3],
            2 : [1, 5, 9],
            4 : [1, 3, 5, 7],
            5 : [2, 4, 8, 12],
            9 : [2, 10, 12],
            13 : [10, 12, 14, 16],
            15 : [8, 12, 16],
            17 : [14, 16]
        }
        for site, neighbors in neighborhood_blocks.items():
            self.assertListEqual(sorted(neighbors), sorted(lF.get_neighbors(site, bond_matrix_blocks)))

    def test_second_neighbors_block_9(self):
        bond_matrix = lF.build_rectangular_bond_matrix(1, 1, 3)
        neighborhood = {
            0 : [2, 4, 6],
            1 : [3, 5, 7],
            2 : [0, 4, 8],
            3 : [1, 5, 7],
            4 : [0, 2, 6, 8],
            5 : [1, 3, 7],
            6 : [0, 4, 8],
            7 : [1, 3, 5],
            8 : [2, 4, 6]
        }
        for site, neighbors in neighborhood.items():
            self.assertListEqual(sorted(neighbors), sorted(lF.get_second_neighbors(site, bond_matrix)))

    def test_diagonal_second_neighbors_block_9(self):
        # Periodic lattice type
        bond_matrix = lF.build_rectangular_bond_matrix(1, 1, 3)
        configuration = bC.Configuration(bond_matrix, 0, 0, [], 'periodic', 1, 1, 3)
        neighborhood = {
            0 : [4],
            1 : [3, 5],
            2 : [4],
            3 : [1, 7],
            4 : [0, 2, 6, 8],
            5 : [1, 7],
            6 : [4],
            7 : [3, 5],
            8 : [4]
        }
        for site, neighbors in neighborhood.items():
            self.assertListEqual(sorted(neighbors), sorted(lF.get_diagonal_second_neighbors(configuration, site)))
        
        # Peripheral lattice type
        bond_matrix_neel = lF.build_bond_matrix_neel_embedding(1, 1, 3)
        configuration_neel = bC.Configuration(bond_matrix_neel, 0, 0, [], 'peripheral', 1, 1, 3)
        neighborhood_neel = {
            0 : [4, 10, 19],
            4 : [0, 2, 6, 8],
            13 : [2, 8],
            15 : [7, 14]
        }
        for site_neel, neighbors_neel in neighborhood_neel.items():
            self.assertListEqual(sorted(neighbors_neel), sorted(lF.get_diagonal_second_neighbors(configuration_neel, site_neel)))

        # Clusters lattice type
        bond_matrix_blocks = lF.build_bond_matrix_block_embedding(1, 1, 3)
        configuration_blocks = bC.Configuration(bond_matrix_blocks, 0, 0, [], 'clusters', 2, 1, 3)
        neighborhood_blocks = {
            4 : [0, 2, 6, 8],
            2 : [4, 12],
            12 : [2, 8, 10, 16],
            15 : [5, 13],
            17 : [13]
        }
        for site_blocks, neighbors_blocks in neighborhood_blocks.items():
            self.assertListEqual(sorted(neighbors_blocks), sorted(lF.get_diagonal_second_neighbors(configuration_blocks, site_blocks)))

    def test_diagonal_second_neighbors_block_9_sites_restriction(self):
        bond_matrix = lF.build_rectangular_bond_matrix(1, 1, 3)
        sites_restriction = [1, 2, 3, 5, 6, 7, 8]
        configuration = bC.Configuration(bond_matrix, 0, 0, sites_restriction, 'periodic', 1, 1, 3)
        neighborhood = {
            0 : [],
            1 : [3, 5],
            2 : [],
            3 : [1, 7],
            4 : [2, 6, 8],
            5 : [1, 7],
            6 : [],
            7 : [3, 5],
            8 : []
        }
        for site, neighbors in neighborhood.items():
            self.assertListEqual(sorted(neighbors), sorted(lF.get_diagonal_second_neighbors(configuration, site)))

if __name__ == '__main__':
    unittest.main()