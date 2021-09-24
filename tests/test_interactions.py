import unittest
import numpy as np
import math as mp

import latticeFunctions as lF
import baseConstruction as bC
import interactions as itr
import initialComputations as iC
import hamiltonianConstruction as hC
import davidson as dav

class TestInteractionMethods(unittest.TestCase):
    def test_hopping_integral(self):
        # Periodic lattice
        bond_matrix = lF.build_rectangular_bond_matrix(1, 1, 3)
        configuration = bC.Configuration(bond_matrix, 1, 4, [], 'periodic', 1, 1, 3)

        interactions = itr.Interactions(t=-0.55, J=0.123, Jhpar=0.156, Jhper=0.104, hSDper=-0.041, hSDpar=-0.08, \
            tNNN=0.112, tnNNN=-0.047, VNN=1.77, VNNNper=0.79)

        det = [4, 1, 3, 6, 8, 0, 2, 5, 7]
        expected_dets = [[1, 3, 4, 6, 8, 0, 2, 5, 7], [3, 1, 4, 6, 8, 0, 2, 5, 7], [5, 1, 3, 6, 8, 0, 2, 4, 7], [7, 1, 3, 6, 8, 0, 2, 4, 5]]
        images = itr.compute_hopping_integral(configuration, interactions, det)
        
        self.assertEqual(len(images), len(expected_dets))
        for image in images:
            self.assertIn(image, expected_dets)

        # Néel embedding lattice
        bond_matrix_neel = lF.build_bond_matrix_neel_embedding(1, 1, 3)
        configuration_neel = bC.Configuration(bond_matrix_neel, 2, 12, [], 'peripheral', 1, 1, 3)

        det_neel = [6, 13, 1, 3, 5, 7, 9, 11, 12, 14, 15, 17, 18, 20, 0, 2, 4, 8, 10, 16, 19]
        expected_dets_neel = [[3, 13, 1, 5, 6, 7, 9, 11, 12, 14, 15, 17, 18, 20, 0, 2, 4, 8, 10, 16, 19], \
            [7, 13, 1, 3, 5, 6, 9, 11, 12, 14, 15, 17, 18, 20, 0, 2, 4, 8, 10, 16, 19], [13, 17, 1, 3, 5, 6, 7, 9, 11, 12, 14, 15, 18, 20, 0, 2, 4, 8, 10, 16, 19], \
            [13, 18, 1, 3, 5, 6, 7, 9, 11, 12, 14, 15, 17, 20, 0, 2, 4, 8, 10, 16, 19], [5, 6, 1, 3, 7, 9, 11, 12, 13, 14, 15, 17, 18, 20, 0, 2, 4, 8, 10, 16, 19], \
            [6, 12, 1, 3, 5, 7, 9, 11, 13, 14, 15, 17, 18, 20, 0, 2, 4, 8, 10, 16, 19], [6, 14, 1, 3, 5, 7, 9, 11, 12, 13, 15, 17, 18, 20, 0, 2, 4, 8, 10, 16, 19]]
        images_neel = itr.compute_hopping_integral(configuration_neel, interactions, det_neel)
        self.assertEqual(len(images_neel), len(expected_dets_neel))
        for image_neel in images_neel:
            self.assertIn(image_neel, expected_dets_neel)

        # Block embedding lattice
        bond_matrix_block = lF.build_bond_matrix_block_embedding(1, 1, 3)
        configuration_block = bC.Configuration(bond_matrix_block, 4, 6, [], 'clusters', 1, 1, 3)

        det_block = [5, 8, 9, 15, 1, 3, 7, 11, 13, 17, 0, 2, 4, 6, 10, 12, 14, 16]

        expected_dets_blocks = [[8, 9, 12, 15, 1, 3, 7, 11, 13, 17, 0, 2, 4, 5, 6, 10, 14, 16], \
            [2, 8, 9, 15, 1, 3, 7, 11, 13, 17, 0, 4, 5, 6, 10, 12, 14, 16], [4, 8, 9, 15, 1, 3, 7, 11, 13, 17, 0, 2, 5, 6, 10, 12, 14, 16], \
            [5, 7, 9, 15, 1, 3, 8, 11, 13, 17, 0, 2, 4, 6, 10, 12, 14, 16], [2, 5, 8, 15, 1, 3, 7, 11, 13, 17, 0, 4, 6, 9, 10, 12, 14, 16], \
            [5, 8, 10, 15, 1, 3, 7, 11, 13, 17, 0, 2, 4, 6, 9, 12, 14, 16], [5, 8, 12, 15, 1, 3, 7, 11, 13, 17, 0, 2, 4, 6, 9, 10, 14, 16], \
            [5, 8, 9, 12, 1, 3, 7, 11, 13, 17, 0, 2, 4, 6, 10, 14, 15, 16], [5, 8, 9, 16, 1, 3, 7, 11, 13, 17, 0, 2, 4, 6, 10, 12, 14, 15], 
        ]
        images_block = itr.compute_hopping_integral(configuration_block, interactions, det_block)
        self.assertEqual(len(images_block), len(expected_dets_blocks))
        for image_block in images_block:
            self.assertIn(image_block, expected_dets_blocks)
    
    def test_next_neighbor_hopping_integral(self):
        # Periodic lattice
        bond_matrix = lF.build_rectangular_bond_matrix(1, 1, 3)
        configuration = bC.Configuration(bond_matrix, 1, 4, [], 'periodic', 1, 1, 3)

        interactions = itr.Interactions(t=-0.55, J=0.123, Jhpar=0.156, Jhper=0.104, hSDper=-0.041, hSDpar=-0.08, \
            tNNN=0.112, tnNNN=-0.047, VNN=1.77, VNNNper=0.79)

        det = [8, 1, 3, 4, 6, 0, 2, 5, 7]

        expected_dets_col = [[2, 1, 3, 4, 6, 0, 5, 7, 8], [6, 1, 3, 4, 8, 0, 2, 5, 7]]
        expected_dets_ortho = [[4, 1, 3, 6, 8, 0, 2, 5, 7]]

        images_col, images_ortho = itr.compute_next_hopping_integral(configuration, interactions, det)
        self.assertEqual(len(images_col), len(expected_dets_col))
        self.assertEqual(len(images_ortho), len(expected_dets_ortho))
        for image_col in images_col:
            self.assertIn(image_col, expected_dets_col)
        for image_ortho in images_ortho:
            self.assertIn(image_ortho, expected_dets_ortho)

        # Néel embedding lattice
        bond_matrix_neel = lF.build_bond_matrix_neel_embedding(1, 1, 3)
        configuration_neel = bC.Configuration(bond_matrix_neel, 1, 9, [], 'peripheral', 1, 1, 3)

        det_neel = [1, 0, 2, 4, 6, 8, 10, 13, 16, 19, 3, 5, 7, 9, 11, 12, 14, 15, 17, 18, 20]
        expected_dets_col_neel = [
            [7, 0, 2, 4, 6, 8, 10, 13, 16, 19, 1, 3, 5, 9, 11, 12, 14, 15, 17, 18, 20], 
            [12, 0, 2, 4, 6, 8, 10, 13, 16, 19, 1, 3, 5, 7, 9, 11, 14, 15, 17, 18, 20], 
            [20, 0, 2, 4, 6, 8, 10, 13, 16, 19, 1, 3, 5, 7, 9, 11, 12, 14, 15, 17, 18]
        ]
        expected_dets_ortho_neel = [
            [5, 0, 2, 4, 6, 8, 10, 13, 16, 19, 1, 3, 7, 9, 11, 12, 14, 15, 17, 18, 20],
            [3, 0, 2, 4, 6, 8, 10, 13, 16, 19, 1, 5, 7, 9, 11, 12, 14, 15, 17, 18, 20],
            [9, 0, 2, 4, 6, 8, 10, 13, 16, 19, 1, 3, 5, 7, 11, 12, 14, 15, 17, 18, 20],
            [11, 0, 2, 4, 6, 8, 10, 13, 16, 19, 1, 3, 5, 7, 9, 12, 14, 15, 17, 18, 20]
        ]

        images_col_neel, images_ortho_neel = itr.compute_next_hopping_integral(configuration_neel, interactions, det_neel)
        self.assertEqual(len(images_col_neel), len(expected_dets_col_neel))
        self.assertEqual(len(images_ortho_neel), len(expected_dets_ortho_neel))
        for image_col_neel in images_col_neel:
            self.assertIn(image_col_neel, expected_dets_col_neel)
        for image_ortho_neel in images_ortho_neel:
            self.assertIn(image_ortho_neel, expected_dets_ortho_neel)

        # Block embedding lattice
        bond_matrix_block = lF.build_bond_matrix_block_embedding(1, 1, 3)
        configuration_block = bC.Configuration(bond_matrix_block, 2, 8, [], 'clusters', 2, 1, 3)

        det_block = [2, 10, 1, 3, 5, 7, 9, 11, 13, 17, 0, 4, 6, 8, 12, 14, 15, 16]

        expected_dets_col_block = [
            [0, 10, 1, 3, 5, 7, 9, 11, 13, 17, 2, 4, 6, 8, 12, 14, 15, 16], 
            [8, 10, 1, 3, 5, 7, 9, 11, 13, 17, 0, 2, 4, 6, 12, 14, 15, 16],
            [2, 16, 1, 3, 5, 7, 9, 11, 13, 17, 0, 4, 6, 8, 10, 12, 14, 15]
        ]
        expected_dets_ortho_block = [
            [4, 10, 1, 3, 5, 7, 9, 11, 13, 17, 0, 2, 6, 8, 12, 14, 15, 16],
            [10, 12, 1, 3, 5, 7, 9, 11, 13, 17, 0, 2, 4, 6, 8, 14, 15, 16],
            [2, 12, 1, 3, 5, 7, 9, 11, 13, 17, 0, 4, 6, 8, 10, 14, 15, 16],
            [2, 14, 1, 3, 5, 7, 9, 11, 13, 17, 0, 4, 6, 8, 10, 12, 15, 16]
        ]

        images_col_block, images_ortho_block = itr.compute_next_hopping_integral(configuration_block, interactions, det_block)
        self.assertEqual(len(images_col_block), len(expected_dets_col_block))
        self.assertEqual(len(images_ortho_block), len(expected_dets_ortho_block))
        for image_col_block in images_col_block:
            self.assertIn(image_col_block, expected_dets_col_block)
        for image_ortho_block in images_ortho_block:
            self.assertIn(image_ortho_block, expected_dets_ortho_block)

    def test_static_spin_interaction(self):
        bond_matrix = lF.build_rectangular_bond_matrix(2, 1, 3)
        configuration = bC.Configuration(bond_matrix, 3, 6, [], 'periodic', 2, 1, 3)

        interactions = itr.Interactions(t=-0.55, J=0.123, Jhpar=0.156, Jhper=0.104, hSDper=-0.041, hSDpar=-0.08, \
            tNNN=0.112, tnNNN=-0.047, VNN=1.77, VNNNper=0.79)
        J = interactions.J
        Jhpar = interactions.Jhpar
        Jhper = interactions.Jhper

        det = [4, 7, 14, 2, 5, 6, 9, 12, 17, 0, 1, 3, 8, 10, 11, 13, 15, 16]
        expected_sz = -J - 2 * Jhpar - 3 * Jhper

        sz = itr.compute_static_spin_interaction(configuration, interactions, det)

        self.assertEqual(sz, expected_sz)

    def test_spin_flip(self):
        bond_matrix = lF.build_rectangular_bond_matrix(2, 1, 3)
        configuration = bC.Configuration(bond_matrix, 2, 8, [], 'periodic', 2, 1, 3)

        interactions = itr.Interactions(t=-0.55, J=0.123, Jhpar=0.156, Jhper=0.104, hSDper=-0.041, hSDpar=-0.08, \
            tNNN=0.112, tnNNN=-0.047, VNN=1.77, VNNNper=0.79)

        det = [7, 10, 0, 3, 4, 9, 11, 15, 16, 17, 1, 2, 5, 6, 8, 12, 13, 14]

        expected_images_par = [[7, 10, 0, 3, 4, 8, 11, 15, 16, 17, 1, 2, 5, 6, 9, 12, 13, 14]]
        expected_images_per = [
            [7, 10, 1, 3, 4, 9, 11, 15, 16, 17, 0, 2, 5, 6, 8, 12, 13, 14],
            [7, 10, 3, 4, 6, 9, 11, 15, 16, 17, 0, 1, 2, 5, 8, 12, 13, 14],
            [7, 10, 0, 3, 5, 9, 11, 15, 16, 17, 1, 2, 4, 6, 8, 12, 13, 14],
            [7, 10, 0, 3, 4, 5, 9, 15, 16, 17, 1, 2, 6, 8, 11, 12, 13, 14]
        ]
        expected_images_undoped = [
            [7, 10, 0, 2, 4, 9, 11, 15, 16, 17, 1, 3, 5, 6, 8, 12, 13, 14],
            [7, 10, 0, 3, 4, 9, 11, 14, 16, 17, 1, 2, 5, 6, 8, 12, 13, 15]
        ]

        images_par, images_per, images_undoped = itr.compute_spin_flip(configuration, interactions, det)

        self.assertEqual(len(images_par), len(expected_images_par))
        for image_par in images_par:
            self.assertIn(image_par, expected_images_par)

        self.assertEqual(len(images_per), len(expected_images_per))
        for image_per in images_per:
            self.assertIn(image_per, expected_images_per)

        self.assertEqual(len(images_undoped), len(expected_images_undoped))
        for image_undoped in images_undoped:
            self.assertIn(image_undoped, expected_images_undoped)

    def test_singlet_displacement(self):
        bond_matrix = lF.build_bond_matrix_block_embedding(1, 1, 3)
        configuration = bC.Configuration(bond_matrix, 2, 8, [], 'clusters', 2, 1, 3)

        interactions = itr.Interactions(t=-0.55, J=0.123, Jhpar=0.156, Jhper=0.104, hSDper=-0.041, hSDpar=-0.08, \
            tNNN=0.112, tnNNN=-0.047, VNN=1.77, VNNNper=0.79)

        det = [4, 15, 3, 6, 7, 8, 10, 11, 12, 14, 0, 1, 2, 5, 9, 13, 16, 17]

        expected_images_par = [
            [12, 15, 3, 4, 6, 7, 8, 10, 11, 14, 0, 1, 2, 5, 9, 13, 16, 17], [12, 15, 3, 5, 6, 7, 8, 10, 11, 14, 0, 1, 2, 4, 9, 13, 16, 17],
            [4, 9, 3, 6, 7, 8, 10, 11, 12, 14, 0, 1, 2, 5, 13, 15, 16, 17], [4, 9, 3, 6, 7, 8, 10, 11, 14, 15, 0, 1, 2, 5, 12, 13, 16, 17]
        ]
        expected_images_per = [
            [0, 15, 3, 6, 7, 8, 10, 11, 12, 14, 1, 2, 4, 5, 9, 13, 16, 17], [0, 15, 4, 6, 7, 8, 10, 11, 12, 14, 1, 2, 3, 5, 9, 13, 16, 17],
            [8, 15, 3, 4, 6, 7, 10, 11, 12, 14, 0, 1, 2, 5, 9, 13, 16, 17], [8, 15, 3, 5, 6, 7, 10, 11, 12, 14, 0, 1, 2, 4, 9, 13, 16, 17],
            [4, 5, 3, 6, 7, 8, 10, 11, 12, 14, 0, 1, 2, 9, 13, 15, 16, 17], [4, 5, 3, 6, 7, 10, 11, 12, 14, 15, 0, 1, 2, 8, 9, 13, 16, 17], 
            [4, 5, 3, 6, 7, 8, 10, 11, 12, 14, 0, 1, 2, 9, 13, 15, 16, 17], [4, 5, 3, 6, 7, 8, 10, 11, 14, 15, 0, 1, 2, 9, 12, 13, 16, 17], 
            [4, 13, 3, 6, 7, 8, 10, 11, 12, 14, 0, 1, 2, 5, 9, 15, 16, 17], [4, 13, 3, 6, 7, 8, 10, 11, 14, 15, 0, 1, 2, 5, 9, 12, 16, 17]
        ]

        images_par, images_per = itr.compute_singlet_displacement(configuration, interactions, det)

        for image_par in images_par:
            for det in image_par:
                self.assertIn(det, expected_images_par)

        for image_per in images_per:
            for det in image_per:
                self.assertIn(det, expected_images_per)

    def test_hole_repulsion(self):
        bond_matrix = lF.build_bond_matrix_block_embedding(1, 1, 3)
        configuration = bC.Configuration(bond_matrix, 6, 6, [], 'clusters', 2, 1, 3)

        interactions = itr.Interactions(t=-0.55, J=0.123, Jhpar=0.156, Jhper=0.104, hSDper=-0.041, hSDpar=-0.08, \
            tNNN=0.112, tnNNN=-0.047, VNN=1.77, VNNNper=0.79)
        VNN = interactions.VNN
        VNNNper = interactions.VNNNper

        det = [1, 4, 8, 9, 14, 15, 3, 5, 7, 10, 11, 16, 0, 2, 6, 12, 13, 17]
        
        expected_repulsion = 2 * VNN + VNNNper
        repulsion = itr.compute_hole_repulsion(configuration, interactions, det)

        self.assertEqual(repulsion, expected_repulsion)

    def test_spin_z_two_bodies(self):
        bond_matrix = lF.build_rectangular_bond_matrix(1, 1, 2)
        configuration = bC.Configuration(bond_matrix, 1, 1, [], 'periodic', 1, 1, 2)

        vect = np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0])

        expected_vect = np.array([0, 0, 0, 1/4, 0, -1/4, 0, 0, 0, 0, 0, 0])

        self.assertListEqual(expected_vect.tolist(), itr.compute_spins_z_two_bodies(configuration, vect, 2, 3).tolist())

    def test_S_plus_S_minus(self):
        bond_matrix = lF.build_rectangular_bond_matrix(1, 1, 2)
        configuration = bC.Configuration(bond_matrix, 1, 1, [], 'periodic', 1, 1, 2)

        vect = np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0])

        expected_vect = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

        self.assertListEqual(expected_vect.tolist(), itr.compute_S_plus_S_minus(configuration, vect, 3, 2).tolist())

    def test_S2(self):
        bond_matrix = lF.build_rectangular_bond_matrix(1, 1, 2)
        configuration = bC.Configuration(bond_matrix, 1, 1, [], 'periodic', 1, 1, 2)

        vect = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

        expected_vect = np.array([0, 0, 0, 1, 1, 7/4, 0, 0, 0, 0, 0, 0])

        self.assertListEqual(expected_vect.tolist(), itr.compute_S2(configuration, vect).tolist())

    def test_spin_number(self):

        bond_matrix = lF.build_rectangular_bond_matrix(1, 1, 2)
        configuration = bC.Configuration(bond_matrix, 1, 1, [], 'periodic', 1, 1, 2)

        # Computed eigenstates
        basis = bC.construct_basis(configuration)
        interactions = itr.Interactions(t=-0.55, J=0.123, Jhpar=0.156, Jhper=0.104, hSDper=-0.041, hSDpar=-0.08, \
            tNNN=0.112, tnNNN=-0.047, VNN=1.77, VNNNper=0.79)
        bool_interactions = iC.BoolInteractions(bt=True, bz=True, bij=True, bSD=True, btN=True, bV=True, b_spin_multipicities=True)

        hamiltonian = hC.build_hamiltonian(configuration, interactions, basis, bool_interactions, [], False)
        eigenstates = dav.modified_davidson_algorithm(hamiltonian, 1)
        self.assertEqual(itr.compute_spin_number(configuration, eigenstates.eigvects[:,0]), 2.)

        # Theoretical eigenstates of the H421 Hamiltonian

        # Hole in 0
        Q0 = np.zeros((12, 1))
        Q0[bC.get_index_in_basis(configuration, [0, 1, 2, 3]), 0], Q0[bC.get_index_in_basis(configuration, [0, 2, 1, 3]), 0], \
            Q0[bC.get_index_in_basis(configuration, [0, 3, 1, 2]), 0] = 1/mp.sqrt(3), 1/mp.sqrt(3), 1/mp.sqrt(3)

        Ds0 = np.zeros((12, 1))
        Ds0[bC.get_index_in_basis(configuration, [0, 1, 2, 3]), 0], Ds0[bC.get_index_in_basis(configuration, [0, 2, 1, 3]), 0] = 1/mp.sqrt(2), -1/mp.sqrt(2)

        Da0 = np.zeros((12, 1))
        Da0[bC.get_index_in_basis(configuration, [0, 1, 2, 3]), 0], Da0[bC.get_index_in_basis(configuration, [0, 2, 1, 3]), 0], \
            Da0[bC.get_index_in_basis(configuration, [0, 3, 1, 2]), 0] = 1/mp.sqrt(6), 1/mp.sqrt(6), -2/mp.sqrt(6)

        # Hole in 3
        Q3 = np.zeros((12, 1))
        Q3[bC.get_index_in_basis(configuration, [3, 1, 0, 2]), 0], Q3[bC.get_index_in_basis(configuration, [3, 2, 0, 1]), 0], \
            Q3[bC.get_index_in_basis(configuration, [3, 0, 1, 2]), 0] = 1/mp.sqrt(3), 1/mp.sqrt(3), 1/mp.sqrt(3)

        Ds3 = np.zeros((12, 1))
        Ds3[bC.get_index_in_basis(configuration, [3, 1, 0, 2]), 0], Ds3[bC.get_index_in_basis(configuration, [3, 2, 0, 1]), 0] = 1/mp.sqrt(2), -1/mp.sqrt(2)

        Da3 = np.zeros((12, 1))
        Da3[bC.get_index_in_basis(configuration, [3, 1, 0, 2]), 0], Da3[bC.get_index_in_basis(configuration, [3, 2, 0, 1]), 0], \
            Da3[bC.get_index_in_basis(configuration, [3, 0, 1, 2]), 0] = 1/mp.sqrt(6), 1/mp.sqrt(6), -2/mp.sqrt(6)

        # Fundamentals doublets
        Dss03=(1/mp.sqrt(2))*(Ds0+Ds3)
        Dsa03=(1/mp.sqrt(2))*(Ds0-Ds3)
        Das03=(1/mp.sqrt(2))*(Da0+Da3)
        Daa03=(1/mp.sqrt(2))*(Da0-Da3) 

        self.assertEqual(itr.compute_spin_number(configuration, Q0), 4.)
        self.assertEqual(itr.compute_spin_number(configuration, Q3), 4.)

        self.assertEqual(itr.compute_spin_number(configuration, Dss03), 2.)
        self.assertEqual(itr.compute_spin_number(configuration, Dsa03), 2.)
        self.assertEqual(itr.compute_spin_number(configuration, Das03), 2.)
        self.assertEqual(itr.compute_spin_number(configuration, Daa03), 2.)

if __name__ == '__main__':
    unittest.main()