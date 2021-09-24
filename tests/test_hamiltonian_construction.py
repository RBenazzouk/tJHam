import unittest
import numpy as np
import math as mp

import latticeFunctions as lF
import baseConstruction as bC
import interactions as itr
import initialComputations as iC
import hamiltonianConstruction as hC
import davidson as dav

class TestHamiltonianMethods(unittest.TestCase):
    def test_build_hamiltonian(self):
        interactions = itr.Interactions(t=-0.55, J=0.123, Jhpar=0.156, Jhper=0.104, hSDper=-0.041, hSDpar=-0.08, \
            tNNN=0.112, tnNNN=-0.047, VNN=1.77, VNNNper=0.79)
        #interactions = itr.Interactions(t=1, J=1, Jhpar=1, Jhper=1, hSDper=1, hSDpar=1, \
        #    tNNN=2, tnNNN=1, VNN=1, VNNNper=1)
        bool_interactions = iC.BoolInteractions(bt=True, bz=True, bij=True, bSD=True, btN=True, bV=True, b_spin_multipicities=False)

        bond_matrix_4 = lF.build_rectangular_bond_matrix(1, 1, 2)
        configuration_4 = bC.Configuration(bond_matrix_4, 1, 1, [], 'periodic', 1, 1, 2)
        basis_4 = bC.construct_basis(configuration_4)

        t = interactions.t
        Jhper = interactions.Jhper
        tNNN = interactions.tNNN
        hSDper = interactions.hSDper
        hamiltonian_4 = hC.build_hamiltonian(configuration_4, interactions, basis_4, bool_interactions, sites_restriction=[], verbose=False).toarray().tolist()
        #print(hC.build_hamiltonian(configuration_4, interactions, basis_4, bool_interactions, sites_restriction=[], verbose=False).toarray())
        expected_hamiltonian_4 = [
            [-Jhper/2, 0, Jhper/2, t, 0, 0, 0, t, 0, hSDper, tNNN - hSDper, 0],
            [0, -Jhper/2, Jhper/2, 0, t, 0, t, 0, 0, hSDper, 0, tNNN - hSDper],
            [Jhper/2, Jhper/2, -Jhper, 0, 0, t, 0, 0, t, tNNN - 2 * hSDper, hSDper, hSDper],
            [t, 0, 0, -Jhper/2, Jhper/2, 0, tNNN - hSDper, hSDper, 0, t, 0, 0],
            [0, t, 0, Jhper/2, -Jhper, Jhper/2, hSDper, tNNN - 2 * hSDper, hSDper, 0, 0, t],
            [0, 0, t, 0, Jhper/2, -Jhper/2, 0, hSDper, tNNN - hSDper, 0, t, 0],
            [0, t, 0, tNNN - hSDper, hSDper, 0, -Jhper/2, Jhper/2, 0, t, 0, 0],
            [t, 0, 0, hSDper, tNNN - 2 * hSDper, hSDper, Jhper/2, -Jhper, Jhper/2, 0, t, 0],
            [0, 0, t, 0, hSDper, tNNN - hSDper, 0, Jhper/2, -Jhper/2, 0, 0, t],
            [hSDper, hSDper, tNNN - 2 * hSDper, t, 0, 0, t, 0, 0, -Jhper, Jhper/2, Jhper/2],
            [tNNN - hSDper, 0, hSDper, 0, 0, t, 0, t, 0, Jhper/2, -Jhper/2, 0],
            [0, tNNN - hSDper, hSDper, 0, t, 0, 0, 0, t, Jhper/2, 0, -Jhper/2],
        ]
        self.assertListEqual(expected_hamiltonian_4, hamiltonian_4)

    def test_assert_hermitian(self):
        bond_matrix = lF.build_rectangular_bond_matrix(1, 1, 3)
        interactions = itr.Interactions(t=-0.55, J=0.123, Jhpar=0.156, Jhper=0.104, hSDper=-0.041, hSDpar=-0.08, \
            tNNN=0.112, tnNNN=-0.047, VNN=1.77, VNNNper=0.79)
        bool_interactions = iC.BoolInteractions(bt=True, bz=True, bij=True, bSD=True, btN=True, bV=True, b_spin_multipicities=True)
        for nholes in range(9):
            nbeta = (9 - nholes)//2
            configuration = bC.Configuration(bond_matrix, nholes, nbeta, [], 'periodic', 1, 1, 3)
            basis = bC.construct_basis(configuration)

            hamiltonian = hC.build_hamiltonian(configuration, interactions, basis, bool_interactions, sites_restriction=[], verbose=False)

            self.assertListEqual(hamiltonian.toarray().tolist(), hamiltonian.toarray().T.tolist())

    def test_assert_rotation_invariant(self):
        bond_matrix = lF.build_rectangular_bond_matrix(1, 1, 3)
        configuration = bC.Configuration(bond_matrix, 1, 4, [], 'periodic', 1, 1, 3)
        interactions = itr.Interactions(t=-0.55, J=0.123, Jhpar=0.156, Jhper=0.104, hSDper=-0.041, hSDpar=-0.08, \
            tNNN=0.112, tnNNN=-0.047, VNN=1.77, VNNNper=0.79)
        bool_interactions = iC.BoolInteractions(bt=True, bz=True, bij=True, bSD=True, btN=True, bV=True, b_spin_multipicities=False)
        basis = bC.construct_basis(configuration)
        determinants = basis.determinants

        hamiltonian = hC.build_hamiltonian(configuration, interactions, basis, bool_interactions, [], False)

        for ind, det in enumerate(determinants):
            for rot in range(1, 4):
                det_rot = bC.rotate_determinant(rot, configuration, det)
                ind_rot = bC.get_index_in_basis(configuration, det_rot)
                if round(hamiltonian[ind, ind], 5) != round(hamiltonian[ind_rot, ind_rot], 5):
                    print("[emb] {} : {} : {}".format(ind, det, round(hamiltonian[ind, ind], 5)))
                    print("[emb] {} : {} : {} [{}]".format(ind_rot, det_rot, round(hamiltonian[ind_rot, ind_rot]), rot))
                self.assertAlmostEqual(hamiltonian[ind, ind], hamiltonian[ind_rot, ind_rot], 5)


TestHamiltonianMethods().test_assert_rotation_invariant()
#if __name__ == '__main__':
#    unittest.main()