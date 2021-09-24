import latticeFunctions as lF
import baseConstruction as bC
import interactions as itr
import davidson as dav
import hamiltonianConstruction as hC
import initialComputations as iC
import embedding as emb
import postProcessing as pP

import numpy as np
np.set_printoptions(linewidth=1000)

import math as mp
import scipy.sparse as sp
import scipy.sparse.linalg as lin
import time

## eV
interactions = itr.Interactions(t=-0.55, J=0.123, Jhpar=0.156, Jhper=0.104, hSDper=-0.041, hSDpar=-0.08, \
    tNNN=0.112, tnNNN=-0.047, VNN=1.77, VNNNper=0.79)


### H4-1 -------------------------------------------------------------------------------------------------------
# Lattice
bond_matrix = lF.build_rectangular_bond_matrix(nbh=1, nbv=1, nblocs_side=2)
# Configuration
configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=1, nbeta=0, nbh=1, nbv=1, nblocs_side=2)
# diagonalization
diagonalization = dav.Diagonalization(nb_val=12, stp=1e4, tol=1e-3, dim_max=0)
# Interactions booleans
bool_interactions = iC.BoolInteractions(bt=True, bz=True, bij=True, bSD=True, btN=True, bV=True, bV_embedding=False, b_spin_multipicities=True)

# Data
data = iC.Data(configuration=configuration, interactions=interactions, diagonalization=diagonalization, bool_interactions=bool_interactions)
# Plotting Inputs
input_plotting = pP.InputPlotting(nb_excited=12, nb_plot=4, bool_spectrum=True, bool_holes=True, \
    bool_electrons=True, bool_determinants=True, bool_neel_embedding=False)

# iC.perform_configurations_computations(data)
# pP.generate_plots(data, input_plotting)
# pP.save_phase_space(data, iVal=0, nbt=25, nbV=25)


# data_computations_421 = iC.load_data("tJHam-421Full.json")
# ham421 = data_computations_421.hamiltonian.toarray()
# eig = np.linalg.eigh(ham421)
# V = np.zeros((12, 12))
# V[0, 5] = 1
# V[1, 3] = 1
# V[2, 4] = 1
# V[3, 10] = 1
# V[4, 9] = 1
# V[5, 11] = 1
# V[6, 0] = 1
# V[7, 2] = 1
# V[8, 1] = 1
# V[9, 7] = 1
# V[10, 8] = 1
# V[11, 6] = 1
# V = V/np.linalg.norm(V)
# eigv = np.linalg.eig(V)
# eigvects_V = eigv[1]

# eigvects_V[:, [4]], eigvects_V[:, [1]] = eigvects_V[:, [1]], eigvects_V[:, [4]]
# eigvects_V[:, [8]], eigvects_V[:, [2]] = eigvects_V[:, [2]], eigvects_V[:, [8]]
# eigvects_V[:, [8]], eigvects_V[:, [3]] = eigvects_V[:, [3]], eigvects_V[:, [8]]
# eigvects_V[:, [6]], eigvects_V[:, [4]] = eigvects_V[:, [4]], eigvects_V[:, [6]]
# eigvects_V[:, [10]], eigvects_V[:, [5]] = eigvects_V[:, [5]], eigvects_V[:, [10]]
# eigvects_V[:, [10]], eigvects_V[:, [7]] = eigvects_V[:, [7]], eigvects_V[:, [10]]
# eigvects_V[:, [9]], eigvects_V[:, [8]] = eigvects_V[:, [8]], eigvects_V[:, [9]]

# print(np.around(eig[0], 3))
# new_ham421 = np.matmul(np.linalg.inv(eigvects_V), np.matmul(ham421, eigvects_V))
# new_eig = np.linalg.eigh(new_ham421)
# print("\n", np.around(new_ham421, 3))
# print("\n", np.around(np.matmul(np.linalg.inv(eig[1]), np.matmul(ham421, eig[1])), 3))
# print("\n", np.around(np.matmul(np.linalg.inv(new_eig[1]), np.matmul(new_ham421, new_eig[1])), 3))
# stab = np.matmul(eigvects_V, np.matmul(new_eig[1], np.linalg.inv(eig[1])))
# eig_stab = np.linalg.eig(stab)
# print(np.around(np.matmul(np.linalg.inv(eig_stab[1]), np.matmul(ham421, eig_stab[1])), 3))


# Embedding
configuration_emb = bC.Configuration(bond_matrix=bond_matrix, nholes=1, nbeta=0, nbh=1, nbv=1, nblocs_side=2)
interactions_emb = itr.Interactions(t=-3, J=0, Jhpar=0, Jhper=0, hSDper=0, hSDpar=0, \
    tNNN=1, tnNNN=1, VNN=2, VNNNper=1)
data_emb = iC.Data(configuration=configuration_emb, interactions=interactions_emb, diagonalization=diagonalization, bool_interactions=bool_interactions)

# Embedding Néel
# emb.static_embedding_neel(data, self_consistency_tol=1e-2, max_iter=1000)
# pP.generate_plots_configuration(data, input_plotting)

# Embedding Blocks
# iC.save_data(iC.perform_initial_computations(data_emb))
#emb.embedding_blocks(data_emb, self_consistency_tol=1e-3, max_iter=300, dynamic_embedding=True, neel_embedding=False, neighbor_embedding=True, threshold_mel=0, reverse_spins=False)



# Embedding Neel H422
configuration_emb = bC.Configuration(bond_matrix=bond_matrix, nholes=0, nbeta=2, nbh=1, nbv=1, nblocs_side=2)
data_emb = iC.Data(configuration=configuration_emb, interactions=interactions, diagonalization=diagonalization, bool_interactions=bool_interactions)
#H422_emb = emb.embedding_neel(data_emb, 1e-6, 1000)
#print(H422_emb.eigenstates.eigvects[:,[0]])

### H4-2 -------------------------------------------------------------------------------------------------------
# Lattice
bond_matrix = lF.build_rectangular_bond_matrix(nbh=1, nbv=1, nblocs_side=2)
# Configuration
configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=2, nbeta=0, nbh=1, nbv=1, nblocs_side=2)
# diagonalization
diagonalization = dav.Diagonalization(nb_val=12, stp=1e3, tol=1e-9, dim_max=0)
# Interactions booleans
bool_interactions = iC.BoolInteractions(bt=True, bz=True, bij=True, bSD=True, btN=True, bV=True, bV_embedding=False, b_spin_multipicities=False)

# Data
data = iC.Data(configuration=configuration, interactions=interactions, diagonalization=diagonalization, bool_interactions=bool_interactions)
# Plotting Inputs
input_plotting = pP.InputPlotting(nb_excited=12, nb_plot=4, bool_spectrum=True, bool_holes=True, \
    bool_electrons=True, bool_determinants=True)

# iC.perform_configurations_computations(data)
# pP.generate_plots(data, input_plotting)
# pP.savePhaseSpace(data, iVal=0, nbt=25, nbV=25)

### H4-3 -------------------------------------------------------------------------------------------------------
# Lattice
bond_matrix = lF.build_rectangular_bond_matrix(nbh=1, nbv=1, nblocs_side=2)
# Configuration
configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=3, nbeta=0, nbh=1, nbv=1, nblocs_side=2)
# diagonalization
diagonalization = dav.Diagonalization(nb_val=4, stp=1e3, tol=1e-9, dim_max=0)
# Interactions booleans
bool_interactions = iC.BoolInteractions(bt=True, bz=True, bij=True, bSD=True, btN=True, bV=True, bV_embedding=False, b_spin_multipicities=False)

# Data
data = iC.Data(configuration=configuration, interactions=interactions, diagonalization=diagonalization, bool_interactions=bool_interactions)
# Plotting Inputs
input_plotting = pP.InputPlotting(nb_excited=12, nb_plot=4, bool_spectrum=True, bool_holes=True, \
    bool_electrons=True, bool_determinants=True)

# iC.perform_configurations_computations(data)
# pP.generate_plots(data, input_plotting)
# pP.savePhaseSpace(data, iVal=0, nbt=25, nbV=25)

### H4-0 - Heisenberg Model -----------------------------------------------------------------------------------
interactions_heisenberg = itr.Interactions(t=0, J=0.123, Jhpar=0, Jhper=0, hSDper=0, hSDpar=0, \
    tNNN=0, tnNNN=0, VNN=0, VNNNper=0)

# Lattice
bond_matrix = lF.build_rectangular_bond_matrix(nbh=1, nbv=1, nblocs_side=2)
# Configuration
configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=0, nbeta=2, nbh=1, nbv=1, nblocs_side=2)
# diagonalization
diagonalization = dav.Diagonalization(nb_val=6, stp=1e3, tol=1e-9, dim_max=0)
# Interactions booleans
bool_interactions = iC.BoolInteractions(bt=True, bz=True, bij=True, bSD=True, btN=True, bV=True, bV_embedding=False, b_spin_multipicities=False)

# Data
data = iC.Data(configuration=configuration, interactions=interactions_heisenberg, diagonalization=diagonalization, bool_interactions=bool_interactions)
# Plotting Inputs
input_plotting = pP.InputPlotting(nb_excited=6, nb_plot=4, bool_spectrum=True, bool_holes=True, \
    bool_electrons=True, bool_determinants=True)

# print("Heisenberg model on 2x2")
# data_computations_4_0 = iC.perform_initial_computations(data)
# print("Lowests eigenvalues : ", data_computations_4_0.eigenstates.eigvals)
# iC.save_data(data_computations_4_0)
# pP.generate_plots(data, input_plotting)
# pP.savePhaseSpace(data, iVal=0, nbt=25, nbV=25)

# Embedding
configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=0, nbeta=2, nbh=1, nbv=1, nblocs_side=2)
data = iC.Data(configuration=configuration, interactions=interactions_heisenberg, diagonalization=diagonalization, bool_interactions=bool_interactions)

# Embedding Néel
# emb.embedding_neel(data, self_consistency_tol=1e-5, max_iter=1000)
# pP.generate_plots_configuration(data, input_plotting)

### H8-1 -------------------------------------------------------------------------------------------------------
# Lattice
bond_matrix = lF.build_rectangular_bond_matrix(nbh=2, nbv=1, nblocs_side=2)
# Configuration
configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=1, nbeta=0, nbh=2, nbv=1, nblocs_side=2)
# diagonalization
diagonalization = dav.Diagonalization(nb_val=30, stp=1e4, tol=1e-6, dim_max=0)
# Interactions booleans
bool_interactions = iC.BoolInteractions(bt=True, bz=True, bij=True, bSD=True, btN=True, bV=True, bV_embedding=False, b_spin_multipicities=False)

# Data
data = iC.Data(configuration=configuration, interactions=interactions, diagonalization=diagonalization, bool_interactions=bool_interactions)
# Plotting Inputs
input_plotting = pP.InputPlotting(nb_excited=6, nb_plot=4, bool_spectrum=True, bool_holes=True, \
    bool_electrons=True, bool_determinants=True)

# iC.perform_configurations_computations(data)
# pP.generatePlots(data, inputPlotting)
# pP.savePhaseSpace(data, iVal=0, nbt=10, nbV=10)

### H8-2 -------------------------------------------------------------------------------------------------------
# Lattice
bond_matrix = lF.build_rectangular_bond_matrix(nbh=2, nbv=1, nblocs_side=2)
# Configuration
configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=2, nbeta=0, nbh=2, nbv=1, nblocs_side=2)
# diagonalization
diagonalization = dav.Diagonalization(nb_val=30, stp=1e4, tol=1e-6, dim_max=0)
# Interactions booleans
bool_interactions = iC.BoolInteractions(bt=True, bz=True, bij=True, bSD=True, btN=True, bV=True, bV_embedding=False, b_spin_multipicities=False)

# Data
data = iC.Data(configuration=configuration, interactions=interactions, diagonalization=diagonalization, bool_interactions=bool_interactions)
# Plotting Inputs
input_plotting = pP.InputPlotting(nb_excited=6, nb_plot=4, bool_spectrum=True, bool_holes=True, \
    bool_electrons=True, bool_determinants=True)

# iC.perform_configurations_computations(data)
# pP.generatePlots(data, inputPlotting)
# pP.savePhaseSpace(data, iVal=0, nbt=15, nbV=15)

#configuration_833 = bC.Configuration(bond_matrix=bond_matrix, nholes=2, nbeta=3, nbh=2, nbv=1, nblocs_side=2)
#data_833 = iC.load_data("tJHam-833Full.json")
#most_dominant_dets = iC.get_most_dominant_determinants(data_833.determinants, data_833.eigenstates.eigvects[:,0], 20)
#for coeff, det in most_dominant_dets.items():
#    print("-- {} :\n{}".format(coeff, bC.get_displayable_rectangular_lattice(configuration_833, det)))

### H8-3 -------------------------------------------------------------------------------------------------------
# Lattice
bond_matrix = lF.build_rectangular_bond_matrix(nbh=2, nbv=1, nblocs_side=2)
# Configuration
configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=3, nbeta=0, nbh=2, nbv=1, nblocs_side=2)
# diagonalization
diagonalization = dav.Diagonalization(nb_val=30, stp=1e4, tol=1e-6, dim_max=0)
# Interactions booleans
bool_interactions = iC.BoolInteractions(bt=True, bz=True, bij=True, bSD=True, btN=True, bV=True, bV_embedding=False, b_spin_multipicities=False)

# Data
data = iC.Data(configuration=configuration, interactions=interactions, diagonalization=diagonalization, bool_interactions=bool_interactions)
# Plotting Inputs
input_plotting = pP.InputPlotting(nb_excited=6, nb_plot=4, bool_spectrum=True, bool_holes=True, \
    bool_electrons=True, bool_determinants=True)

# iC.perform_configurations_computations(data)
# pP.generatePlots(data, inputPlotting)
# pP.savePhaseSpace(data, iVal=0, nbt=10, nbV=10)

### H9-1 -------------------------------------------------------------------------------------------------------
interactions = itr.Interactions(t=-0.55, J=0.123, Jhpar=0.156, Jhper=0.104, hSDper=-0.041, hSDpar=-0.08, \
    tNNN=0.112, tnNNN=-0.047, VNN=1.77, VNNNper=0.79)
# Lattice
bond_matrix = lF.build_rectangular_bond_matrix(nbh=1, nbv=1, nblocs_side=3)
# Configuration
configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=1, nbeta=0, nbh=1, nbv=1, nblocs_side=3)
# diagonalization
diagonalization = dav.Diagonalization(nb_val=25, stp=1e4, tol=1e-4, dim_max=0, initial_guess=None)
# Interactions booleans
bool_interactions = iC.BoolInteractions(bt=True, bz=True, bij=True, bSD=True, btN=True, bV=True, bV_embedding=False, b_spin_multipicities=True)

# Data
data = iC.Data(configuration=configuration, interactions=interactions, diagonalization=diagonalization, bool_interactions=bool_interactions)
# Plotting Inputs
input_plotting = pP.InputPlotting(nb_excited=25, nb_plot=4, bool_spectrum=True, bool_holes=True, \
    bool_electrons=True, bool_determinants=True, bool_neel_embedding=False, bool_dynamic_embedding=False, bool_neighbor_embedding=False)



#iC.perform_configurations_computations(data)
#pP.generate_plots(data, input_plotting)
# pP.savePhaseSpace(data, iVal=0, nbt=15, nbV=15) 

# H944
configuration_944 = bC.Configuration(bond_matrix=bond_matrix, nholes=1, nbeta=4, nbh=1, nbv=1, nblocs_side=3)
#interactions_944 = itr.Interactions(t=-0.55, J=0.123, Jhpar=0.123, Jhper=0.123, hSDper=-0.041, hSDpar=-0.08, \
#    tNNN=0.112, tnNNN=-0.047, VNN=1.77, VNNNper=0.79)
#data_944 = iC.Data(configuration=configuration_944, interactions=interactions_944, diagonalization=diagonalization, bool_interactions=bool_interactions)
#data_computation_944 = iC.perform_initial_computations(data_944)
#iC.save_data(data_computation_944)
#data_computation_944 = iC.load_data("tJHam-944Full.json")
#most_dominant_dets = iC.get_most_dominant_determinants(data_computation_944.determinants, data_computation_944.eigenstates.eigvects[:,0], 30)
#for coeff, det in most_dominant_dets.items():
#    print("-- {} :\n{}".format(coeff, bC.get_displayable_rectangular_lattice(configuration_944, det)))
#print("probabilities of finding a hole :\n", bC.get_probability_spin_rectangular_lattice(bC.Basis(data_computation_944.data.configuration, data_computation_944.determinants), data_computation_944.eigenstates.eigvects[:,[0]], 0))

# Embedding Neel H944
configuration_emb = bC.Configuration(bond_matrix=bond_matrix, nholes=1, nbeta=4, nbh=1, nbv=1, nblocs_side=3)
interactions_emb = itr.Interactions(t=-0.55, J=0.123, Jhpar=0.123, Jhper=0.123, hSDper=-0.041, hSDpar=-0.08, \
    tNNN=0.112, tnNNN=-0.047, VNN=1.77, VNNNper=0.79)
data_emb = iC.Data(configuration=configuration_emb, interactions=interactions_emb, diagonalization=diagonalization, bool_interactions=bool_interactions)
#data_computation_944_emb = emb.embedding_neel(data_emb, 1e-6, 100, 10)
#data_computation_944_emb = iC.load_data("Peripheral_embedding/tJHam-944Full.json")
#most_dominant_dets = iC.get_most_dominant_determinants(data_computation_944_emb.determinants, data_computation_944_emb.eigenstates.eigvects[:,0], 30)
#for coeff, det in most_dominant_dets.items():
#    print("-- {} :\n{}".format(coeff, bC.get_displayable_rectangular_lattice(configuration_emb, det)))
#print(bC.get_probability_spin_rectangular_lattice(bC.Basis(data_computation_944_emb.data.configuration, data_computation_944_emb.determinants), data_computation_944_emb.eigenstates.eigvects[:,[0]], 0))
#pP.plot_embedding_effect_on_determinants(data_computation_944, data_computation_944_emb, 100)

#most_dominant_dets = iC.get_most_dominant_determinants(H944_emb.determinants, H944_emb.eigenstates.eigvects[:,0], 30)
#for coeff, det in most_dominant_dets.items():
#    print("-- {} :\n{}".format(coeff, bC.get_displayable_rectangular_lattice(configuration_emb, det)))

# Embedding blocks H944
#data_computation_944_emb = emb.embedding_blocks(data_emb, self_consistency_tol=1e-6, max_iter=1e3, dynamic_embedding=False, neel_embedding=True, neighbor_embedding=False, threshold_mel=0, reverse_spins=True)
#emb.save_data_embedding_blocks(data_computation_944_emb, dynamic_embedding=False, neel_embedding=True, neighbor_embedding=False)
#pP.plot_embedding_effect_on_determinants(data_computation_944, data_computation_944_emb, 100)
#print(H944_emb.eigenstates.eigvects[:,[0]])
#H944 = iC.load_data("tJHam-944Full.json")
#H944_emb = iC.load_data("Static_embedding/tJHam-944Full.json")
#print("isolated gs energy : {} - embedded gs energy : {}".format(H944.eigenstates.eigvals[0], H944_emb.eigenstates.eigvals[0]))
#most_dominant_dets = iC.get_most_dominant_determinants(H944_emb.determinants, H944_emb.eigenstates.eigvects[:,0], 30)
#for coeff, det in most_dominant_dets.items():
#    print("-- {} :\n{}".format(coeff, bC.get_displayable_rectangular_lattice(configuration_emb, det)))


# Embedding
configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=1, nbeta=4, nbh=1, nbv=1, nblocs_side=3)
data = iC.Data(configuration=configuration, interactions=interactions, diagonalization=diagonalization, bool_interactions=bool_interactions)

# Embedding Néel
#emb.embedding_neel(data, self_consistency_tol=1e-3, max_iter=300)
#pP.generate_plots_configuration(data, input_plotting)

# Embedding Blocks
#emb.embedding_blocks(data, self_consistency_tol=1e-3, max_iter=300, dynamic_embedding=True, neel_embedding=True, neighbor_embedding=True)
#pP.generate_plots_configuration(data, input_plotting)
#data_944 = iC.load_data("tJHam-944Full.json")
#data_944_blocks = iC.load_data("Embedding_blocks/tJHam-944Full.json")

#print("Eigenvalues isolated block : ", data_944.eigenstates.eigvals)
#print("Eigenvalues static embedding : ", data_944_blocks.eigenstates.eigvals)

#ground_state_iso = data_944.eigenstates.eigvects[:,0]
#ground_state_emb = data_944_blocks.eigenstates.eigvects[:,0]
#for name, ground_state in (("isolated : ", ground_state_iso), ("embedding : ", ground_state_emb)):
#    print(name)
#    ground_state_list = ground_state.tolist()
#    amplitudes = abs(ground_state).tolist()
#    max_amplitudes = list(reversed(sorted(amplitudes[:])))[:10]
#    indices = [amplitudes.index(max_amp) for max_amp in max_amplitudes]
#    for ind, index in enumerate(indices):
#        coeff = ground_state_list[index]
#        det_max = bC.get_hash_basis(1, 4, 4, index)
#        print("- {} : {} ------------------------\n".format(ind, coeff), bC.get_displayable_rectangular_lattice(configuration, det_max))
#    print("---------------------------------------------------\n---------------------------------------------------\n\n")

# data_9_neighbor_embedding = iC.load_data("Embedding_blocks/tJHam-dyn-neel-neigh-944Full.json")
# basis_9 = bC.Basis(data_9_neighbor_embedding.data.configuration, data_9_neighbor_embedding.determinants)
# ground_state_emb_block = data_9_neighbor_embedding.eigenstates.eigvects[:,[0]]

# data_18 = iC.load_data('tJHam-1888SDVJttN.json')
# basis_18 = bC.Basis(data_18.data.configuration, data_18.determinants)
# ground_state_18 = data_18.eigenstates.eigvects[:,[0]]

# vect_9p9m = np.zeros(ground_state_18.shape)
# vect_9m9p = np.zeros(ground_state_18.shape)
# for ind_1, det_1 in enumerate(basis_9.determinants):
#     for ind_2, det_2 in enumerate(basis_9.determinants):
#         configuration_flip, det_2_flip = bC.reverse_spins(configuration, det_2)
#         det_18_pm = bC.extend_determinant(configuration, configuration_flip, det_1, det_2_flip)
#         det_18_mp = bC.extend_determinant(configuration_flip, configuration, det_2_flip, det_1)
#         vect_9p9m[bC.get_index_in_basis(basis_18.configuration, det_18_pm)] = ground_state_emb_block[ind_1] * ground_state_emb_block[ind_2]
#         vect_9m9p[bC.get_index_in_basis(basis_18.configuration, det_18_mp)] = ground_state_emb_block[ind_1] * ground_state_emb_block[ind_2]

# vect_99 = (1/mp.sqrt(2)) * (vect_9m9p + vect_9p9m)

# print("Embedding fidelity = ", np.dot(vect_99.T, ground_state_18) / (np.linalg.norm(vect_99) * np.linalg.norm(ground_state_18)))
# print("Norms : ", np.linalg.norm(vect_99), " - ", np.linalg.norm(ground_state_18))


### H9-2 -------------------------------------------------------------------------------------------------------
# Lattice
bond_matrix = lF.build_rectangular_bond_matrix(nbh=1, nbv=1, nblocs_side=3)
# Configuration
configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=2, nbeta=0, nbh=1, nbv=1, nblocs_side=3)
# diagonalization
diagonalization = dav.Diagonalization(nb_val=15, stp=2e4, tol=1e-6, dim_max=0)
# Interactions booleans
bool_interactions = iC.BoolInteractions(bt=True, bz=True, bij=True, bSD=True, btN=True, bV=True, bV_embedding=False, b_spin_multipicities=False)

# Data
data = iC.Data(configuration=configuration, interactions=interactions, diagonalization=diagonalization, bool_interactions=bool_interactions)
# Plotting Inputs
input_plotting = pP.InputPlotting(nb_excited=6, nb_plot=4, bool_spectrum=True, bool_holes=True, \
    bool_electrons=True, bool_determinants=True)

# iC.perform_configurations_computations(data)
# pP.generatePlots(data, inputPlotting)
# pP.savePhaseSpace(data, iVal=0, nbt=10, nbV=10)

# Embedding
configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=2, nbeta=3, nbh=1, nbv=1, nblocs_side=3)
data = iC.Data(configuration=configuration, interactions=interactions, diagonalization=diagonalization, bool_interactions=bool_interactions)

# emb.static_embedding_neel(data, self_consistency_tol=1e-2, max_iter=5000)

### H9-3 -------------------------------------------------------------------------------------------------------
# Lattice
bond_matrix = lF.build_rectangular_bond_matrix(nbh=1, nbv=1, nblocs_side=3)
# Configuration
configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=3, nbeta=0, nbh=1, nbv=1, nblocs_side=3)
# diagonalization
diagonalization = dav.Diagonalization(nb_val=15, stp=3e4, tol=1e-6, dim_max=0)
# Interactions booleans
bool_interactions = iC.BoolInteractions(bt=True, bz=True, bij=True, bSD=True, btN=True, bV=True, bV_embedding=False, b_spin_multipicities=False)

# Data
data = iC.Data(configuration=configuration, interactions=interactions, diagonalization=diagonalization, bool_interactions=bool_interactions)
# Plotting Inputs
input_plotting = pP.InputPlotting(nb_excited=6, nb_plot=4, bool_spectrum=True, bool_holes=True, \
    bool_electrons=True, bool_determinants=True)

# iC.perform_configurations_computations(data)
# pP.generatePlots(data, inputPlotting)
# pP.savePhaseSpace(data, iVal=0, nbt=10, nbV=10)

# Embedding
configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=3, nbeta=3, nbh=1, nbv=1, nblocs_side=3)
data = iC.Data(configuration=configuration, interactions=interactions, diagonalization=diagonalization, bool_interactions=bool_interactions)

# emb.static_embedding_neel(data, self_consistency_tol=1e-2, max_iter=5000)

### H9-0 -------------------------------------------------------------------------------------------------------
# Heisenberg Model

# Lattice
bond_matrix = lF.build_rectangular_bond_matrix(nbh=1, nbv=1, nblocs_side=3)
# Configuration
configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=0, nbeta=0, nbh=1, nbv=1, nblocs_side=3)
# diagonalization
diagonalization = dav.Diagonalization(nb_val=30, stp=2e4, tol=1e-6, dim_max=0)
# Interactions booleans
bool_interactions = iC.BoolInteractions(bt=False, bz=True, bij=True, bSD=False, btN=False, bV=False, bV_embedding=False, b_spin_multipicities=False)

# Data
data = iC.Data(configuration=configuration, interactions=interactions, diagonalization=diagonalization, bool_interactions=bool_interactions)
# Plotting Inputs
input_plotting = pP.InputPlotting(nb_excited=15, nb_plot=4, bool_spectrum=True, bool_holes=True, \
    bool_electrons=True, bool_determinants=True)

#iC.perform_configurations_computations(data)
# pP.generatePlots(data, inputPlotting)
# pP.savePhaseSpace(data, iVal=0, nbt=15, nbV=15)

### H16-4 -------------------------------------------------------------------------------------------------------
# Lattice
bond_matrix = lF.build_rectangular_bond_matrix(nbh=2, nbv=2, nblocs_side=2)
# Configuration
configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=4, nbeta=6, nbh=2, nbv=2, nblocs_side=2)
# Initial guess
initial_guesses_det = [\
    [0, 2, 12, 14, 1, 3, 4, 6, 9, 11, 5, 7, 8, 10, 13, 15], \
    [0, 2, 9, 11, 1, 3, 4, 6, 12, 14, 5, 7, 8, 10, 13, 15], \
    [5, 7, 12, 14, 1, 3, 4, 6, 9, 11, 0, 2, 8, 10, 13, 15], \
    [1, 3, 13, 15, 4, 6, 9, 11, 12, 14, 0, 2, 5, 7, 8, 10], \
    [1, 3, 8, 10, 4, 6, 9, 11, 12, 14, 0, 2, 5, 7, 13, 15], \
    [4, 6, 13, 15, 1, 3, 9, 11, 12, 14, 0, 2, 5, 7, 8, 10],]
initial_guess = np.zeros(shape=(configuration.nb_conf, 6))
for ind, det in enumerate(initial_guesses_det):
    initial_guess[bC.compute_hash_table(configuration, det), ind] = 1
# diagonalization
diagonalization = dav.Diagonalization(nb_val=6, stp=10000, tol=1e-4, dim_max=500)
# Interactions booleans
bool_interactions = iC.BoolInteractions(bt=True, bz=True, bij=True, bSD=True, btN=True, bV=False, bV_embedding=False, b_spin_multipicities=False)

# Data
data = iC.Data(configuration=configuration, interactions=interactions, diagonalization=diagonalization, bool_interactions=bool_interactions)

#data_computations_16_4 = iC.perform_initial_computations(data)
#iC.save_data(data_computations_16_4)
#most_dominant_dets = iC.get_most_dominant_determinants(data_computations_16_4.determinants, data_computations_16_4.eigenstates.eigvects[:,0], 100)
#for coeff, det in most_dominant_dets.items():
#    print("-- {} :\n{}".format(coeff, bC.get_displayable_rectangular_lattice(configuration, det)))


### H18-2 -------------------------------------------------------------------------------------------------------
# Lattice
bond_matrix = lF.build_rectangular_bond_matrix(nbh=2, nbv=1, nblocs_side=3)
# Configuration
configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=2, nbeta=8, nbh=2, nbv=1, nblocs_side=3)
# Initial guess
initial_guesses_det = [\
    [4, 13, 1, 3, 5, 7, 9, 11, 15, 17, 0, 2, 6, 8, 10, 12, 14, 16], \
    [4, 13, 0, 2, 6, 8, 10, 12, 14, 16, 1, 3, 5, 7, 9, 11, 15, 17], \
    [8, 9, 1, 3, 5, 7, 11, 13, 15, 17, 0, 2, 4, 6, 10, 12, 14, 16], \
    [8, 9, 0, 2, 4, 6, 10, 12, 14, 16, 1, 3, 5, 7, 11, 13, 15, 17], \
    [6, 11, 0, 2, 4, 8, 10, 12, 14, 16, 1, 3, 5, 7, 9, 13, 15, 17], \
    [6, 11, 1, 3, 5, 7, 9, 13, 15, 17, 0, 2, 4, 8, 10, 12, 14, 16]]
initial_guess = np.zeros(shape=(configuration.nb_conf, 6))
for ind, det in enumerate(initial_guesses_det):
    initial_guess[bC.compute_hash_table(configuration, det), ind] = 1
# diagonalization
diagonalization = dav.Diagonalization(nb_val=4, stp=10000, tol=1e-4, dim_max=1000)
# Interactions booleans
bool_interactions = iC.BoolInteractions(bt=True, bz=True, bij=True, bSD=True, btN=True, bV=True, bV_embedding=False, b_spin_multipicities=False)

# Data
data = iC.Data(configuration=configuration, interactions=interactions, diagonalization=diagonalization, bool_interactions=bool_interactions)
# Plotting Inputs
input_plotting = pP.InputPlotting(nb_excited=4, nb_plot=4, bool_spectrum=False, bool_holes=True, \
    bool_electrons=True, bool_determinants=False)

#data_computations_18_2 = iC.perform_initial_computations(data)
#print("Lowests eigenvalues : ", data_computations_18_2.eigenstates.eigvals)
#iC.save_data(data_computations_18_2)


data_computations_18_2 = iC.load_data('tJHam-1888Full.json')
#most_dominant_dets = iC.get_most_dominant_determinants(data_computations_18_2.determinants, data_computations_18_2.eigenstates.eigvects[:,0], 100)
#for coeff, det in most_dominant_dets.items():
#    print("-- {} :\n{}".format(coeff, bC.get_displayable_rectangular_lattice(configuration, det)))
#basis_18 = bC.Basis(data_computations_18_2.data.configuration, data_computations_18_2.determinants)
#print("\n\n")
#print("Probability of finding hole on each site")
#print("----------------------------------------\n----------------------------------------\n")
#print("Ground state :")
#print(bC.get_probability_spin_rectangular_lattice(basis_18, data_computations_18_2.eigenstates.eigvects[:,[0]], 0))
#print("1st excited state :")
#print(bC.get_probability_spin_rectangular_lattice(basis_18, data_computations_18_2.eigenstates.eigvects[:,[1]], 0))
#print("2nd excited state :")
#print(bC.get_probability_spin_rectangular_lattice(basis_18, data_computations_18_2.eigenstates.eigvects[:,[2]], 0))
#print("3rd excited state :")
#print(bC.get_probability_spin_rectangular_lattice(basis_18, data_computations_18_2.eigenstates.eigvects[:,[3]], 0))



#print("Lowests eigenvalues : \n", data_computations_18_2.eigenstates.eigvals, "\n\n")
#ground_state = data_computations_18_2.eigenstates.eigvects[:,1]
#for ind in range(10):
#    amplitude = max(abs(ground_state).tolist())
#    index_max = abs(ground_state).tolist().index(amplitude)
#    coeff = ground_state.tolist().pop(index_max)
#    det_max = bC.get_hash_basis(2, 8, 8, index_max) 
#    print("- {} : {} ------------------------\n".format(ind, coeff), bC.get_displayable_rectangular_lattice(configuration, det_max))

# Embedding Neel
H1888 = emb.embedding_neel(data, 1e-6, 1000)
print("isolated gs energy : {} - embedded gs energy : {}".format(H1888.eigenstates.eigvals[0], H1888.eigenstates.eigvals[0]))
most_dominant_dets = iC.get_most_dominant_determinants(H1888.determinants, H1888.eigenstates.eigvects[:,0], 50)
for coeff, det in most_dominant_dets.items():
    print("-- {} :\n{}".format(coeff, bC.get_displayable_rectangular_lattice(configuration, det)))

#pP.generate_plots_configuration(data, input_plotting)
# pP.savePhaseSpace(data, iVal=0, nbt=10, nbV=10)

# Embedding
# emb.static_embedding_neel(data, self_consistency_tol=1e-2, max_iter=5000)

### H16-4 -------------------------------------------------------------------------------------------------------

# Lattice
bond_matrix = lF.build_rectangular_bond_matrix(nbh=1, nbv=2, nblocs_side=3)
# Configuration
configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=2, nbeta=8, nbh=1, nbv=2, nblocs_side=3)
# diagonalization
diagonalization = dav.Diagonalization(nb_val=6, stp=5e3, tol=1e-2, dim_max=0)
# Interactions booleans
bool_interactions = iC.BoolInteractions(bt=True, bz=False, bij=False, bSD=False, btN=False, bV=False, bV_embedding=False)

# Data
data = iC.Data(configuration=configuration, interactions=interactions, diagonalization=diagonalization, bool_interactions=bool_interactions)
# Plotting Inputs
input_plotting = pP.InputPlotting(nb_excited=6, nb_plot=4, bool_spectrum=False, bool_holes=False, \
    bool_electrons=False, bool_determinants=False, bool_neel_embedding=True)

# iC.save_data(iC.perform_initial_computations(data))
# pP.generate_plots(data, input_plotting)
# pP.save_phase_space(data, iVal=0, nbt=25, nbV=25)
