import latticeFunctions as lF
import baseConstruction as bC
import interactions as itr
import hamiltonianConstruction as hC
import davidson as dav
import initialComputations as iC

import numpy as np
import numpy.linalg as lin
import math as mp

import copy
import os
import json
from tqdm import tqdm

import matplotlib.pyplot as plt

def embedding_neel(data, self_consistency_tol, max_iter, nb_val=10): # Fix the generation of determinants
    """
    Embeds the cluster in a Néel lattice.
    Args:
        data (class initialComputations.Data): the definition of the computations
        self_consistency_tol (float): tolerance for the self-consistency convergence
        max_iter (int): maximum number of self-consistency iterations
        nb_val (int): number of eigenstates to compute at each iteration, limits the dimension of the Krylov space and thus affects the precision of the diagonalizer
    Returns:
        data_computations (class initialComputations.DataComputations): the results of the static embedding in the Néel lattice
    """
    print("[emb] Static embedding of the lattice within the self-coherent field of an outer Néel lattice.")
    # Interactions
    interactions = data.interactions
    J = interactions.J
    VNN = interactions.VNN
    VNNNper = interactions.VNNNper
    # Diagonalization
    diagonalization = data.diagonalization
    stp = diagonalization.stp
    tol = diagonalization.tol
    dim_max = diagonalization.dim_max
    # Bool Interactions
    bool_interactions = data.bool_interactions
    bt = bool_interactions.bt
    bz = bool_interactions.bz
    bij = bool_interactions.bij
    bSD = bool_interactions.bSD
    btN = bool_interactions.btN
    bV = bool_interactions.bV

    ### Block ###

    # Configuration Block
    configuration_block = data.configuration
    nsites_block = configuration_block.nsites
    nalpha_block = configuration_block.nalpha
    nbeta_block = configuration_block.nbeta
    nholes_block = configuration_block.nholes
    nb_conf_block = configuration_block.nb_conf
    nbh = configuration_block.nbh
    nbv = configuration_block.nbv
    nblocs_side = configuration_block.nblocs_side
    bond_matrix_block = configuration_block.bond_matrix


    # File Name
    file_name = "tJHam-{}{}{}".format(nsites_block, nalpha_block, nbeta_block)
    if bt and bz and bij and bSD and btN and bV:
        file_name += "Full.json"
    else:
        extensions = {"bt":"t", "bij":"J", "bSD":"SD", "btN":"tN", "bV":"V"}
        for name in dir(bool_interactions):
            if name in extensions:
                file_name += extensions[name]
        file_name += ".json"

    # Loading data
    data_computation = iC.load_data(file_name)
    # Basis
    determinants_block = data_computation.determinants
    basis_block = bC.Basis(configuration_block, determinants_block)
    # Hamiltonian
    ham_block = data_computation.hamiltonian
    # Eigenstates
    eigenstates_block = data_computation.eigenstates
    ground_state_block = eigenstates_block.eigvects[:,[0]]


    ### Néel lattice ###
    # Definition of the lattice
    bond_matrix_rest = lF.build_bond_matrix_neel_outer_lattice(nbh, nbv, nblocs_side)
    nsites_rest = 2 * (nbh + nbv) * nblocs_side

    # Definition of the initial Néel spin wave on peripheral sites
    beta_neel = [site for site in range(nsites_rest) \
        if (site + 1 + (site >= nbh * nblocs_side) + (site >= (nbh + nbv) * nblocs_side) + (site >= (2 * nbh + nbv) * nblocs_side)) % 2 == 1]
    det_neel = beta_neel + list(set(range(nsites_rest)).difference(set(beta_neel)))
    nholes_rest = 0
    nbeta_rest = len(beta_neel)

    configuration_rest = bC.Configuration(bond_matrix_rest, nholes_rest, nbeta_rest, nbh=nbh, nbv=nbv, nblocs_side=nblocs_side)
    nb_conf_rest = configuration_rest.nb_conf

    basis_rest = bC.construct_basis(configuration_rest)
    

    ### Embedding ###
    bond_matrix_embedding = lF.build_bond_matrix_neel_embedding(nbh, nbv, nblocs_side)

    frontier_A = [site for site in range(nsites_block) if len(lF.get_neighbors(site, bond_matrix_block)) < len(lF.get_neighbors(site, bond_matrix_embedding))]

    sites_restriction_embedding = list(range(nsites_block))
    for site in range(nsites_block):
        sites_restriction_embedding += lF.get_neighbors(site, bond_matrix_embedding)
    sites_restriction_embedding = list(set(sites_restriction_embedding))

    configuration_embedding = bC.Configuration(bond_matrix_embedding, nholes_block + nholes_rest, nbeta_block + nbeta_rest, sites_restriction_embedding, 'peripheral', nbh, nbv, nblocs_side)

    # Computations of probabilities of presence of each particle on the surrounding sites
    vect_neel = np.zeros(shape=(nb_conf_rest, 1))
    vect_neel[bC.get_index_in_basis(configuration_rest, det_neel)] = 1

    probabilities_spins_R = bC.get_spins_probabilitites_lattice(basis_rest, vect_neel)

    ### Self-coherent iterations ###

    ind_iter = 0
    ground_state_temp = np.copy(ground_state_block)

    with tqdm(total=max_iter) as pbar :
        while ind_iter < max_iter:

            #probabilities_spins_lattice = bC.get_spins_probabilitites_lattice(basis_block, ground_state_temp)
            #print("[emb] probabilities_spins_lattice = ", probabilities_spins_lattice)
            #print("[emb] probabilities_spins_R = ", probabilities_spins_R)
            #for site, (paL, pbL, phL) in enumerate(zip(probabilities_spins_lattice[1], probabilities_spins_lattice[-1], probabilities_spins_lattice[0])):
                #print("[emb] site     : {} - (a:{})    (b:{})    (h:{})".format(site, round(paL, 4), round(pbL, 4), round(phL, 4)))
                #neighbors = lF.get_neighbors(site, bond_matrix_embedding, sites_restriction=[site_rest + nsites_block for site_rest in range(nsites_rest)])
                #for neighbor in neighbors:
                    #neighbor = neighbor - nsites_block
                    #print("[emb] neighbor : {} - (a:{})    (b:{})    (h:{})".format(neighbor, round(probabilities_spins_R[1][neighbor], 4), round(probabilities_spins_R[-1][neighbor], 4), round(probabilities_spins_R[0][neighbor], 4)))
                #print("")

            ham_block_embedded = ham_block.tocoo(copy=True)

            for ind_A, _ in enumerate(determinants_block):

                vect_A = np.zeros(shape=(nb_conf_block, 1))
                vect_A[ind_A, 0] = 1
                probabilities_spins_A = bC.get_spins_probabilitites_lattice(basis_block, vect_A)
                #print("--[emb] probabilities_spins_A = ", probabilities_spins_A)

                intr = 0

                for site_A in frontier_A:
                    proba_alpha_A = probabilities_spins_A[1][site_A]
                    proba_beta_A = probabilities_spins_A[-1][site_A]
                    proba_holes_A = probabilities_spins_A[0][site_A]

                    foreign_neighbors = lF.get_neighbors(site_A, bond_matrix_embedding, sites_restriction=range(nsites_block, nsites_block + nsites_rest))
                    
                    for site_R in foreign_neighbors:
                        site_R = site_R - nsites_block

                        proba_alpha_R = probabilities_spins_R[1][site_R]
                        proba_beta_R = probabilities_spins_R[-1][site_R]
                        proba_holes_R = probabilities_spins_R[0][site_R]

                        intr += -(J/2) * (proba_alpha_A * proba_beta_R + proba_beta_A * proba_alpha_R) + VNN * proba_holes_A * proba_holes_R
                        #print("--[emb] {} - A : {} - R : {} || intr += {} * ({} * {} + {} * {}) + {} * {} * {} = {}".format(ind_A, site_A, site_R + nsites_block, -J/2, proba_alpha_A, proba_beta_R, proba_beta_A, proba_alpha_R, VNN, proba_holes_A, proba_holes_R, -(J/2) * (proba_alpha_A * proba_beta_R + proba_beta_A * proba_alpha_R) + VNN * proba_holes_A * proba_holes_R))
                        
                    foreign_diagonal_second_neighbors = list(set(range(nsites_block, nsites_block + nsites_rest)).intersection(
                        set(lF.get_diagonal_second_neighbors(configuration_embedding, site_A))
                        ))
                    #if ind_A in (0, 446): #####
                    #    print("[emb]\n[emb] site_A = ", site_A)
                    #    print("[emb] diagonal_second_neighbors = ", lF.get_diagonal_second_neighbors(configuration_embedding, site_A))
                    #    print("[emb] foreign_diagonal_second_neighbors = ", foreign_diagonal_second_neighbors)
                    #    print("[emb]")
                    for site_R in foreign_diagonal_second_neighbors:
                        site_R = site_R - nsites_block

                        proba_holes_R = probabilities_spins_R[0][site_R]

                        intr += VNNNper * proba_holes_A * proba_holes_R
                        #print("--[emb] {} - A : {} - R : {} || intr += {} * {} * {} = {}".format(ind_A, site_A, site_R, VNNNper, proba_holes_A, proba_holes_R, VNNNper * proba_holes_A * proba_holes_R))


                #print("[emb] {} - ham[{}, {}] = {} || intr = {}".format(ind_A, ind_A, ind_A, ham_block[ind_A, ind_A], intr))
                ham_block_embedded = hC.add_ham_coo(ham=ham_block_embedded, \
                    elt=intr, \
                    ind_row=ind_A, ind_col=ind_A)
            
            ham_block_embedded = ham_block_embedded.tocsr(copy=False)


            while True:
                eigenstates_embedded = dav.modified_davidson_algorithm(
                    A=ham_block_embedded, nb_val=nb_val, initial_guess=None, stp=stp, tol=tol, dim_max=dim_max, verbose=False
                    )
                ground_state_embedded = eigenstates_embedded.eigvects[:,[0]]
                if eigenstates_embedded.eigvals:
                    print("[emb] Probability of finding a hole on each site of the lattice :\n", bC.get_probability_spin_rectangular_lattice(basis_block, ground_state_embedded, 0))
                    break
            
            #print("[emb] diagonalization precision = ", ground_state_embedded - np.linalg.eigh(ham_block_embedded.toarray())[1][:,[0]])

            # Attribution of the probabilities of presence of each particle on the current ground state on the sites of the peripheral lattice
            probabilities_spins_lattice = bC.get_spins_probabilitites_lattice(basis_block, ground_state_embedded)
            probabilities_spins_R = {1 : [0 for _ in range(nsites_rest)], -1 : [0 for _ in range(nsites_rest)], 0 : [0 for _ in range(nsites_rest)]}
            for site_A in frontier_A:
                #print("[emb] site_A = ", site_A)
                probability_alpha_A = probabilities_spins_lattice[1][site_A]
                probability_beta_A = probabilities_spins_lattice[-1][site_A]
                probability_holes_A = probabilities_spins_lattice[0][site_A]
                #print("[emb] (pa, pb, ph) = ({}, {}, {})".format(probability_alpha_A, probability_beta_A, probability_holes_A))

                foreign_neighbors = lF.get_neighbors(site_A, bond_matrix_embedding, sites_restriction=range(nsites_block, nsites_block + nsites_rest))
                for site_R in foreign_neighbors:
                    #print("--[emb] site_R = ", site_R)
                    site_R = site_R - nsites_block
                    probabilities_spins_R[1][site_R] = probability_beta_A
                    probabilities_spins_R[-1][site_R] = probability_alpha_A
                    probabilities_spins_R[0][site_R] = probability_holes_A
                    #print("--[emb] (pa, pb, ph) = ({}, {}, {})".format(probabilities_spins_R[1][site_R], probabilities_spins_R[-1][site_R], probabilities_spins_R[0][site_R]))

            #for site, (paL, pbL, phL) in enumerate(zip(probabilities_spins_lattice[1], probabilities_spins_lattice[-1], probabilities_spins_lattice[0])):
                #print("[emb] site : {} - a : {} b : {} h : {}".format(site, paL, pbL, phL))
                #neighbors = lF.get_neighbors(site, bond_matrix_embedding, sites_restriction=[s + nsites_block for s in range(nsites_rest)])
                #for neighbor in neighbors:
                    #neighbor = neighbor - nsites_block
                    #print("[emb] neighbor : {} - a : {} b : {} h : {}".format(neighbor, probabilities_spins_R[1][neighbor], probabilities_spins_R[-1][neighbor], probabilities_spins_R[0][neighbor]))

            #####
            #printed_difference = [] 
            #for ind, diag in enumerate(ham_block_embedded.diagonal()):
                #diff = diag - ham_block.diagonal()[ind]
                #if abs(diff) > 1e-3:
                    #printed_difference.append(diff)
                #else:
                #    printed_difference.append(0)
            #print("[emb] diag(ham_block__embedded - ham_block) =\n", printed_difference) 
            #print("[emb] ground state energy = ", eigenstates_embedded.eigvals[0])
            #gs = list(eigenstates_embedded.eigvects[:,[0]])
            #for ind, coeff in enumerate(gs):
            #    print("-- {} :\n{}".format(coeff, bC.get_displayable_rectangular_lattice(configuration_block, determinants_block[ind])))
            #####

            if lin.norm(ground_state_embedded - ground_state_temp) / lin.norm(ground_state_temp) < self_consistency_tol:
                data_computation = iC.DataComputations(data, determinants_block, ham_block, eigenstates_embedded)
                print("[emb] ground state energy = {} | embedded ground state energy = {}".format(\
                    eigenstates_block.eigvals[0], eigenstates_embedded.eigvals[0]))
                save_data_peripheral_embedding(data_computation)
                return data_computation
            ind_iter += 1
            pbar.update(1)
            ground_state_temp = ground_state_embedded
    print("Maximum number of iterations exceeded")
    print("[emb] ground state energy = {} | embedded ground state energy = {}".format(\
                eigenstates_block.eigvals[0], eigenstates_embedded.eigvals[0]))
    data_computation = iC.DataComputations(data, determinants_block, ham_block, eigenstates_embedded)
    save_data_peripheral_embedding(data_computation)
    return data_computation    

def save_data_peripheral_embedding(data_computations):
    """
    Saves the data from the computations in a JSON format
    Args:
        data_computations (class DataComputation): the results of the computations
    """

    # Configuration
    configuration = data_computations.data.configuration
    nsites = configuration.nsites
    nalpha = configuration.nalpha
    nbeta = configuration.nbeta

    # BoolInteractions
    bool_interactions = data_computations.data.bool_interactions
    bt = bool_interactions.bt
    bz = bool_interactions.bz
    bij = bool_interactions.bij
    bSD = bool_interactions.bSD
    btN = bool_interactions.btN
    bV = bool_interactions.bV

    # File Name
    file_name = "tJHam-{}{}{}".format(nsites, nalpha, nbeta)
    if bt and bz and bij and bSD and btN and bV:
        file_name += "Full.json"
    else:
        extensions = {"bt":"t", "bij":"J", "bSD":"SD", "btN":"tN", "bV":"V"}
        for name in dir(bool_interactions):
            if name in extensions:
                file_name += extensions[name]
        file_name += ".json"
 
    # Saving results
    try :
        with open("./initialComputations/Peripheral_embedding/{}".format(file_name), "w") as savefile:
            savefile.write(json.dumps({
                "configuration" : configuration.__to_json__(), 
                "interactions" : data_computations.data.interactions.__dict__, 
                "diagonalization" : data_computations.data.diagonalization.__dict__, 
                "bool_interactions" : data_computations.data.bool_interactions.__dict__,
                "determinants" : data_computations.determinants, 
                "hamiltonian" : {
                    "data" : data_computations.hamiltonian.data.tolist(),
                    "indices" : data_computations.hamiltonian.indices.tolist(),
                    "indptr" : data_computations.hamiltonian.indptr.tolist()
                }, 
                "eigenstates" : {
                    "eigvals" : data_computations.eigenstates.eigvals,
                    "spins" : data_computations.eigenstates.spins,
                    "eigvects" : data_computations.eigenstates.eigvects.tolist()
                    }
            })) 
    except OSError:
        os.mkdir("./initialComputations/Peripheral_embedding")
        with open("./initialComputations/Peripheral_embedding/{}".format(file_name), "w") as savefile:
            savefile.write(json.dumps({
                "configuration" : configuration.__to_json__(), 
                "interactions" : data_computations.data.interactions.__dict__, 
                "diagonalization" : data_computations.data.diagonalization.__dict__, 
                "bool_interactions" : data_computations.data.bool_interactions.__dict__,
                "determinants" : data_computations.determinants, 
                "hamiltonian" : {
                    "data" : data_computations.hamiltonian.data.tolist(),
                    "indices" : data_computations.hamiltonian.indices.tolist(),
                    "indptr" : data_computations.hamiltonian.indptr.tolist()
                }, 
                "eigenstates" : {
                    "eigvals" : data_computations.eigenstates.eigvals,
                    "spins" : data_computations.eigenstates.spins,
                    "eigvects" : data_computations.eigenstates.eigvects.tolist()
                    }
            })) 


# Embedding in neighboring blocks

def embedding_blocks(data, self_consistency_tol, max_iter, dynamic_embedding=False, neel_embedding=False, neighbor_embedding=False, threshold_mel=1e-6, reverse_spins=True):
    """
    Embeds the cluster within the influence of neighboring blocks, with static and dynamic interactions.
    Args:
        data (class initialComputations.Data): the definition of the computations
        self_consistency_tol (float): tolerance for the self-consistency convergence
        max_iter (int): maximum number of self-consistency iterations
        dynamical_embedding (bool): True to perform dynamical embedding
        neel_embedding (bool): True to perform a Néel embedding beforehand
        neighbor_embedding (bool): True to perform an embedding with a single neighboring block
        threshold_mel (float): minimal non-zero value non neglected when building the Hamiltonian
    Returns:
        data_computations (class initialComputations.DataComputations): the results of the embedding
    """
    print("-----------------------------------------------------------------------------------------------------------------")
    if dynamic_embedding:
        print("[emb] Static and dynamic embedding of the lattice in the self-coherent field generated by neighboring blocks.")
    else:
        print("[emb] Static embedding of the lattice within the self-coherent field generated by neighboring blocks.")
    print("-----------------------------------------------------------------------------------------------------------------")
    # Interactions
    interactions = data.interactions
    J = interactions.J
    VNN = interactions.VNN
    VNNNper = interactions.VNNNper
    # Diagonalization
    diagonalization = data.diagonalization
    stp = diagonalization.stp
    tol = diagonalization.tol
    dim_max = diagonalization.dim_max
    # Bool Interactions
    bool_interactions = data.bool_interactions
    bt = bool_interactions.bt
    bz = bool_interactions.bz
    bij = bool_interactions.bij
    bSD = bool_interactions.bSD
    btN = bool_interactions.btN
    bV = bool_interactions.bV

    ### Block ###

    # Configuration Block
    configuration_block = data.configuration
    nsites_block = configuration_block.nsites
    nholes_block = configuration_block.nholes
    nalpha_block = configuration_block.nalpha
    nbeta_block = configuration_block.nbeta
    nbh = configuration_block.nbh
    nbv = configuration_block.nbv
    nblocs_side = configuration_block.nblocs_side
    nb_conf_block = configuration_block.nb_conf
    bond_matrix_block = configuration_block.bond_matrix


    # File Name
    file_name = "tJHam-{}{}{}".format(nsites_block, nalpha_block, nbeta_block)
    if bt and bz and bij and bSD and btN and bV:
        file_name += "Full.json"
    else:
        extensions = {"bt":"t", "bij":"J", "bSD":"SD", "btN":"tN", "bV":"V"}
        for name in dir(bool_interactions):
            if name in extensions:
                file_name += extensions[name]
        file_name += ".json"
    if neel_embedding:
        file_name = "Peripheral_embedding/" + file_name

    # Loading data
    data_computation = iC.load_data(file_name)
    # Basis
    determinants_block = data_computation.determinants
    basis_block = bC.Basis(configuration_block, determinants_block)
    # Hamiltonian
    ham_block = data_computation.hamiltonian
    # Eigenstates
    eigenstates_block = data_computation.eigenstates
    ground_state_block = eigenstates_block.eigvects[:,[0]]
    gs_energy_block = eigenstates_block.eigvals[0]


    ### Neighboring block ###
    if reverse_spins:
        basis_block_flip = bC.reverse_spins_basis(basis_block)
        configuration_neighbor = basis_block_flip.configuration
        determinants_neighbor = basis_block_flip.determinants
    else:
        configuration_neighbor = copy.deepcopy(configuration_block)
        determinants_neighbor = copy.deepcopy(determinants_block)
    nsites_neighbor = configuration_neighbor.nsites
    nholes_neighbor = configuration_neighbor.nholes
    nbeta_neighbor = configuration_neighbor.nbeta
        

    ### Embedding ###
    bond_matrix_block_embedding = lF.build_bond_matrix_block_embedding(nbh, nbv, nblocs_side)

    frontier = []
    for site in range(nsites_block):
        if len(lF.get_neighbors(site, bond_matrix_block)) < len(lF.get_neighbors(site, bond_matrix_block_embedding)):
            frontier += lF.get_neighbors(site, bond_matrix_block_embedding)
    for site in range(nsites_neighbor):
        if len(lF.get_neighbors(site, bond_matrix_block)) < len(lF.get_neighbors(site + nsites_block, bond_matrix_block_embedding)):
            frontier += lF.get_neighbors(site + nsites_block, bond_matrix_block_embedding)
    frontier = list(set(frontier))

    frontier_A = [site for site in frontier if site < nsites_block]
    frontier_B = [site for site in frontier if site >= nsites_block]
    sites_restriction_embedding = (frontier_A, frontier_B)

    configuration_embedding = bC.Configuration(
        bond_matrix=bond_matrix_block_embedding, 
        nholes=nholes_block + nholes_neighbor, 
        nbeta=nbeta_block + nbeta_neighbor,
        sites_restriction=sites_restriction_embedding,
        lattice_type='clusters',
        nbh=2 * nbh, nbv=nbv, nblocs_side=nblocs_side)

    configuration_full = bC.Configuration(
        bond_matrix=bond_matrix_block_embedding, 
        nholes=nholes_block + nholes_neighbor, 
        nbeta=nbeta_block + nbeta_neighbor,
        sites_restriction=[],
        lattice_type='clusters',
        nbh=2 * nbh, nbv=nbv, nblocs_side=nblocs_side)

    ### Self-coherent iterations ###

    ind_iter = 0
    ground_state_temp = np.copy(ground_state_block)
    ground_state_energy = gs_energy_block

    with tqdm(total=max_iter) as pbar :
        while ind_iter < max_iter:

            ham_block_embedded = ham_block.tocoo(copy=True)

            probabilities_spins_B = bC.get_spins_probabilitites_lattice(basis_block, ground_state_temp, reverse_spins=reverse_spins)

            for ind_A, det_A in enumerate(determinants_block):
                #print("[emb] ---------------------------------------------------------------------------------------------------------------------------------------")

                #print("[emb] ground_state_energy = ", ground_state_energy)

                vect_A = np.zeros(shape=(nb_conf_block, 1))
                vect_A[ind_A] = 1
                probabilities_spins_A = bC.get_spins_probabilitites_lattice(basis_block, vect_A)

                intr = 0

                for site_A in frontier_A:
                    proba_alpha_A = probabilities_spins_A[1][site_A]
                    proba_beta_A = probabilities_spins_A[-1][site_A]
                    proba_holes_A = probabilities_spins_A[0][site_A]

                    foreign_neighbors = list(set(frontier_B).intersection(set(lF.get_neighbors(site_A, bond_matrix_block_embedding, sites_restriction_embedding))))
                    
                    for site_B in foreign_neighbors:
                        site_B = site_B - nsites_block

                        proba_alpha_B = probabilities_spins_B[1][site_B]
                        proba_beta_B = probabilities_spins_B[-1][site_B]
                        proba_holes_B = probabilities_spins_B[0][site_B]

                        intr += -(J/2) * (proba_alpha_A * proba_beta_B + proba_beta_A * proba_alpha_B) + VNN * proba_holes_A * proba_holes_B

                    foreign_diagonal_second_neighbors = list(set(frontier_B).intersection(
                        set(lF.get_diagonal_second_neighbors(configuration_embedding, site_A))
                        ))
                    for site_B in foreign_diagonal_second_neighbors:
                        site_B = site_B - nsites_block

                        proba_holes_B = probabilities_spins_B[0][site_B]

                        intr += VNNNper * proba_holes_A * proba_holes_B
                if neighbor_embedding:
                    ham_block_embedded = hC.add_ham_coo(ham=ham_block_embedded, \
                        elt= intr, \
                        ind_row=ind_A, ind_col=ind_A)
                else:
                    for rot in range(4):
                        rotated_ind_A = bC.get_index_in_basis(configuration_block, 
                            bC.rotate_determinant(rot, configuration_block, det_A))

                        ham_block_embedded = hC.add_ham_coo(ham=ham_block_embedded, \
                            elt= intr, \
                            ind_row=rotated_ind_A, ind_col=rotated_ind_A)
                
                if dynamic_embedding:
                    # Register of generated perturbators
                    perturbators_memory = {}
                    for delta_nalpha, delta_nbeta in ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)):
                        configuration_temp_A = bC.Configuration(
                            bond_matrix_block,
                            nholes_block + (delta_nalpha + delta_nbeta == -1) - (delta_nalpha + delta_nbeta == 1),
                            nbeta_block + delta_nbeta,
                            configuration_block.sites_restriction,
                            "periodic",
                            nbh, nbv, nblocs_side
                        )
                        configuration_temp_B = bC.Configuration(
                            bond_matrix_block,
                            nholes_block - (delta_nalpha + delta_nbeta == -1) + (delta_nalpha + delta_nbeta == 1),
                            nbeta_block - delta_nbeta,
                            configuration_block.sites_restriction,
                            "periodic",
                            nbh, nbv, nblocs_side
                        )
                        perturbators_memory[(configuration_temp_A.get_id(), configuration_temp_B.get_id())] = []
                    spins_attributes = {-1/2 : "nbeta", 0 : "nholes", 1/2 : "nalpha"}
                    # Definition of the perturbators
                    #print("[emb] det_A : \n", bC.get_displayable_rectangular_lattice(configuration_block, det_A))
                    for site in frontier_A:
                        spin = bC.get_spin_site(configuration_block, det_A, site)
                        #print("[emb] site : {} | spin = {}".format(site, spin))
                        other_spins = [-1/2, 0, 1/2]
                        other_spins.remove(spin)
                        for new_spin in other_spins:
                            if getattr(configuration_neighbor, spins_attributes[new_spin]) > 0:

                                configuration_perturbed_A = copy.deepcopy(configuration_block)
                                setattr(configuration_perturbed_A, spins_attributes[new_spin], \
                                    getattr(configuration_block, spins_attributes[new_spin]) + 1)
                                setattr(configuration_perturbed_A, spins_attributes[spin], \
                                    getattr(configuration_block, spins_attributes[spin]) - 1)

                                configuration_perturbed_A = bC.Configuration(
                                    bond_matrix=configuration_perturbed_A.bond_matrix, nholes=configuration_perturbed_A.nholes, nbeta=configuration_perturbed_A.nbeta, 
                                    sites_restriction=configuration_block.sites_restriction, 
                                    nbh=configuration_perturbed_A.nbh, nbv=configuration_perturbed_A.nbv, nblocs_side=configuration_perturbed_A.nblocs_side
                                )


                                configuration_perturbed_B = copy.deepcopy(configuration_neighbor)
                                setattr(configuration_perturbed_B, spins_attributes[spin], \
                                    getattr(configuration_block, spins_attributes[spin]) + 1)
                                setattr(configuration_perturbed_B, spins_attributes[new_spin], \
                                    getattr(configuration_block, spins_attributes[new_spin]) - 1)

                                configuration_perturbed_B = bC.Configuration(
                                    bond_matrix=configuration_perturbed_B.bond_matrix, nholes=configuration_perturbed_B.nholes, nbeta=configuration_perturbed_B.nbeta, 
                                    sites_restriction=(configuration_neighbor.sites_restriction[1], configuration_neighbor.sites_restriction[0]), 
                                    nbh=configuration_perturbed_B.nbh, nbv=configuration_perturbed_B.nbv, nblocs_side=configuration_perturbed_B.nblocs_side
                                )

                                perturbator_A = det_A[:]
                                lists_particles_A = {
                                    "nholes" : bC.get_indexes_of(configuration_block, perturbator_A, 0),
                                    "nbeta" : bC.get_indexes_of(configuration_block, perturbator_A, -1),
                                    "nalpha" : bC.get_indexes_of(configuration_block, perturbator_A, 1)
                                }
                                lists_particles_A[spins_attributes[spin]].remove(site)
                                lists_particles_A[spins_attributes[new_spin]].append(site)
                                perturbator_A = lists_particles_A["nholes"] + lists_particles_A["nbeta"] + lists_particles_A["nalpha"]

                                perturbator_A = bC.reorder_basis_element(configuration_perturbed_A, perturbator_A)
                                

                                new_sites_restriction = [new_site for new_site in lF.get_neighbors(site, bond_matrix_block_embedding, frontier_B) \
                                        + lF.get_second_neighbors(site, bond_matrix_block_embedding, frontier_B)]



                                for det_B in determinants_neighbor:
                                    #print("[emb] det_B : \n", bC.get_displayable_rectangular_lattice(configuration_block, det_B))
                                    #print("[emb] det : \n", bC.get_displayable_neighbor(configuration_block, configuration_block, det_A, det_B))
                                    
                                    for new_site in new_sites_restriction:
                                        #print("[emb] new_site : {} | new_spin = {}".format(new_site, new_spin))
                                        new_site = new_site - nsites_block
                                        if bC.get_spin_site(configuration_neighbor, det_B, new_site) == new_spin: # Research for new_spin in new_sites_restriction may be optimized
                                            reverse_interactions = {
                                                0 : iC.BoolInteractions(bt=False, bz=False, bij=True, bSD=False, btN=False, bV=False, bV_embedding=False, b_spin_multipicities=False),
                                                1 : iC.BoolInteractions(bt=True, bz=False, bij=False, bSD=True, btN=True, bV=False, bV_embedding=False, b_spin_multipicities=False),
                                                -1 : iC.BoolInteractions(bt=True, bz=False, bij=False, bSD=True, btN=True, bV=False, bV_embedding=False, b_spin_multipicities=False)
                                            }

                                            perturbator_B = det_B[:]
                                            lists_particles_B = {
                                                "nholes" : bC.get_indexes_of(configuration_neighbor, perturbator_B, 0),
                                                "nbeta" : bC.get_indexes_of(configuration_neighbor, perturbator_B, -1),
                                                "nalpha" : bC.get_indexes_of(configuration_neighbor, perturbator_B, 1)}
                                            lists_particles_B[spins_attributes[new_spin]].remove(new_site)
                                            lists_particles_B[spins_attributes[spin]].append(new_site)
                                            perturbator_B = lists_particles_B["nholes"] \
                                                + lists_particles_B["nbeta"] + lists_particles_B["nalpha"]
                                            perturbator_B = bC.reorder_basis_element(configuration_perturbed_B, perturbator_B)

                                            ind_perturbator_A = bC.compute_hash_table(configuration_perturbed_A, perturbator_A)
                                            ind_perturbator_B = bC.compute_hash_table(configuration_perturbed_B, perturbator_B)


                                            if (ind_perturbator_A, ind_perturbator_B) not in perturbators_memory[(configuration_perturbed_A.get_id(), configuration_perturbed_B.get_id())]:

                                                perturbators_memory[(configuration_perturbed_A.get_id(), configuration_perturbed_B.get_id())].append((ind_perturbator_A, ind_perturbator_B))

                                                delta_n_B = configuration_perturbed_B.nalpha + configuration_perturbed_B.nbeta \
                                                    - configuration_neighbor.nalpha - configuration_neighbor.nbeta

                                                delta_ms_B = (configuration_perturbed_B.nalpha - configuration_perturbed_B.nbeta \
                                                    - configuration_neighbor.nalpha + configuration_neighbor.nbeta) / 2

                                        
                                                perturbator = bC.extend_determinant(configuration_perturbed_A, \
                                                    configuration_perturbed_B, perturbator_A, perturbator_B)

                                                energy_perturbator = itr.compute_energy(configuration_full, interactions, perturbator)
                                                
                                                #print("[emb] dets : {} - {}\n".format(det_A, det_B), bC.get_displayable_neighbor(configuration_block, configuration_block, det_A, det_B))
                                                #print("[emb] perturbator : {}\n".format(perturbator), bC.get_displayable_neighbor(configuration_perturbed_A, configuration_perturbed_B, perturbator_A, perturbator_B))
                                                #print("[emb] energy_perturbator = ", energy_perturbator)

                                                configuration_embedding_reverse = copy.deepcopy(configuration_embedding)
                                                configuration_embedding_reverse.sites_restriction = (
                                                    list(set(configuration_embedding.sites_restriction[1]).intersection(set([site_B + nsites_block for site_B in \
                                                        bC.get_indexes_of(configuration_perturbed_B, perturbator_B, max(delta_n_B, 0) * 2 * delta_ms_B)]))), 
                                                    list(set(configuration_embedding.sites_restriction[0]).intersection(set([site_A for site_A in \
                                                        bC.get_indexes_of(configuration_perturbed_A, perturbator_A, min(0, delta_n_B) * 2 * delta_ms_B)])))
                                                )
                                                #print("[emb] reverse sites restriction = ", configuration_embedding_reverse.sites_restriction)
                                                #print("[emb] lattice_type = ", configuration_embedding_reverse.lattice_type)

                                                coupled_dets = itr.compute_dynamic_interactions(configuration_embedding_reverse, interactions, reverse_interactions[delta_n_B], perturbator)
                                                #print("[emb] coupled_dets :")
                                                for det_coupled, coupling_det in coupled_dets.items():
                                                    #print("-- ", coupling_det, " : ", det_coupled)
                                                    det_coupled_A = [site for site in det_coupled if site < nsites_block]
                                                    det_coupled_B = [site - nsites_block for site in det_coupled if site >= nsites_block]
                                                    #print(bC.get_displayable_neighbor(configuration_block, configuration_block, det_coupled_A, det_coupled_B))
                                                

                                                coupled_dets_list = list(coupled_dets.keys())
                                                for ind, new_det_1 in enumerate(coupled_dets_list[:-1]):
                                                    for ind_temp, new_det_2 in enumerate(coupled_dets_list[ind + 1:]):
                                                        new_det_1 = tuple(new_det_1)
                                                        coupling_1 = coupled_dets[new_det_1]
                                                        coupling_2 = coupled_dets[new_det_2]

                                                        ind_new_det_B_1 = bC.get_index_in_basis(
                                                            configuration_block, [site - nsites_block for site in new_det_1 if site >= nsites_block])
                                                        ind_new_det_B_2 = bC.get_index_in_basis(
                                                            configuration_block, [site - nsites_block for site in new_det_2 if site >= nsites_block])

                                                        coupling = (np.conj(ground_state_temp[ind_new_det_B_1, 0]) \
                                                            * ground_state_temp[ind_new_det_B_2, 0] \
                                                            * coupling_1 * np.conj(coupling_2)) / (2 * ground_state_energy \
                                                                - energy_perturbator)
                                                        #print("[emb] ({} - {}) : coupling = {} * {} * ({} * {}) / (2 * {} - {}) = ".format(ind, ind_temp + ind + 1, np.conj(ground_state_temp[ind_new_det_B_1, 0]), ground_state_temp[ind_new_det_B_2, 0], coupling_1, np.conj(coupling_2), ground_state_energy, energy_perturbator), coupling)


                                                        if neighbor_embedding:
                                                            new_det_A_1 = [site for site in new_det_1 if site < nsites_block]
                                                            new_det_A_2 = [site for site in new_det_2 if site < nsites_block]
                                                            ind_1 = bC.get_index_in_basis(configuration_block, new_det_A_1)
                                                            ind_2 = bC.get_index_in_basis(configuration_block, new_det_A_2)

                                                            if abs(coupling) >= threshold_mel:
                                                                ham_block_embedded = hC.add_ham_coo(ham=ham_block_embedded, \
                                                                    elt= coupling, \
                                                                    ind_row=ind_1, ind_col=ind_2)

                                                                ham_block_embedded = hC.add_ham_coo(ham=ham_block_embedded, \
                                                                    elt= np.conj(coupling), \
                                                                    ind_row=ind_2, ind_col=ind_1)
                                                        else:
                                                            for rot in range(4):
                                                                
                                                                new_det_A_1 = [site for site in new_det_1 if site < nsites_block]
                                                                new_det_A_2 = [site for site in new_det_2 if site < nsites_block]
                                                                rotated_ind_1 = bC.get_index_in_basis(configuration_block, bC.rotate_determinant(rot, configuration_block, new_det_A_1))
                                                                rotated_ind_2 = bC.get_index_in_basis(configuration_block, bC.rotate_determinant(rot, configuration_block, new_det_A_2))

                                                                if abs(coupling) >= threshold_mel:
                                                                    ham_block_embedded = hC.add_ham_coo(ham=ham_block_embedded, \
                                                                        elt= coupling, \
                                                                        ind_row=rotated_ind_1, ind_col=rotated_ind_2)

                                                                    ham_block_embedded = hC.add_ham_coo(ham=ham_block_embedded, \
                                                                        elt= np.conj(coupling), \
                                                                        ind_row=rotated_ind_2, ind_col=rotated_ind_1)
                                                #print("\n\n")

            
            ham_block_embedded = ham_block_embedded.tocsr(copy=False)

            #####
            printed_difference = [] 
            for ind, diag in enumerate(ham_block_embedded.diagonal()):
                diff = diag - ham_block.diagonal()[ind]
                if abs(diff) > 1e-3:
                    printed_difference.append(diff)
                else:
                    printed_difference.append(0)
            #print("[emb] diag(ham_block__embedded - ham_block) =\n", printed_difference) 
            #####

            ###
            #dressing = ham_block_embedded - ham_block
            #print("[emb] dressing =\n", (16 * 11 * 12) * dressing.toarray())
            ###

            while True:
                eigenstates_embedded = dav.modified_davidson_algorithm(
                    ham_block_embedded, nb_val=1, initial_guess=ground_state_block, stp=stp, tol=tol, dim_max=dim_max, verbose=False
                )
                if eigenstates_embedded.eigvals:
                    break
            
            ground_state_embedded = eigenstates_embedded.eigvects[:,[0]]
            ground_state_energy = eigenstates_embedded.eigvals[0]

            if lin.norm(ground_state_embedded - ground_state_temp) / lin.norm(ground_state_temp) < self_consistency_tol:

                data_computation = iC.DataComputations(data, determinants_block, ham_block_embedded, eigenstates_embedded)
                print("[emb] ground state energy = {} | embedded ground state energy = {}".format(\
                    eigenstates_block.eigvals[0], eigenstates_embedded.eigvals[0]))
                save_data_embedding_blocks(data_computation, dynamic_embedding, neel_embedding, neighbor_embedding)

                return data_computation

            ind_iter += 1
            pbar.update(1)
            ground_state_temp = ground_state_embedded

    print("Maximum number of iterations exceeded")
    print("[emb] ground state energy = {} | embedded ground state energy = {}".format(\
                eigenstates_block.eigvals[0], eigenstates_embedded.eigvals[0]))

def save_data_embedding_blocks(data_computations, dynamic_embedding, neel_embedding, neighbor_embedding):
    """
    Saves the data from the computations in a JSON format
    Args:
        data_computations (class DataComputation): the results of the computations
        dynamical_embedding (bool): True to perform dynamical embedding
        neel_embedding (bool): True to perform a Néel embedding beforehand
        neighbor_embedding (bool): True to perform an embedding with a single neighboring block
    """

    # Configuration
    configuration = data_computations.data.configuration
    nsites = configuration.nsites
    nalpha = configuration.nalpha
    nbeta = configuration.nbeta

    # BoolInteractions
    bool_interactions = data_computations.data.bool_interactions
    bt = bool_interactions.bt
    bz = bool_interactions.bz
    bij = bool_interactions.bij
    bSD = bool_interactions.bSD
    btN = bool_interactions.btN
    bV = bool_interactions.bV

    # File Name
    file_name = "tJHam"
    if dynamic_embedding:
        file_name += "-dyn"
    if neel_embedding:
        file_name += "-neel"
    if neighbor_embedding:
        file_name += "-neigh"
    file_name += "-{}{}{}".format(nsites, nalpha, nbeta)
    if bt and bz and bij and bSD and btN and bV:
        file_name += "Full.json"
    else:
        extensions = {"bt":"t", "bij":"J", "bSD":"SD", "btN":"tN", "bV":"V"}
        for name in dir(bool_interactions):
            if name in extensions:
                file_name += extensions[name]
        file_name += ".json"
 
    # Saving results
    try :
        with open("./initialComputations/Embedding_blocks/{}".format(file_name), "w") as savefile:
            savefile.write(json.dumps({
                "configuration" : configuration.__to_json__(), 
                "interactions" : data_computations.data.interactions.__dict__, 
                "diagonalization" : data_computations.data.diagonalization.__dict__, 
                "bool_interactions" : data_computations.data.bool_interactions.__dict__,
                "determinants" : data_computations.determinants, 
                "hamiltonian" : {
                    "data" : data_computations.hamiltonian.data.tolist(),
                    "indices" : data_computations.hamiltonian.indices.tolist(),
                    "indptr" : data_computations.hamiltonian.indptr.tolist()
                }, 
                "eigenstates" : {
                    "eigvals" : data_computations.eigenstates.eigvals,
                    "spins" : data_computations.eigenstates.spins,
                    "eigvects" : data_computations.eigenstates.eigvects.tolist()
                    }
            })) 
    except OSError:
        os.mkdir("./initialComputations/Embedding_blocks")
        with open("./initialComputations/Embedding_blocks/{}".format(file_name), "w") as savefile:
            savefile.write(json.dumps({
                "configuration" : configuration.__to_json__(), 
                "interactions" : data_computations.data.interactions.__dict__, 
                "diagonalization" : data_computations.data.diagonalization.__dict__, 
                "bool_interactions" : data_computations.data.bool_interactions.__dict__,
                "determinants" : data_computations.determinants, 
                "hamiltonian" : {
                    "data" : data_computations.hamiltonian.data.tolist(),
                    "indices" : data_computations.hamiltonian.indices.tolist(),
                    "indptr" : data_computations.hamiltonian.indptr.tolist()
                }, 
                "eigenstates" : {
                    "eigvals" : data_computations.eigenstates.eigvals,
                    "spins" : data_computations.eigenstates.spins,
                    "eigvects" : data_computations.eigenstates.eigvects.tolist()
                    }
            })) 