import math
import numpy as np
import itertools as it
import copy
import scipy.special as spec

import baseConstruction as bC

### Combinatorial functions
    
def compute_dimension(nsites, nalpha, nbeta):
    """
    Computes the dimension of the Hilbert space for a block of nsites sites, with nalpha spins up and nbeta spins down.
    Args:
        nsites (int): number of sites
        nalpha (int): number of spins up
        nbeta (int): number of spins down
    Returns :
        dim (int): the dimension of the states (Hilbert) space
    """
    nholes = nsites - nalpha - nbeta
    return spec.comb(nsites, nholes, exact=True)*spec.comb(nsites - nholes, nalpha, exact=True)
    
def compute_ms(nalpha, nbeta):
    """
    Computes the spin number of a configuration with nalpha spins up and nbeta spins down.
    Args:
        nalpha (int): number of spins up
        nbeta (int): number of spins down
    Returns:
        ms (float): the spin number of the configuration
    """
    return nalpha/2 - nbeta/2
  
    
### Square lattice functions

def build_J_matrix(dim):
    """
    Builds the J square matrix :math:`\\delta_{i,j-1} + \\delta_{i,j+1}` used in the definitions of the bond matrices.
    Args:
        dim (int): the dimension of the J matrix
    Returns:
        J (np.array): the J matrix
    """
    return np.diag(np.array([1 for _ in range(dim - 1)]), 1) + np.diag(np.array([1 for _ in range(dim - 1)]), -1)

def build_K_matrix(dim):
    """
    Builds the K matrix of dimension (4 x dim) used in the definition of the bond matrices.
    Args:
        dim (int): the number of columns of the K matrix
    Returns:
        K (np.array): the K matrix
    """
    K = np.zeros((4, dim))
    K[0, 0] = 1
    K[1, 1] = 1
    K[2, dim - 2] = 1
    K[3, dim - 1] = 1
    return K

def build_L_matrix(dim):
    """
    Builds the L matrix of dimension (dim x 4) used in the definition of the bond matrices.
    Args:
        dim (int): the number of rows of the L matrix
    Returns:
        L (np.array): the L matrix
    """
    L = np.zeros((dim, 4))
    L[0, 1] = 1
    L[dim - 1, 2] = 1
    return L

def build_U_matrix(nbh, nbv, nblocs_side):
    """
    Builds the U matrix of dimension (dim x dim) used for the connection of side by side square blocks
    Args:
        nbh (int): number of subblocks horizontally
        nbv (int): number of subblocks vertically
        nblocs_side (int): number of sites per side on a subblock
    Returns:
        U (np.array): the U matrix
    """
    dim = nbh * nbv * (nblocs_side ** 2)
    U = np.zeros((dim, dim))
    for ind_i in range(nbv * nblocs_side):
        U[ind_i * nbh * nblocs_side, (ind_i + 1) * nbh * nblocs_side - 1] = 1
    return U
def build_T_matrix(nbh, nbv, nblocs_side):
    """
    Builds the T matrix of dimension (nsites x nsites_periph = 2 * (nbh + nbv) * nblocs_side) used for the connection of a block with a surrounding Néel lattice
    """
    # Block 1
    T = np.concatenate((np.eye(nbh * nblocs_side), np.zeros(((nbv * nblocs_side - 1) * nbh * nblocs_side, nbh * nblocs_side))), axis=0)
    # Block 2
    col = np.zeros((nbh * nbv * (nblocs_side ** 2), nbv * nblocs_side))
    for ind_col in range(nbv * nblocs_side):
        col[(ind_col + 1) * nbh * nblocs_side - 1, ind_col] = 1
    T = np.concatenate((T, col), axis=1)
    # Block 3
    inv_id = np.zeros((nbh * nblocs_side, nbh * nblocs_side))
    for ind_col in range(nbh * nblocs_side):
        inv_id[nbh * nblocs_side - 1 - ind_col, ind_col] = 1
    col = np.concatenate((np.zeros(((nbv * nblocs_side - 1) * nbh * nblocs_side, nbh * nblocs_side)), inv_id), axis=0)
    T = np.concatenate((T, col), axis=1)
    # Block 4
    col = np.zeros((nbh * nbv * (nblocs_side ** 2), nbv * nblocs_side))
    for ind_col in range(nbv * nblocs_side):
        col[(nbv * nblocs_side - ind_col - 1) * nbh * nblocs_side, ind_col] = 1
    T = np.concatenate((T, col), axis=1)
    return T

def build_rectangular_bond_matrix(nbh, nbv, nblocs_side): 
    """
    Computes the adjacency matrix in a rectangular lattice made of nbv nblocs*nblocs square blocs vertically \
        and nbh horizontally
    Args:
        nbh (int): number of sites horizontally on a subblock
        nbv (int): number of sites vertically on a subblock
        nblocs_side (int): the number of blocs on each side of the lattice
    Returns:
        bond_matrix (np.array): the adjacency matrix of the lattice
    """
    nsites = nbh * nbv * (nblocs_side ** 2)

    J_matrix = build_J_matrix(nblocs_side * nbh)
            
    bond_matrix = np.zeros((nsites, nsites))
    
    for iB in range(nblocs_side * nbv):
        bond_matrix[nblocs_side * nbh * iB : nblocs_side * nbh * (iB + 1), nblocs_side * nbh * iB : nblocs_side * nbh * (iB + 1)] = J_matrix
    for jB in range(nblocs_side * nbv - 1):
        bond_matrix[nblocs_side * nbh * (jB + 1) : nblocs_side * nbh*(jB + 2), nblocs_side * nbh * jB : nblocs_side * nbh * (jB + 1)] = np.eye(nblocs_side * nbh)
        bond_matrix[nblocs_side * nbh * jB : nblocs_side * nbh * (jB + 1), nblocs_side * nbh * (jB + 1) : nblocs_side * nbh * (jB + 2)] = np.eye(nblocs_side * nbh)
    return bond_matrix

def build_bond_matrix_neel_outer_lattice(nbh, nbv, nblocs_side):
    """
    Computes the adjacency matrix of the Néel lattice sourrounding a rectangular lattice \
        made of nbv nblocs*nblocs square blocs vertically and nbh horizontally
    Args:
        nbh (int): number of sites horizontally on a subblock
        nbv (int): number of sites vertically on a subblock
        nblocs_side (int): the number of blocs on each side of the lattice
    Returns:
        bond_matrix_neel (np.array): the adjacency matrix of the Néel surrounding lattice
    """
    # Column 1
    bond_matrix_rest = np.concatenate((build_J_matrix(nbh * nblocs_side), np.zeros(((nbh + 2 * nbv) * nblocs_side, nbh * nblocs_side))), axis=0)
    # Column 2
    col = np.concatenate((np.zeros((nbh * nblocs_side, nbv * nblocs_side)), build_J_matrix(nbv * nblocs_side), np.zeros(((nbh + nbv) * nblocs_side, nbv * nblocs_side))), axis=0)
    bond_matrix_rest = np.concatenate((bond_matrix_rest, col), axis=1)
    # Column 3
    col = np.concatenate((np.zeros(((nbh + nbv) * nblocs_side, nbh * nblocs_side)), build_J_matrix(nbh * nblocs_side), np.zeros((nbv * nblocs_side, nbh * nblocs_side))), axis=0)
    bond_matrix_rest = np.concatenate((bond_matrix_rest, col), axis=1)
    # Column 4
    col = np.concatenate((np.zeros(((2 * nbh + nbv) * nblocs_side, nbv * nblocs_side)), build_J_matrix(nbv * nblocs_side)), axis=0)
    bond_matrix_rest = np.concatenate((bond_matrix_rest, col), axis=1)

    return bond_matrix_rest

def build_bond_matrix_neel_embedding(nbh, nbv, nblocs_side):
    """
    Computes the adjacency matrix in a rectangular lattice made of nbv nblocs*nblocs square blocs vertically \
        and nbh horizontally, embedded in a Néel lattice
    Args:
        nbh (int): number of sites horizontally on a subblock
        nbv (int): number of sites vertically on a subblock
        nblocs_side (int): the number of blocs on each side of the lattice
    Returns:
        bond_matrix_neel (np.array): the adjacency matrix of the complete lattice
    """

    ### Block ########################################
    bond_matrix = build_rectangular_bond_matrix(nbh, nbv, nblocs_side)

    ### Néel lattice ########################################

    bond_matrix_rest = build_bond_matrix_neel_outer_lattice(nbh, nbv, nblocs_side)

    ### Connection ########################################

    # blocks 1 - 2
    connexion = build_T_matrix(nbh, nbv, nblocs_side)

    ### Full matrix ########################################
    bond_matrix_neel = np.concatenate((\
        np.concatenate((bond_matrix, connexion.T), axis=0), \
        np.concatenate((connexion, bond_matrix_rest), axis=0)), axis=1)

    return bond_matrix_neel

def build_bond_matrix_block_embedding(nbh, nbv, nblocs_side):
    """
    Computes the adjacency matrix in a rectangular lattice made of a block of nbv nblocs*nblocs square subblocs vertically \
        and nbh horizontally, embedded within a similar neighboring block side by side
    Args:
        nbh (int): number of sites horizontally on a subblock
        nbv (int): number of sites vertically on a subblock
        nblocs_side (int): the number of blocs on each side of the lattice
    Returns:
        bond_matrix_emb_blocks (np.array): the adjacency matrix of the complete lattice
    """
    ### Blocks ########################################
    bond_matrix = build_rectangular_bond_matrix(nbh, nbv, nblocs_side)

    ### Connections ###################################
    connection_matrix = build_U_matrix(nbh, nbv, nblocs_side)

    ### Assembly ######################################
    row_0 = np.concatenate((bond_matrix, connection_matrix.T), axis=1)
    row_1 = np.concatenate((connection_matrix, bond_matrix), axis=1)

    bond_matrix_emb_blocks = np.concatenate((row_0, row_1), axis=0)

    return bond_matrix_emb_blocks



### Neighbors functions
    
def build_bond_list(bond_matrix, sites_restriction=[]):
    """
    Returns the list of adjacent sites, for a configuration defined by nblocs subblocks with nbh sites horizontally \
        and nbv sites vertically
    Args:
        bond_matrix (np.array): the bond matrix of the lattice
        sites_restriction (tuple<list<int>>): a restriction of the bonds on the sites that appears here
    Returns:
        bond_list (list<tuple<int, int>>): the list of adjacent sites in the lattice
    """
    nsites = bond_matrix.shape[0]
    if not sites_restriction:
        sites_restriction = (list(range(nsites)), list(range(nsites)))

    bond_list = []
    for ind_i in list(set(range(nsites)) & set(sites_restriction[0])):
        for ind_j in list(set(range(ind_i, nsites)) & set(sites_restriction[1])):
            if bond_matrix[ind_i, ind_j]==1:
                bond_list.append([ind_i, ind_j])
    return bond_list

def get_neighbors(n_site, bond_matrix, sites_restriction=[]): 
    """
    Return the indices of sites in the lattice, defined by its adjacency matrix bond_matrix,\
         that are neighbors to the site of index n_site
    Args:
        n_site (int): index of the specified site in the adjacency matrix
        bond_matrix (np.array): the bond_matrix describing the lattice
        sites_restriction (tuple<list<int>>): a restriction of the bonds on the sites that appears here
    Returns:
        neigh (list<int>): the list of neighbors of n_site in the lattice
    """
    nsites = bond_matrix.shape[0]
    if not sites_restriction:
        sites_restriction = list(range(nsites))
    elif isinstance(sites_restriction, tuple):
        sites_restriction = sites_restriction[1]
    neigh = []
    bond_matrix_n = bond_matrix[n_site,:]
    for ind in list(set(range(nsites)) & set(sites_restriction)):
        if bond_matrix_n[ind] == 1:
            neigh.append(ind)
    return neigh

def get_second_neighbors(n_site, bond_matrix, sites_restriction=[]):
    """
    Return the indices of sites in the lattice, defined by its adjacency matrix bond_matrix,\
         that are second neighbors to the site of index n_site
    Args:
        n_site (int): index of the specified site in the adjacency matrix
        bond_matrix (np.array): the bond_matrix describing the lattice
        sites_restriction (tuple<list<int>>): a restriction of the bonds on the sites that appears here
    Returns:
        sec_neighbors (list<int>): the list of second neighbors of n_site in the lattice
    """
    nsites = bond_matrix.shape[0]
    if not sites_restriction:
        sites_restriction = (list(range(nsites)), list(range(nsites)))

    neighbors = get_neighbors(n_site, bond_matrix, [])
    non_second_neighbors = {n_site}.union(set(neighbors))
    sec_neighbors = []
    for neigh in neighbors:
        sec_neigh = list(set(get_neighbors(neigh, bond_matrix, sites_restriction)).difference(non_second_neighbors))
        sec_neighbors += sec_neigh
    return list(set(sec_neighbors))

def get_diagonal_second_neighbors(configuration, site):
    """
    Return the indices of sites in the lattice, defined by its adjacency matrix bond_matrix,\
         that are diagonally second neighbors to the site of index n_site
    Args:
        site (int): index of the specified site in the adjacency matrix
        configuration (class baseConstruction.Configuration): the configuration of our lattice (necessary to define diagonal bonds)
    Returns:
        diagonal_sec_neighbors (list<int>): the list of second neighbors of n_site in the lattice
    """
    bond_matrix = configuration.bond_matrix
    sites_restriction = configuration.sites_restriction

    neighbors = get_neighbors(site, bond_matrix, [])
    non_second_neighbors = {site}.union(set(neighbors))
    diagonal_sec_neighbors = []
    for neigh in neighbors:
        sec_neigh = list(set(get_neighbors(neigh, bond_matrix, sites_restriction)).difference(non_second_neighbors))
        for sec_neighbor in sec_neigh:
            if sec_neighbor not in diagonal_sec_neighbors and not bC.is_bond_linear(configuration, site, neigh, sec_neighbor):
                diagonal_sec_neighbors.append(sec_neighbor)
    return list(set(diagonal_sec_neighbors))
