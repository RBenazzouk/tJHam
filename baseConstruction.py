import copy
import numpy as np
import math as mp
import scipy.special as spec

import latticeFunctions as lF

from itertools import combinations, permutations, product
from tqdm import tqdm, trange # Install tqdm via pip    

class Configuration :
    """
    Defines a configuration, ie a rectangular lattice occupied by electrons or holes
    """
    def __init__(self, bond_matrix, nholes, nbeta, sites_restriction=[], lattice_type='periodic', nbh=0, nbv=0, nblocs_side=0):
        """
        Args:
            bond_matrix (np.array): the bond matrix of the lattice
            nholes (int): number of holes in the lattice
            nbeta (int): number of spins down in the lattice
            sites_restriction (tupe<list<int>> or list<int>): a restriction of the sites that interacts. If a tuple, the first element is the "starting" sites restriction \
                and the second element is the "target" sites restriction
            lattice_type (str): type of indexation for the lattice, either 'periodic', 'clusters' or 'peripheral'. In a periodic lattice, sites are indexed row by row, \
                in a cluster-type lattice, they are indexed row by row within each cluster, \
                in a peripheral lattice, they are indexed, in the block, as for a periodic-type lattice, and the peripheral sites are then indexed counter clockwised.
            nbh (int): if rectangular lattice, number of subblocks horizontally, else 0
            nbv (int): if rectangular lattice, number of subblocks vertically, else 0
            nblocs_side (int): if rectangular lattice, number of sites per each edge of subblocks, else 0
        """
        self.bond_matrix = bond_matrix
        self.nsites = bond_matrix.shape[0]
        self.nholes = nholes
        self.nbeta = nbeta
        self.nalpha = self.nsites - nholes - nbeta
        self.spin_number = lF.compute_ms(self.nalpha,nbeta)
        self.doping = nholes/self.nsites
        self.nb_conf = lF.compute_dimension(self.nsites, self.nalpha, nbeta)
        if not sites_restriction:
            self.sites_restriction = (list(range(self.nsites)), list(range(self.nsites)))
        elif isinstance(sites_restriction, tuple) :
            self.sites_restriction = sites_restriction
        elif isinstance(sites_restriction, list):
            self.sites_restriction = (sites_restriction, sites_restriction)     
        if lattice_type in ("periodic", "clusters", "peripheral"):
            self.lattice_type = lattice_type
        else:
            raise ValueError("Wrong argument for lattice_type : {}. Should either be 'periodic', 'clusters' or 'peripheral'.".format(lattice_type))
        self.nbh = nbh
        self.nbv = nbv
        self.nblocs_side = nblocs_side
    def __to_json__(self):
        """
        Returns a dictionary that allows this class to be serialized with JSON.
        Returns:
            configuration_dict (dict): a JSON serializable object representing an instance of this class
        """
        configuration_dict = {key : value for key, value in self.__dict__.items()}
        configuration_dict["bond_matrix"] = self.bond_matrix.tolist()
        return configuration_dict

    def get_id(self):
        """
        For a given lattice topology, returns as a tuple the elements that differentiate the possible configurations
        Returns:
            id (tuple<ind, ind>): a tuple containing the number of holes and beta electrons in the configuration, which defines unequivocally one of these.
        """
        return (self.nholes, self.nbeta)

class Basis :
    """
    Defines the basis of the Hilbert space of our lattice
    """
    def __init__(self, configuration, determinants):
        """
        Args:
            configuration (class Configuration): the configuration of our lattice
            determinants (list<list<int>>): list of every Slater determinant in the basis, encoded by listing first \
                the indices of the sites containing holes, then spins down, and finally spins up
        """
        self.configuration = configuration
        self.determinants = determinants

# Basis functions

def construct_basis(configuration): 
    """
    Return the canonical basis of Slater determinants given a specified configuration
    Args:
        configuration (class Configuration): the configuration of our lattice
    Returns:
        basis (list<list<int>>): list of every Slater determinant in the basis, encoded by listing first the indices \
            of the sites containing holes, then spins down, and finally spins up
    """

    # Configuration
    nsites = configuration.nsites
    nbeta = configuration.nbeta
    nholes = configuration.nholes
    nb_conf = configuration.nb_conf

    determinants = []
    if nholes < 0:
        raise ValueError("More electrons than sites")
        
    possible_sites_holes = [list(comb) for comb in combinations(range(nsites), nholes)] # every possible arrangement for the holes
    with tqdm(total=nb_conf) as prog_bar :
        for sites_holes in possible_sites_holes : # Configuration for the holes, to be filled with electrons

            sites_electrons = list(range(nsites))
            for ind_hole in sites_holes :
                sites_electrons.remove(ind_hole)
                
            possible_sites_beta = [list(comb) for comb in combinations(sites_electrons, nbeta)]

            for sites_beta in possible_sites_beta :
                
                sites_alpha = copy.deepcopy(sites_electrons)
                for ind_b in sites_beta:
                    sites_alpha.remove(ind_b)
                determinants.append(sites_holes + sites_beta + sites_alpha)
                prog_bar.update(1)
    basis = Basis(configuration, determinants)
    return basis
    
def get_index_in_basis(configuration, det): 
    """
    Return the index of the determinant vect in the basis of our configuration
    Args:
        configuration (class Basis): the configuration of our lattice
        det (list<int>): the Slater determinant we seek to find the index in basis
    Returns:
        index (int) the index of vect in basis
    """
    
    return compute_hash_table(configuration, reorder_basis_element(configuration, det))

def reverse_spins_basis(basis):
    """
    Reverses the z-component of every spins in the basis
    Args:
        basis (class Basis): the basis of our configuration
    Returns:
        basis_flip (class Basis): the basis of a new configuration with reversed spins z-components
    """
    configuration = basis.configuration
    determinants = basis.determinants

    basis_flip = Basis(reverse_spins(configuration, determinants[0])[0], [reverse_spins(configuration, det)[1] for det in determinants])
    return basis_flip

def reverse_spins_vect(basis, vect):
    """
    Reverses the z-component of every spins in the state vect, expressed in basis
    Args:
        basis (class Configuration): the basis of Slater determinants of the lattice
        vect (np.array): the state of the lattice
    Returns:
        vect_flip (np.array): the corresponding state of the lattice with reversed spin-z components
    """
    configuration = basis.configuration
    determinants = basis.determinants

    vect_flip = np.zeros(vect.shape)
    for ind, det in enumerate(determinants):
        configuration_flip, det_flip = reverse_spins(configuration, det)
        ind_det_flip = get_index_in_basis(configuration_flip, det_flip)
        vect_flip[ind_det_flip] = vect[ind]
    return vect_flip


# Determinants functions

def get_indexes_of(configuration, det, spin, target=0): 
    """
    Return the indexes of the holes (if input spin is 0), alpha (if spin is 1) or beta (if spin is -1) electrons in the lattice \
        in the determinant vect
    Args:
        configuration (class Configuration): the configuration of the lattice
        vect (list<int>): a Slater determinant
        spin (int): either 0 for a hole, 1 for a spin up, and -1 for a spin down
        target (int): index of the sites restriction array if it is a tuple
    Returns:
        sites (list<int>): the list of sites containing spin
    """
    # Configuration
    nbeta = configuration.nbeta
    nholes = configuration.nholes
    sites_restriction = configuration.sites_restriction
    if isinstance(sites_restriction, tuple):
        sites_restriction = sites_restriction[target]

    if spin == 0 :
        return list(set(det[0 : nholes]) & set(sites_restriction))
    elif spin == -1 :
        return list(set(det[nholes : nholes + nbeta]) & set(sites_restriction))
    elif spin == 1 :
        return list(set(det[nholes + nbeta :]) & set(sites_restriction))
    else :
        raise ValueError("Inadequate value for s, should either be 0 (hole), 1 (alpha) or -1 (beta)")

def reorder_basis_element(configuration, vect):
    """
    Reorder the sites of holes, beta and alpha electrons respectively in vect to obtain a well-ordered element of a basis
    Args:
        configuration (class Basis): the configuration of our lattice
        vect (list<int>): the Slater determinant we seek to find the index in basis
    Returns:
        det (list<int>): vect ordered
    """
    # Configuration
    nbeta = configuration.nbeta
    nholes = configuration.nholes
    
    vect_hole = vect[0 : nholes]
    vect_hole.sort()
    vect_beta = vect[nholes : nholes + nbeta]
    vect_beta.sort()
    vect_alpha = vect[nholes + nbeta :]
    vect_alpha.sort()
    return vect_hole + vect_beta + vect_alpha

def extend_determinant(configuration_1, configuration_2, det_1, det_2):
    """
    Returns the determinant resulting of the extention of lattice 1, in the state det_1, with lattice 2, in the state det_2
    Args:
        configuration_1 (class Configuration): the configuration of the original lattice
        configuration_2 (class Configuration): the configuration of the supplementary lattice
        det_1 (list<int>): the determinant in the original lattice
        det_2 (list<int>): the determinant in the extension lattice
    Returns:
        det (list<int>): the corresponding determinant in the extended lattice
    """
    nsites_1 = configuration_1.nsites

    det_1_hole, det_2_hole = get_indexes_of(configuration_1, det_1, 0), get_indexes_of(configuration_2, det_2, 0)
    det_1_beta, det_2_beta = get_indexes_of(configuration_1, det_1, -1), get_indexes_of(configuration_2, det_2, -1)
    det_1_alpha, det_2_alpha = get_indexes_of(configuration_1, det_1, 1), get_indexes_of(configuration_2, det_2, 1)

    det_hole = det_1_hole + [site + nsites_1 for site in det_2_hole]
    det_beta = det_1_beta + [site + nsites_1 for site in det_2_beta]
    det_alpha = det_1_alpha + [site + nsites_1 for site in det_2_alpha]

    return det_hole + det_beta + det_alpha

def reverse_spins(configuration, det):
    """
    Reverses the z-components of every spins in the Slater determinant det.
    Args:
        configuration (class Configuration): the configuration of the lattice
        det (list<int>): the Slater determinant for which we want to reverse the spins z-component
    Returns:
        configuration_flip (class Configuration): the configuration of the lattice with reversed spin z-components
        det_flip (list<int>): the corresponding Slater determinant with reversed spin z-components
    """
    configuration_flip = Configuration(
        configuration.bond_matrix, 
        configuration.nholes, 
        configuration.nalpha, 
        configuration.sites_restriction, 
        configuration.lattice_type,
        configuration.nbh, 
        configuration.nbv, 
        configuration.nblocs_side
    )
    det_flip = get_indexes_of(configuration, det, 0) + get_indexes_of(configuration, det, 1) + get_indexes_of(configuration, det, -1)
    return configuration_flip, reorder_basis_element(configuration_flip, det_flip)


# Sites functions

def get_spin_site(configuration, det, site): 
    """
    Return the spin on the site of index site on the lattice
    Args:
        configuration (class Configuration): the configuration of the lattice
        det (list<int>): the Slater determinant we seek to find the index in basis
        site (int): the index of the desired site
    Returns:
        spin(float): the spin on the desired site    
    """
    # Configuration
    nbeta = configuration.nbeta
    nholes = configuration.nholes
    if det.index(site) < nholes :
        return 0
    elif det.index(site) < nholes + nbeta :
        return -1/2
    else :
        return 1/2

def is_bond_linear(configuration, site1, site2, site3): # Should implement methods specific for each lattice_type, \
                                                        # here works for any lattice type and any two bond matrix, but not optimal
    """
    Asserts if three sites are colinear or orthogonal. 
    Args:
        configuration (class Configuration): the configuration of the lattice
        site1 (list<int>): the first site
        site2 (list<int>): the second site
        site3 (list<int>): the third site
    Returns:
        is_linear (bool): True if colinear, False if orthogonal
    """
    if site1 == site2 or site1 == site3 or site2 == site3:
        raise ValueError("At least two sites provided are equals : ({}, {}, {}).".format(site1, site2, site3))

    bond_matrix = configuration.bond_matrix

    if bond_matrix[site1, site2] == 0 and bond_matrix[site1, site3] == 0:
        raise ValueError("Site {} not in a bond".format(site1))
    if bond_matrix[site1, site2] == 0 and bond_matrix[site2, site3] == 0:
        raise ValueError("Site {} not in a bond".format(site2))
    if bond_matrix[site1, site3] == 0 and bond_matrix[site2, site3] == 0:
        raise ValueError("Site {} not in a bond".format(site3))

    sites = (site1, site2, site3)
    site_2 = [site for site in sites if sum([bond_matrix[site, other_site] for other_site in sites]) == 2][0]
    site_3 = max([site for site in sites if site != site_2])
    site_1 = min([site for site in sites if site != site_2])
    lattice_type = configuration.lattice_type

    if lattice_type == 'periodic': # Sites are indexed row by row
        if abs(site_3 - site_2) == abs(site_2 - site_1):
            return True
        else:
            return False

    elif lattice_type == 'peripheral': # Sites are indexed as in 'periodic' in the cluster, and counter clockwise in the peripheral lattice
        nbh = configuration.nbh
        nbv = configuration.nbv
        nblocs_side = configuration.nblocs_side
        nsites = nbh * nbv * (nblocs_side ** 2)
        if site_1 < nsites and site_2 < nsites and site_3 < nsites:
            configuration_periodic = copy.deepcopy(configuration)
            configuration_periodic.bond_matrix = lF.build_rectangular_bond_matrix(configuration.nbh, configuration.nbv, configuration.nblocs_side)
            configuration_periodic.lattice_type = 'periodic' 
            return is_bond_linear(configuration_periodic, site_1, site_2, site_3)
        elif site_1 >= nsites and site_2 >= nsites and site_3 >= nsites: # If sites are all on the peripheral lattice, there are obviously colinear
            return True
        elif len([site for site in (site_1, site_2, site_3) if site >= nsites]) == 2: # If 2 sites are on the peripheral lattice, the last one can't be colinear
            return False 
        else: # There is one site on the peripheral lattice
            if abs(site_2 - site_1) == 1: # First bond horizontal
                if (site_3 >= nsites and site_3 < nsites + nbh * nblocs_side) or (site_3 >= nsites + (nbh + nbv) * nblocs_side \
                    and site_3 < nsites + (2 * nbh + nbv) * nblocs_side): # Second bond vertical
                    return False
                else: # Second bond horizontal
                    return True
            else: # First bond vertical
                if (site_3 >= nsites and site_3 < nsites + nbh * nblocs_side) or (site_3 >= nsites + (nbh + nbv) * nblocs_side \
                    and site_3 < nsites + (2 * nbh + nbv) * nblocs_side): # Second bond vertical
                    return True
                else: # Second bond horizontal
                    return False

            
    else: # We look for the permutation of the sites indices that makes the indexing as in a lattice of type 'periodic'
        configuration_periodic = copy.deepcopy(configuration)
        configuration_periodic.bond_matrix = lF.build_rectangular_bond_matrix(configuration.nbh, configuration.nbv, configuration.nblocs_side)
        configuration_periodic.lattice_type = 'periodic'
        nbh, nbv, nblocs_side, nsites = configuration.nbh, configuration.nbv, configuration.nblocs_side, configuration.nsites
        
        bond_matrix_clusters = configuration.bond_matrix
        bond_matrix_periodic = lF.build_rectangular_bond_matrix(nbh, nbv, nblocs_side)

        # We compute transfer_matrix, the matrix of the permutation of sites from the cluster type indexing (or any other indexing) to the periodic type indexing
        # so that we find the indexes of the considered sites in a periodic type indexing

        clusters_eigenstates = np.linalg.eigh(bond_matrix_clusters)
        clusters_eigenvectors = clusters_eigenstates[1]

        periodic_eigenstates = np.linalg.eigh(bond_matrix_periodic)
        periodic_eigenvectors = periodic_eigenstates[1]

        # We find that, if C = bond_matrix_clusters, P = bond_matrix_periodic, and V = transfer_matrix, 
        # with VCV.T = P and C = UDU.T, P = WDW.T, with D diagonal, it yields that W.TVU commutes with D, hence is diagonal : V=WD'U.T
        # Given that V is unitary (VV.T=V.TV=I), (D')^2 = I so its diagonal coefficients are either 1 or -1
        # We look for the correct D', that stabilizes 0 (V[0, 0] = 1) and make every coefficient of V live in [0, 1] (V[i, j] = 0 or 1)

        possibles_diag = tuple(product([1, -1], repeat=nsites))
        ind_diag = 0
        is_permutation_matrix = False
        while ind_diag < 2 ** nsites and not is_permutation_matrix :
            commuting_matrix = np.diag(possibles_diag[ind_diag])
            transfer_matrix = np.matmul(periodic_eigenvectors, np.matmul(commuting_matrix, clusters_eigenvectors.T))
            if round(transfer_matrix[0, 0], 10) == 1.:
                ind_row = 0
                ind_col = 0
                while ind_row < nsites and round(transfer_matrix[ind_row, ind_col], 10) in (0., 1.):
                    if ind_col < nsites - 1:
                        ind_col += 1
                    else:
                        ind_col = 0
                        ind_row += 1
                if ind_row == nsites:
                    is_permutation_matrix = True
                else:
                    ind_diag += 1
            else:
                ind_diag += 1

        new_s1 = [row for row in range(nsites) if round(transfer_matrix[row, site_1], 10) == 1][0]
        new_s2 = [row for row in range(nsites) if round(transfer_matrix[row, site_2], 10) == 1][0]
        new_s3 = [row for row in range(nsites) if round(transfer_matrix[row, site_3], 10) == 1][0]

        return is_bond_linear(configuration_periodic, new_s1, new_s2, new_s3)


# Display functions

def get_displayable_determinant(configuration, vect):
    """
    Returns a displayable representation of some element of a basis.
    Args:
        configuration (class Configuration): the configuration of our lattice
        vect (list<int>): a Slater determinant in this basis
    """
    # Configuration
    nbeta = configuration.nbeta
    nholes = configuration.nholes
    
    print_vect=[]
    ind_site = 0
    print_vect = ['+' for _ in vect]
    while ind_site < nholes :
        print_vect[vect[ind_site]] = '0'
        ind_site += 1
    while (ind_site >= nholes) and (ind_site < nholes + nbeta) :
        print_vect[vect[ind_site]] = '-'
        ind_site += 1
    return print_vect

def get_displayable_basis(basis):
    """
    Returns a list of printable lists representing the basis of our configuration

    Args:
        basis (class Basis): the basis to print
    Returns:
        print_basis (list<list<str>>): a printable representation of the basis
    """
    determinants = basis.determinants
    
    print_basis = []
    for det in determinants :
        print_basis.append(get_displayable_determinant(basis, det))
    return print_basis
    
def get_displayable_rectangular_lattice(configuration, vect):
    """
    Returns a displayable representation of a Slater determinant on a rectangular lattice occupied by holes or spins.
    Args:
        configuration (class Configuration): the configuration of our lattice
        vect (list<int>): a Slater determinant in this basis
    Returns:
        lattice (str): displayable representation of the Slater determinant
    """
    # Configuration
    nbh = configuration.nbh
    nbv = configuration.nbv
    nblocs_side = configuration.nblocs_side


    lattice = ''

    for ind_v in range(nblocs_side * nbv - 1, 0, -1) :
        for ind_h in range(nblocs_side * nbh - 1) :
            site = ind_v * nblocs_side * nbh + ind_h
            if get_spin_site(configuration, vect, site) == 1/2 :
                lattice += '\u2191--'
            elif get_spin_site(configuration, vect, site) == -1/2 :
                lattice += '\u2193--'
            else :
                lattice += '0--'
        if get_spin_site(configuration, vect, ind_v * nblocs_side * nbh + nblocs_side * nbh - 1) == 1/2 :
            lattice += '\u2191\n'
        elif get_spin_site(configuration, vect, ind_v * nblocs_side * nbh + nblocs_side * nbh - 1) == -1/2 :
            lattice += '\u2193\n'
        else :
            lattice += '0\n'
        for _ in range(nblocs_side * nbh - 1) :
            lattice += '|  '
        lattice += '|\n'
    for ind_h in range(nblocs_side * nbh - 1) :
        site = ind_h
        if get_spin_site(configuration, vect, site) == 1/2 :
            lattice += '\u2191--'
        elif get_spin_site(configuration, vect, site) == -1/2 :
            lattice += '\u2193--'
        else :
            lattice += '0--'
    if get_spin_site(configuration, vect, nblocs_side * nbh - 1) == 1/2 :
        lattice += '\u2191'
    elif get_spin_site(configuration, vect, nblocs_side * nbh - 1) == -1/2 :
        lattice += '\u2193'
    else :
        lattice += '0'
    return lattice

def get_displayable_outer_embedding_lattice(configuration, vect):
    """
    Returns a displayable representation of a Slater determinant on an outer embedding of a rectangular lattice, occupied by holes or spins.
    Args:
        configuration (class Configuration): the configuration of our lattice
        vect (list<int>): a Slater determinant in this basis
    Returns:
        lattice (str): displayable representation of the Slater determinant
    """
    # Configuration
    nbh = configuration.nbh
    nbv = configuration.nbv
    nblocs_side = configuration.nblocs_side
    nsites = nbh * nbv * (nblocs_side ** 2)


    lattice = '   '

    # Upper surrounding Néel lattice
    for ind in range((2 * nbh + nbv) * nblocs_side - 1, (nbh + nbv) * nblocs_side, -1):
        site = nsites + ind
        if get_spin_site(configuration, vect, site) == 1/2 :
            lattice += '\u2191--'
        elif get_spin_site(configuration, vect, site) == -1/2 :
            lattice += '\u2193--'
        else :
            lattice += '0--'
    site = (nbh + nbv) * nblocs_side + nsites
    if get_spin_site(configuration, vect, site) == 1/2 :
        lattice += '\u2191\n\n'
    elif get_spin_site(configuration, vect, site) == -1/2 :
        lattice += '\u2193\n\n'
    else :
        lattice += '0\n\n'
    # Lattice and lateral surrounding Néel lattices
    for ind_v in range(nblocs_side * nbv - 1, 0, -1) :
        site = 2 * (nbh + nbv) * nblocs_side - 1 - ind_v + nsites
        if get_spin_site(configuration, vect, site) == 1/2 :
            lattice += '\u2191  '
        elif get_spin_site(configuration, vect, site) == -1/2 :
            lattice += '\u2193  '
        else :
            lattice += '0  '

        for ind_h in range(nblocs_side * nbh - 1) :
            site = ind_v * nblocs_side * nbh + ind_h
            if get_spin_site(configuration, vect, site) == 1/2 :
                lattice += '\u2191--'
            elif get_spin_site(configuration, vect, site) == -1/2 :
                lattice += '\u2193--'
            else :
                lattice += '0--'
        if get_spin_site(configuration, vect, ind_v * nblocs_side * nbh + nblocs_side * nbh - 1) == 1/2 :
            lattice += '\u2191  '
        elif get_spin_site(configuration, vect, ind_v * nblocs_side * nbh + nblocs_side * nbh - 1) == -1/2 :
            lattice += '\u2193  '
        else :
            lattice += '0  '

        site = ind_v + nbh * nblocs_side + nsites
        if get_spin_site(configuration, vect, site) == 1/2 :
            lattice += '\u2191\n'
        elif get_spin_site(configuration, vect, site) == -1/2 :
            lattice += '\u2193\n'
        else :
            lattice += '0\n'
        
        lattice += '|  '
        for _ in range(nblocs_side * nbh) :
            lattice += '|  '
        lattice += '|\n'
    # Last row in the block
    site = 2 * (nbh + nbv) * nblocs_side - 1 + nsites
    if get_spin_site(configuration, vect, site) == 1/2 :
        lattice += '\u2191  '
    elif get_spin_site(configuration, vect, site) == -1/2 :
        lattice += '\u2193  '
    else :
        lattice += '0  '
    for site in range(nblocs_side * nbh - 1) :
        if get_spin_site(configuration, vect, site) == 1/2 :
            lattice += '\u2191--'
        elif get_spin_site(configuration, vect, site) == -1/2 :
            lattice += '\u2193--'
        else :
            lattice += '0--'
    if get_spin_site(configuration, vect, nblocs_side * nbh - 1) == 1/2 :
        lattice += '\u2191  '
    elif get_spin_site(configuration, vect, nblocs_side * nbh - 1) == -1/2 :
        lattice += '\u2193  '
    else :
        lattice += '0  '
    site = nbh * nblocs_side + nsites
    if get_spin_site(configuration, vect, site) == 1/2 :
        lattice += '\u2191\n\n   '
    elif get_spin_site(configuration, vect, site) == -1/2 :
        lattice += '\u2193\n\n   '
    else :
        lattice += '0\n\n   '
    # Lower surrounding Néel lattice
    for ind in range(nbh * nblocs_side - 1):
        site = nsites + ind
        if get_spin_site(configuration, vect, site) == 1/2 :
            lattice += '\u2191--'
        elif get_spin_site(configuration, vect, site) == -1/2 :
            lattice += '\u2193--'
        else :
            lattice += '0--'
    site = nbh * nblocs_side - 1 + nsites
    if get_spin_site(configuration, vect, site) == 1/2 :
        lattice += '\u2191'
    elif get_spin_site(configuration, vect, site) == -1/2 :
        lattice += '\u2193'
    else :
        lattice += '0'

    return lattice

def get_displayable_neighbor(configuration_A, configuration_B, vect_A, vect_B):
    """
    Returns a displayable representation of two neighboring Slater determinants on rectangular lattices.
    Args:
        configuration_A (class Configuration): the configuration of the first block
        configuration_B (class Configuration): the configuration of the second block
        vect_A (list<int>): the Slater determinant on the first block
        vect_B (list<int>): the Slater determinant on the second block
    Returns:
        lattice (str): the displayable representation of the two neighboring Slater determinants
    """
    lattice_A = get_displayable_rectangular_lattice(configuration_A, vect_A)
    lattice_B = get_displayable_rectangular_lattice(configuration_B, vect_B)
    lattice = ''

    lines_A, lines_B = [], []
    line_A = ''
    for char_A in lattice_A:
        if char_A != '\n':
            line_A += char_A
        else:
            lines_A.append(line_A)
            line_A = ''
    lines_A.append(line_A)
    line_B = ''
    for char_B in lattice_B:
        if char_B != '\n':
            line_B += char_B
        else:
            lines_B.append(line_B)
            line_B = ''
    lines_B.append(line_B)
    for line_A, line_B in zip(lines_A, lines_B):
        lattice += line_A + ' ' + line_B + '\n'
    return lattice


# Probabilities functions

def get_probability_spin_lattice(basis, vect, spin, reverse_spins=False):
    """ 
    Return the probability to find a particle of spin s (0 : hole, 1 : alpha, -1 : beta) \
        on each site of the vector vect with coordinates expressed in basis, \
            with site 0 in position south-west of the array.
    Args:
        basis (class Basis): the basis of our configuration
        vect (np.array): vector expressed in the basis of Slater determinants
        spin (int): either 0 for a hole, 1 for a spin up, and -1 for a spin down
        reverse_spins (bool): if True, reverse the z-component of every spin
    Returns:
        proba_sites (list<float>): probability of finding the particle spin on each site of the lattice in the state vect
    """
    if spin not in (-1, 0, 1):
        raise ValueError("Spin s must either be 0 : hole, 1 : alpha or -1 : beta")
    if reverse_spins:
        spin = -spin
    # Configuration
    configuration = basis.configuration
    nsites = configuration.nsites
    # Basis
    determinants = basis.determinants
    
    proba_sites = []
    for site in range(nsites) :
        proba_site = 0 # Probability of the presence of a spin on site ind_site
        for ind_basis, det in enumerate(determinants):
            proba_site += abs(vect[ind_basis, 0]) ** 2 * (get_spin_site(configuration, det, site) == spin/2)
        proba_sites.append(proba_site / (np.linalg.norm(vect) ** 2))
    return proba_sites


def get_spins_probabilitites_lattice(basis, vect, reverse_spins=False):
    """
    Returns the probability, on each site of the lattice in the specified state, to find the different particles.
    Args:
        basis (class Basis): the basis of our configuration
        vect (np.array): vector expressed in the basis of Slater determinants
        reverse_spins (bool): if True, reverse the z-component of every spin
    Returns:
        probabilities (dict<int : list<float>>): a dictionary, indexed by spins, with the corresponding \
             probability of presence on each site of the lattice
    """
    # Probabilities
    probabilities = {}
    for spin in (-1, 0, 1):
        probabilities[spin] = get_probability_spin_lattice(basis, vect, spin, reverse_spins)
    return probabilities


def get_probability_spin_rectangular_lattice(basis, vect, spin):
    """ 
    Return the probability to find a particle of spin s (0 : hole, 1 : alpha, -1 : beta) \
        on each site of the vector vect with coordinates expressed in basis, \
        with site 0 in position south-west of the array, in a periodic-type indexing.
    Args:
        basis (class Basis): the basis of our configuration
        vect (np.array): vector expressed in the basis of Slater determinants
        spin (int): either 0 for a hole, 1 for a spin up, and -1 for a spin down
    Returns:
        proba_lattice (np.array): probability of finding the particle spin on each site of the lattice in the state vect
    """
    if spin not in (-1, 0, 1):
        raise ValueError("Spin s must either be 0 : hole, 1 : alpha or -1 : beta")
    # Configuration
    configuration = basis.configuration
    if configuration.lattice_type != 'periodic':
        raise ValueError("lattice_type must be 'periodic'.")
    nbh = configuration.nbh
    nbv = configuration.nbv
    nblocs_side = configuration.nblocs_side
    
    proba_lattice = np.zeros((nblocs_side * nbv, nblocs_side * nbh))
    for ind_v in range(nblocs_side * nbv) :
        for ind_h in range(nblocs_side * nbh) :
            # Index of the site with coordinates (nblocs_side*nbh,nblocs_side*nbv) in the lattice
            ind_site = ind_v * nblocs_side * nbh + ind_h 
            
            proba_sites = get_probability_spin_lattice(basis, vect, spin)
            proba_lattice[nblocs_side * nbv - 1 - ind_v, ind_h] = proba_sites[ind_site] / (np.linalg.norm(vect) ** 2)
    return proba_lattice

        
def get_dominant_determinants(basis, vect, ndet): 
    """
    Return the ndet determinants with the highests coefficients in an eigenvalue vect
    Args:
        basis (class Basis): the basis of our configuration
        vect (np.array): vector expressed in the basis of Slater determinants basis
        ndet (int): the number of determinants, with the highests coefficients in vect, that we want to display
    Returns:
        list_coeffs (list<int>): the list of ndet indexes in the basis with the highests coefficients in vect
        list_dets (list<int>): the list of corresponding determinants in the basis
    """
    # Configuration
    configuration = basis.configuration
    nalpha = configuration.nalpha
    nbeta = configuration.nbeta
    nholes = configuration.nholes

    # Computation of the highests coefficients
    temp_vect = np.copy(vect)
    temp_vect.shape = (max(vect.shape),)
    temp_vect = temp_vect.tolist()
    
    abs_temp_vect = [abs(elt) for elt in temp_vect]
    sorted_abs_temp_vect = sorted(abs_temp_vect, reverse=True)[:min(ndet, len(temp_vect))]
    
    # Determination of the most dominant determinants and their indexes
    list_dets, list_coeffs = [], []
    for abs_elt in sorted_abs_temp_vect:
        list_coeffs.append(temp_vect[abs_temp_vect.index(abs_elt)])
        list_dets.append(get_hash_basis(nholes, nalpha, nbeta, abs_temp_vect.index(abs_elt)))

    return (list_coeffs, list_dets)

def get_displayable_dominant_determinants(basis, vect, ndet):
    """
    Return a displayable version of the ndet determinants with the highests coefficients in an eigenvalue vect
    Args:
        basis (class Basis): the basis of our configuration
        vect (np.array): vector expressed in the basis of Slater determinants basis
        ndet (int): the number of determinants, with the highests coefficients in vect, that we want to display
    Returns:
        list_coeffs (list<int>): the list of ndet indexes in the basis with the highests coefficients in vect
        list_det (list<str>): the list of corresponding determinants in the basis, as a displayable string
    """
    list_coeffs, list_dets = get_dominant_determinants(basis, vect, ndet)
    list_dets_displayable = [get_displayable_rectangular_lattice(basis, det) for det in list_dets]
    return (list_coeffs, list_dets_displayable)

### Symmetry functions

def rotate_determinant(nb_rot, configuration, det):
    """
    Rotates a square lattice by nb_rot * :math:`\\frac{\\pi}{2}` rad clockwise and returns the resulting rotated determinant
    Args:
        nb_rot (int): the number of :math:`\\frac{\\pi}{2}` rad rotations to perform on the configuration
        configuration (class Configuration): the configuration of our lattice
        det (list<int>): the determinant to be rotated
    Returns:
        det_rotated (list<int>): the rotated determinant
    """
    nbh = configuration.nbh
    nbv = configuration.nbv
    nblocs_side = configuration.nblocs_side
    nsites = nbh * nbv * (nblocs_side ** 2)

    if nb_rot % 4 == 1:
        det_rotated = reorder_basis_element(configuration, \
            [nbh * nblocs_side * (nbv * nblocs_side - 1 - site % (nbh * nblocs_side)) + site // (nbh * nblocs_side) for site in det])
    elif nb_rot % 4 == 2:
        det_rotated = reorder_basis_element(configuration, [nsites - 1 - site for site in det])
    elif nb_rot % 4 == 3:
        det_rotated = reorder_basis_element(configuration, \
            [(site % (nbh * nblocs_side)) * nbv * nblocs_side + nbh * nblocs_side - 1 - site // (nbh * nblocs_side)  for site in det])
    else:
        det_rotated = det[:]
    return det_rotated
        

### Hash functions
        
def project_hash(list_h, list_b): 
    """
    Takes two ordered lists of sites in [[0, nsites - 1]], and returns a list which, at each index ind, contains the number of elements \
        in list_h that are lower than or equal to list_b[ind]
    Args:
        list_h (list<int>): a list of sites (occupied by holes, in [[0,nsites-1]])
        list_b (list<int>): a list of sites (occupied by beta electrons, in [[0,nsites-1]])
    Returns: 
        list_b1 (list<int>): a list, which at each index ind, contains the number of elements in list_h that are lower than list_b[ind]
    """
    list_b1 = [sum([elt_h <= elt_b for elt_h in list_h]) for elt_b in list_b]
    return list_b1
    
def get_distance_hash(nsites, list_h, list_b): 
    """
    Compute the lexicographical distance between h and b as sublists of [[0,nsites-1]]
    Args:
        list_h (list<int>): a list of sites (occupied by holes, in [[0,nsites-1]])
        list_b (list<int>): a list of sites (occupied by beta electrons, in [[0,nsites-1]])
    Returns:
        dist (int): the distance between list_h and list_b, derived from lexicographical order
    """
    if list_b == []:
        return 0
    else:
        nbeta = len(list_b)
        list_bh = project_hash(list_h, list_b)
        list_b1 = [b - bh for b, bh in zip(list_b, list_bh)]

        nsites_free = nsites - len(list_h)
        
        dist = 0
        for ind in range(list_b1[0]):
            dist += spec.comb(nsites_free - ind - 1, nbeta - 1, exact=True)
        for ind_i in range(1, nbeta):
            for ind_j in range(list_b1[ind_i - 1] + 1, list_b1[ind_i]):
                dist += spec.comb(nsites_free - ind_j - 1, nbeta - ind_i - 1, exact=True)
        return dist
        
def compute_hash_table(configuration, det): 
    """
    Takes a vector det from the basis and returns its index in basis
    Args:
        configuration (class Configuration): the configuration of our lattice
        det (list<int>): an element of the basis
    Returns:
        hash_table (int): the index in the Hash table corresponding to det
    """
    # Configuration
    nsites = configuration.nsites
    nbeta = configuration.nbeta
    nholes = configuration.nholes
    
    list_h = det[: nholes]
    list_b = det[nholes : nholes + nbeta]
    
    return get_distance_hash(nsites, [], list_h) * spec.comb(nsites - nholes, nbeta, exact=True) + get_distance_hash(nsites, list_h, list_b)
    
def get_target_hash(nsites, size, ind): 
    """
    Returns the ind-th set, ordered lexicographically, of all the subsets of [[0, nsites - 1]] of length size
    Args:
        nsites (int): the number of sites
        size (int): the size of the desired list
        ind (int): the index of the desired list in the parts of [[0, nsites - 1]] of length size
    Returns:
        list_b (list<int>): the desired list
    """
    ind_temp = ind
    list_b = list(range(size))
    if size != 0 :
        while ind_temp > 0:
            rank = -1
            rest = nsites - 1 - list_b[rank]
            if rest > 0 :
                if ind_temp <= rest:
                    list_b[rank] += ind_temp
                    ind_temp = 0
                else:
                    ind_temp += -rest
                    list_b[rank] = nsites + rank
            else:
                while nsites + rank - list_b[rank] == 0 :
                    rank += -1
                list_b[rank] += 1
                for ind_j in range(rank + 1, 0):
                    list_b[ind_j] = list_b[ind_j - 1] + 1
                ind_temp += -1
    return list_b
 
def fusion_hash(list_h, list_b): 
    """
    Gets lists of positions for holes and beta electrons in [[0,nsites-1]], and integrates them together in the lattice
    Args : 
        list_h (list<int>): a list of sites (occupied by holes, in [[0,nsites-1]])
        list_b (list<int>): a list of sites (occupied by beta electrons, in [[0,nsites-1]])
    Returns : 
        list_f (list<int>): the list of indexes of holes and beta electrons integrated together
    """
    list_f = copy.deepcopy(list_h)
    ind_b=0
    while ind_b < len(list_b):
        elt_b = list_b[ind_b]
        for elt_h in list_h:
            elt_b += (elt_h <= elt_b)
        list_f.append(elt_b)
        ind_b += 1
    return list_f

def get_hash_basis(nholes, nalpha, nbeta, ind):  
    """
    Takes an integer ind and return the vector of index ind in the basis associated to the configuration (nholes,nalpha,nbeta)
    Args:
        nholes (int): number of holes in the lattice
        nalpha (int): number of spins up in the lattice
        nbeta (int): number of spins down in the lattice
        site (int): the index in the basis of the desired determinant
    Returns:
        vect (list<int>): a representation of the Slater determinant of index ind in the basis defined \
            with nholes holes, nalpha spins up and nbeta spins down
    """
    nsites = nholes + nalpha + nbeta
    index_h = ind // spec.comb(nsites - nholes, nbeta, exact=True) # We regroup configuration first by indentical hole-configurations
    index_b = ind % spec.comb(nsites - nholes, nbeta, exact=True) # Then we order them according to their beta electron-configurations
    
    list_h = get_target_hash(nsites, nholes, index_h) # Configuration of holes
    list_b = get_target_hash(nsites - nholes, nbeta, index_b) # Configuration of beta electrons
    
    list_hb = fusion_hash(list_h, list_b) # We build the configuration of holes and beta electrons in the lattice
    list_a = list(range(nsites)) # and fill the remainder sites with alpha electrons
    for site_occ in list_hb:
        list_a.remove(site_occ)
    return list_hb + list_a

### Tests get_index_in_basis

# bond_matrix = lF.build_rectangular_bond_matrix(nbh=1, nbv=1, nblocs_side=2)
# configuration = Configuration(bond_matrix, 1, 1)
# basis = construct_basis(configuration)
# for det in basis.determinants:
#     print(get_index_in_basis(configuration, det))

### Tests hash_table

# print("[bC] 0 : [0,1,2,3] : ",distHash(7,[],[0,1,2,3]),"----------------------------------")
# print("[bC] 1 : [0,1,2,4] : ",distHash(7,[],[0,1,2,4]),"----------------------------------")
# print("[bC] 2 : [0,1,2,5] : ",distHash(7,[],[0,1,2,5]),"----------------------------------")
# print("[bC] 3 : [0,1,2,6] : ",distHash(7,[],[0,1,2,6]),"----------------------------------")
# print("[bC] 4 : [0,1,3,4] : ",distHash(7,[],[0,1,3,4]),"----------------------------------")
# print("[bC] 5 : [0,1,3,5] : ",distHash(7,[],[0,1,3,5]),"----------------------------------")
# print("[bC] 6 : [0,1,3,6] : ",distHash(7,[],[0,1,3,6]),"----------------------------------")
# print("[bC] 7 : [0,1,4,5] : ",distHash(7,[],[0,1,4,5]),"----------------------------------")
# print("[bC] 8 : [0,1,4,6] : ",distHash(7,[],[0,1,4,6]),"----------------------------------")
# print("[bC] 9 : [0,1,5,6] : ",distHash(7,[],[0,1,5,6]),"----------------------------------")
# print("[bC] 10 : [0,2,3,4] : ",distHash(7,[],[0,2,3,4]),"----------------------------------")
# print("[bC] 11 : [0,2,3,5] : ",distHash(7,[],[0,2,3,5]),"----------------------------------")
# print("[bC] 12 : [0,2,3,6] : ",distHash(7,[],[0,2,3,6]),"----------------------------------")
# print("[bC] 13 : [0,2,4,5] : ",distHash(7,[],[0,2,4,5]),"----------------------------------")
# print("[bC] 14 : [0,2,4,6] : ",distHash(7,[],[0,2,4,6]),"----------------------------------")
# print("[bC] 15 : [0,2,5,6] : ",distHash(7,[],[0,2,5,6]),"----------------------------------")
# print("[bC] 16 : [0,3,4,5] : ",distHash(7,[],[0,3,4,5]),"----------------------------------")
# print("[bC] 17 : [0,3,4,6] : ",distHash(7,[],[0,3,4,6]),"----------------------------------")
# print("[bC] 18 : [0,3,5,6] : ",distHash(7,[],[0,3,5,6]),"----------------------------------")
# print("[bC] 19 : [0,4,5,6] : ",distHash(7,[],[0,4,5,6]),"----------------------------------")
# print("[bC] 20 : [1,2,3,4] : ",distHash(7,[],[1,2,3,4]),"----------------------------------")
# print("[bC] 21 : [1,2,3,5] : ",distHash(7,[],[1,2,3,5]),"----------------------------------")


# print("[bC] 1 : [0], [1,2,3,4,5,7] : ",distHash(9,[0],[1,2,3,4,5,7]),"----------------------------------")
# print("[bC] 1 : [0,1,2,3,4,6] : ",distHash(8,[],[0,1,2,3,4,6]),"----------------------------------")
# print("[bC] 2 : [0], [1,2,3,4,5,8] : ",distHash(9,[0],[1,2,3,4,5,8]),"----------------------------------")
# print("[bC] 2 : [0,1,2,3,4,7] : ",distHash(8,[],[0,1,2,3,4,7]),"----------------------------------")
# print("[bC] 20 : [0], [1,4,5,6,7,8] : ",distHash(9,[0],[1,4,5,6,7,8]),"----------------------------------")
# print("[bC] 21 : [0], [2,3,4,5,6,7] : ",distHash(9,[0],[2,3,4,5,6,7]),"----------------------------------")

### Tests hashBasis

# print("[bC] targetHash(3,2,0) : ",targetHash(3,2,0))
# print("[bC] targetHash(3,2,1) : ",targetHash(3,2,1))
# print("[bC] targetHash(3,2,2) : ",targetHash(3,2,2)) # Ok

# print("[bC] targetHash(4,2,0) : ",targetHash(4,2,0))
# print("[bC] targetHash(4,2,1) : ",targetHash(4,2,1))
# print("[bC] targetHash(4,2,2) : ",targetHash(4,2,2))
# print("[bC] targetHash(4,2,3) : ",targetHash(4,2,3))
# print("[bC] targetHash(4,2,4) : ",targetHash(4,2,4))
# print("[bC] targetHash(4,2,5) : ",targetHash(4,2,5)) # Ok

# print("[bC] targetHash(6,3,0) : ",targetHash(6,3,0),"--------------------------------")
# print("[bC] targetHash(6,3,1) : ",targetHash(6,3,1),"--------------------------------")
# print("[bC] targetHash(6,3,2) : ",targetHash(6,3,2),"--------------------------------")
# print("[bC] targetHash(6,3,3) : ",targetHash(6,3,3),"--------------------------------")
# print("[bC] targetHash(6,3,4) : ",targetHash(6,3,4),"--------------------------------")
# print("[bC] targetHash(6,3,5) : ",targetHash(6,3,5),"--------------------------------")
# print("[bC] targetHash(6,3,6) : ",targetHash(6,3,6),"--------------------------------")

# Itest11344=[]
# I11344=[i for i in range(lF.nbConfig(9,4,4))]
# Basis11344=baseConstruction(1,1,3,4,4)
# for i in range(lF.nbConfig(9,4,4)):
#     #print("[bC] targetHash(6,3,",i,") : ",targetHash(6,3,i),"--------------------------------")
#     print("[bC] hashBasis(1,4,4,",i,") : ",vectDisplay(baseConstruction(1,1,3,4,4),hashBasis(1,4,4,i)),"--------------------------------")
#     Itest11344.append(hashTable(Basis11344,hashBasis(1,4,4,i)))
# print(Itest11344==I11344)

# Test for H421
# configuration = Configuration(nbh=2, nbv=2, nblocs_side=1, nholes=1, nbeta=1)
# basis=construct_basis(configuration)

# vect = np.array(list(range(lF.compute_dimension(4,2,1))))
# print(get_dominant_determinants(basis,vect,4))

# print("[bC] basis = ",basis)
# hash_basis_list=[]
# for i in range(12):
#     print("[bC] i = ",i) #####
#     hashi = get_hash_basis(1,2,1,i)
#     hash_basis_list.append(hashi)
#     print("[bC] hashBasis(1,2,1,",i,") : ",get_displayable_determinant(construct_basis(configuration),hashi))
# print("[bC] hashBasisList = ",hash_basis_list)
# print("-----------------------------------------------------------------------------------")
# Test for H402
# Basis=baseConstruction(1,1,2,0,2)
# basis=getBasis(Basis)
# print("[bC] basis = ",basis)
# hashBasisList=[]
# for i in range(6):
#     #print("[bC] i = ",i) #####
#     hashBasisList.append(hashBasis(2,0,2,i))
#     #print("[bC] hashBasis(2,0,2,",i,") : ",vectDisplay(baseConstruction(1,1,2,0,2),hashBasis(2,0,2,i)))
# print("[bC] hashBasisList = ",hashBasisList)

### Tests holeProbabilityLattice

# nbh=1
# nbv=1
# nBlocSide=2
# nalpha=2
# nbeta=1
# 
# Basis=baseConstruction(nbh,nbv,nBlocSide,nalpha,nbeta)
# basis=getBasis(Basis)

# np.random.seed(2010)
# vect=np.random.randint(10, size=(len(basis),1))

# vect = np.array([[2],[2],[2],[1],[1],[1],[1],[1],[1],[2],[2],[2]])

# vect1 = np.array([[0.35428435],[0.0366736],[-0.39095795],[0.37875945],[-0.23988441],[-0.13887505],[0.13887505],[0.23988441],[-0.37875945],[0.39095795],[-0.0366736],[-0.35428435]])
# vect2 = np.array([[0.13887505],[-0.37875945],[0.23988441],[0.0366736],[-0.39095795],[0.35428435],[-0.35428435],[0.39095795],[-0.0366736],[-0.23988441],[0.37875945],[-0.13887505]])

# vectp = (1/mp.sqrt(2))*(vect1+vect2)
# vectm = (1/mp.sqrt(2))*(vect1-vect2)


# for j in range(len(basis)):
#     vectj=np.zeros((len(basis),1))
#     vectj[j,0]=1
#     print("----------------------------------------")
#     print("[bC] vect = ",vectDisplay(Basis,basis[j]))
#     print("[bC] P =\n",holeProbabilityLattice(vectj,Basis))
# print("[bC] P =\n",holeProbabilityLattice(vect,Basis))

### Tests spinProbabilityLattice

# nbh=1
# nbv=1
# nBlocSide=2
# nalpha=2
# nbeta=1

# Basis=baseConstruction(nbh,nbv,nBlocSide,nalpha,nbeta)
# basis=getBasis(Basis)

# np.random.seed(2010)
# vect=np.random.randint(10, size=(len(basis),1))

# vect = np.array([[2],[2],[2],[1],[1],[1],[1],[1],[1],[2],[2],[2]])

# vect1 = np.array([[0.35428435],[0.0366736],[-0.39095795],[0.37875945],[-0.23988441],[-0.13887505],[0.13887505],[0.23988441],[-0.37875945],[0.39095795],[-0.0366736],[-0.35428435]])
# vect2 = np.array([[0.13887505],[-0.37875945],[0.23988441],[0.0366736],[-0.39095795],[0.35428435],[-0.35428435],[0.39095795],[-0.0366736],[-0.23988441],[0.37875945],[-0.13887505]])

# vectp = (1/mp.sqrt(2))*(vect1+vect2)
# vectm = (1/mp.sqrt(2))*(vect1-vect2)

# for j in range(len(basis)):
#     vectj=np.zeros((len(basis),1))
#     vectj[j,0]=1
#     print("----------------------------------------")
#     print("[bC] vect = ",vectDisplay(Basis,basis[j]))
#     print("[bC] Palpha =\n",spinProbabilityLattice(vectj,Basis,1))
#     print("[bC] Pbeta =\n",spinProbabilityLattice(vectj,Basis,-1))
#print("[bC] P =\n",holeProbabilityLattice(vect,Basis))

# Tests latticeDisplay
# configuration = configuration(1,1,3,4,4)
# Basis = baseConstruction(configuration)
# vect = [4,1,3,5,7,0,2,6,8]
# print(vectDisplay(Basis,vect))
# print(latticeDisplay(Basis,vect))

# Tests dominantsDeterminants
# configuration = configuration(1,1,2,2,1)
# Basis = baseConstruction(configuration)
# basis = Basis.basis
# for i in range(len(basis)):
#     vect = hashBasis(1,2,1,i)
#     print(latticeDisplay(Basis,vect))
#     print("\n\n")