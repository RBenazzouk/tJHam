import latticeFunctions as lF
import baseConstruction as bC
import hamiltonianConstruction as hC
import copy
import scipy.sparse as sp
import numpy as np
import numpy.linalg as lin

import math as mp

from tqdm import tqdm, trange # Install tqdm via pip

class Interactions :
    """
    Defines the coefficients of each interaction
    """
    def __init__(self, t, J, Jhpar, Jhper, hSDpar, hSDper, tNNN, tnNNN, VNN, VNNNper):
        """
        Args:
            t (float): the nearest-neighbor hopping integral
            J (float): spin coupling in undoped lattice
            Jhpar (float): spin coupling with a hole colinear to the coupled spins
            Jhper (float): spin coupling with a hole perpendicular to the axis of the coupled spins
            hSDpar (float): singlet-displacement along the axis of the singlet
            hSDper (float): singlet displacement in a direction orthogonal to the axis of the singlet
            tNNN (float): next nearest-neighbor hopping integral with othogonal consecutive jump directions
            tnNNN (float): next nearest-neighbor hopping integral with colinear consecutive jump directions
            VNN (float): hole-hole nearest-neighbor repulsion
            VNNNper (float): hole-hole next-nearest neighbor repulsion, with holes opposed on a single tile (ie not aligned along a bond in the lattice)
        """
        self.t = t
        self.J = J
        self.Jhpar = Jhpar
        self.Jhper = Jhper
        self.hSDpar = hSDpar
        self.hSDper = hSDper
        self.tNNN = tNNN
        self.tnNNN = tnNNN
        self.VNN = VNN
        self.VNNNper = VNNNper

### Interactions #########################################################################################################


# Hopping integral

def compute_hopping_integral(configuration, interactions, det):
    """
    Compute the action of the hopping integral on a Slater determinant det in the lattice defined by configuration
    Args:
        configuration (class baseConstruction.Configuration): the configuration of the lattice
        interactions (class interactions.Interactions): the coefficients for any interaction in the Hamiltonian
        det (list<int>): the specified Slater determinant upon which acts the present operator
    Returns:
        images (list<list<int>>): the list of vectors resulting of the action of the hopping integral on det
    """
    # Configuration
    nholes = configuration.nholes
    sites_restriction = configuration.sites_restriction
    bond_matrix = configuration.bond_matrix

    images = []

    if nholes > 0 :
        sites_holes = bC.get_indexes_of(configuration, det, 0)
        for site_hole in sites_holes:
            hole_neighbors = lF.get_neighbors(site_hole, bond_matrix, sites_restriction)
            for neighbor in hole_neighbors:
                if not(neighbor in sites_holes):
                    ind_hole = det.index(site_hole)
                    ind_neighbor = det.index(neighbor)
                    temp = copy.deepcopy(det)
                    temp[ind_hole], temp[ind_neighbor] = temp[ind_neighbor], temp[ind_hole]
                    images.append(bC.reorder_basis_element(configuration, temp))
    return images

def add_hopping_integral(ham, basis, interactions): 
    """
    Fills the Hamiltonian ham with the t-hopping integral terms
    Args:
        ham (scipy.sparse.coo_matrix): our Hamiltonian
        basis (class baseConstruction.Basis): the basis of our configuration
        interactions (class interactions.Interactions): the coefficients for any interaction in our Hamiltonian
    """
    # Configuration
    configuration = basis.configuration
    nholes = configuration.nholes
    # Basis
    determinants = basis.determinants
    # Interactions
    t = interactions.t

    if nholes > 0:
        for ind_vect, vect in tqdm(enumerate(determinants), total=len(determinants)):
            images = compute_hopping_integral(configuration, interactions, vect)
            for temp in images :
                ind_temp = bC.get_index_in_basis(configuration, temp)
                hC.add_ham_coo(ham, t, ind_vect, ind_temp)


def compute_next_hopping_integral(configuration, interactions, vect):
    """
    Compute the action of the next-neighbor hopping integral on a Slater determinant vect in the lattice defined by configuration
    Args:
        configuration (class baseConstruction.Configuration): the configuration of the lattice
        interactions (class interactions.Interactions): the coefficients for any interaction in the Hamiltonian
        vect (list<int>): the specified vector upon which acts the present operator
    Returns:
        images_col (list<list<int>>): the list of vectors resulting of the action of the \
            colinear next-neighbor hopping integral on vect
        images_ortho (list<list<int>>): the list of vectors resulting of the action of the \
            orthogonal next-neighbor hopping integral on vect
    """
    # Configuration
    nholes = configuration.nholes
    sites_restriction = configuration.sites_restriction
    bond_matrix = configuration.bond_matrix

    images_col = [] # tNNN
    images_ortho = [] # tnNNN

    if nholes > 0 :
        next_neighbors_jump_sites = []
        sites_holes = bC.get_indexes_of(configuration, vect, 0) # list of the sites containing holes in vect
        for site_hole in sites_holes:
            neighbors=lF.get_neighbors(site_hole, bond_matrix, [])
            for neighbor in neighbors:
                sites_next_neigh_electrons = list(set(bC.get_indexes_of(configuration, vect, 1, 1) \
                    + bC.get_indexes_of(configuration, vect, -1, 1)) & \
                    set(lF.get_neighbors(neighbor, bond_matrix, sites_restriction))) # list of next-neighboring electrons
                if sites_next_neigh_electrons:
                    for site_next_neigh_elec in sites_next_neigh_electrons:
                        ind_hole = vect.index(site_hole)
                        ind_next_neigh_elec = vect.index(site_next_neigh_elec)

                        if not [site_hole, site_next_neigh_elec] in next_neighbors_jump_sites:
                            next_neighbors_jump_sites.append([site_hole, site_next_neigh_elec])
                        
                            temp = copy.deepcopy(vect)      
                            temp[ind_hole] = site_next_neigh_elec
                            temp[ind_next_neigh_elec] = site_hole

                            if bC.is_bond_linear(configuration, site_next_neigh_elec, neighbor, site_hole):
                                images_col.append(bC.reorder_basis_element(configuration, temp))
                            else:
                                images_ortho.append(bC.reorder_basis_element(configuration, temp))
    return images_col, images_ortho

def add_next_hopping_integral(ham, basis, interactions): 
    """
    Fills the Hamiltonian ham with the next-neighbor tNNN-jump terms
    Args:
        ham (scipy.sparse.coo_matrix): our Hamiltonian
        basis (class baseConstruction.Basis): the basis of our configuration
        interactions (class interactions.Interactions): the coefficients for any interaction in our Hamiltonian
    """
    # Configuration
    configuration = basis.configuration
    nholes = configuration.nholes

    # Interactions
    tNNN = interactions.tNNN
    tnNNN = interactions.tnNNN

    if nholes > 0:
        determinants = basis.determinants
        for ind_vect, vect in tqdm(enumerate(determinants), total=len(determinants)):
            images_col, images_ortho = compute_next_hopping_integral(configuration, interactions, vect)     
            for temp_col in images_col:      
                ind_temp_col = bC.get_index_in_basis(configuration, temp_col)
                hC.add_ham_coo(ham, tnNNN, ind_vect, ind_temp_col)

            for temp_ortho in images_ortho:
                ind_temp_ortho = bC.get_index_in_basis(configuration, temp_ortho)
                hC.add_ham_coo(ham, tNNN, ind_vect, ind_temp_ortho)

# Spin interactions

def compute_static_spin_interaction(configuration, interactions, vect):
    """
    Compute the action of the static spin interaction on a Slater determinant vect in the lattice defined by configuration
    Args:
        configuration (class baseConstruction.Configuration): the configuration of the lattice
        interactions (class interactions.Interactions): the coefficients for any interaction in the Hamiltonian
        vect (list<int>): the specified vector upon which acts the present operator
    Returns:
        sz (float): the value of the static spins interaction
    """
    # Configuration
    sites_restriction = configuration.sites_restriction
    bond_matrix = configuration.bond_matrix
    # Interactions
    J = interactions.J
    Jhpar = interactions.Jhpar
    Jhper = interactions.Jhper

    sz = 0
    bond_list = lF.build_bond_list(bond_matrix, sites_restriction)
    for bond in bond_list:
        spin_0 = bC.get_spin_site(configuration, vect, bond[0])
        spin_1 = bC.get_spin_site(configuration, vect, bond[1])

        if spin_0 * spin_1 != 0:

            # Computation of J depending on potential neighboring electrons
            ind_0 = min(bond[0],bond[1])
            ind_1 = max(bond[0],bond[1])

            ind_neigh_0 = [neighbor for neighbor in lF.get_neighbors(ind_0, bond_matrix, []) if neighbor != ind_1]
            ind_neigh_1 = [neighbor for neighbor in lF.get_neighbors(ind_1, bond_matrix, []) if neighbor != ind_0]

            ind_neigh_hole_0 = list(set(ind_neigh_0) & set(bC.get_indexes_of(configuration, vect, 0))) # List of holes neighbors of ind_0
            ind_neigh_hole_1 = list(set(ind_neigh_1) & set(bC.get_indexes_of(configuration, vect, 0))) # List of holes neighbors of ind_1

            if (ind_neigh_hole_0 or ind_neigh_hole_1) \
                and True in [bC.is_bond_linear(configuration, ind_1, ind_0, ind_hole) for ind_hole in ind_neigh_hole_0 + ind_neigh_hole_1]: \
                    # Check if a neighboring hole is aligned with the pair   # A adapter à lattice_type = cluster
                sz += Jhpar * (spin_0 * spin_1 - (1/4) * (spin_0 * spin_1 != 0)) # Result of Si_z.Sj_z
            elif ind_neigh_hole_0 or ind_neigh_hole_1 \
                and True not in [bC.is_bond_linear(configuration, ind_1, ind_0, ind_hole) for ind_hole in ind_neigh_hole_0 + ind_neigh_hole_1] : \
                    # Check if a neighboring hole is orthogonal to the axis of the pair
                sz += Jhper * (spin_0 * spin_1 - (1/4) * (spin_0 * spin_1 != 0)) # Result of Si_z.Sj_z
            else:
                sz += J * (spin_0 * spin_1 - (1/4) * (spin_0 * spin_1 != 0)) # Result of Si_z.Sj_z
    return sz
                        
def add_static_spin_interaction(ham, basis, interactions): 
    """
    Fills the Hamiltonian ham with the J/4-interaction terms
    Args:
        ham (scipy.sparse.coo_matrix): our Hamiltonian
        basis (class baseConstruction.Basis): the basis of our configuration
        interactions (class interactions.Interactions): the coefficients for any interaction in our Hamiltonian
    """
    # Configuration
    configuration = basis.configuration
    # Basis
    determinants = basis.determinants

    for ind_vect, vect in tqdm(enumerate(determinants), total=len(determinants)):
        sz = compute_static_spin_interaction(configuration, interactions, vect)
        hC.add_ham_coo(ham, sz, ind_vect, ind_vect)


def compute_spin_flip(configuration, interactions, vect):
    """
    Compute the action of the spin flip operator on a Slater determinant vect in the lattice defined by configuration
    Args:
        configuration (class baseConstruction.Configuration): the configuration of the lattice
        interactions (class interactions.Interactions): the coefficients for any interaction in the Hamiltonian
        vect (list<int>): the specified vector upon which acts the present operator
    Returns:
        images_par (list<list<int>>): the list of vectors resulting of a spin flip on vect with a hole colinear to the spins axis
        images_per (list<list<int>>): the list of vectors resulting of a spin flip on vect with a hole orthogonal to the spins axis
        images_undoped (list<list<int>>): the list of vectors resulting of a spin flip on vect in an undoped environment
    """
    # Configuration
    sites_restriction = configuration.sites_restriction
    bond_matrix = configuration.bond_matrix

    bond_list = lF.build_bond_list(bond_matrix, sites_restriction)

    images_undoped, images_par, images_per = [], [], []
    
    for bond in bond_list:
        spin_0 = bC.get_spin_site(configuration, vect, bond[0])
        spin_1 = bC.get_spin_site(configuration, vect, bond[1])
        if spin_0 * spin_1 == -1/4: # Opposite spins on the bond [s0 - s1]
            temp = copy.deepcopy(vect)
            # Spin flip
            temp[vect.index(bond[0])], temp[vect.index(bond[1])] = temp[vect.index(bond[1])], temp[vect.index(bond[0])]

            # Computation of J depending on potentially neighboring electrons
            ind_0 = min(bond[0], bond[1])
            ind_1 = max(bond[0], bond[1])

            ind_neigh_0 = lF.get_neighbors(ind_0, bond_matrix)
            ind_neigh_1 = lF.get_neighbors(ind_1, bond_matrix)


            ind_neigh_hole_0 = list(set(ind_neigh_0) & set(bC.get_indexes_of(configuration, vect, 0)))
            ind_neigh_hole_1 = list(set(ind_neigh_1) & set(bC.get_indexes_of(configuration, vect, 0)))
            if (ind_neigh_hole_0 or ind_neigh_hole_1) \
                and (ind_1 - ind_0 in [abs(ind_0 - ind_hole) for ind_hole in ind_neigh_hole_0] + \
                [abs(ind_1 - ind_hole) for ind_hole in ind_neigh_hole_1]): # Check if a neighboring hole is aligned with the pair
                
                images_par.append(bC.reorder_basis_element(configuration, temp))

            elif ind_neigh_hole_0 or ind_neigh_hole_1: # Check if a neighboring hole is orthogonal to the axis of the pair
                images_per.append(bC.reorder_basis_element(configuration, temp))
            
            else:
                images_undoped.append(bC.reorder_basis_element(configuration, temp))
    return (images_par, images_per, images_undoped)

def add_spin_flip(ham, basis, interactions): 
    """
    Fills the Hamiltonian ham with the J/2-interaction terms
    Args:
        ham (scipy.sparse.coo_matrix): our Hamiltonian
        basis (class baseConstruction.Basis): the basis of our configuration
        interactions (class interactions.Interactions): the coefficients for any interaction in our Hamiltonian
        
    """
    # Configuration
    configuration = basis.configuration
    # Basis
    determinants = basis.determinants    
    # Interactions
    J = interactions.J
    Jhpar = interactions.Jhpar
    Jhper = interactions.Jhper

    for ind_vect, vect in tqdm(enumerate(determinants), total=len(determinants)):
        images_par, images_per, images_undoped = compute_spin_flip(configuration, interactions, vect)

        for temp_par in images_par:
            ind_temp_par = bC.get_index_in_basis(configuration, temp_par)
            hC.add_ham_coo(ham, Jhpar/2, ind_vect, ind_temp_par)
        for temp_per in images_per:
            ind_temp_per = bC.get_index_in_basis(configuration, temp_per)
            hC.add_ham_coo(ham, Jhper/2, ind_vect, ind_temp_per)
        for temp_undoped in images_undoped:
            ind_temp_undoped = bC.get_index_in_basis(configuration, temp_undoped)
            hC.add_ham_coo(ham, J/2, ind_vect, ind_temp_undoped)


# Singlet displacement

def compute_singlet_displacement(configuration, interactions, vect):
    """
    Compute the action of the singlet displacement operator on a determinant vect in the lattice defined by configuration
    Args:
        configuration (class baseConstruction.Configuration): the configuration of the lattice
        interactions (class interactions.Interactions): the coefficients for any interaction in the Hamiltonian
        vect (list<int>): the specified vector upon which acts the present operator
    Returns:
        images_par (list<list<list<int>>>): the list of vectors resulting of a singlet displacement on vect with a hole colinear to the spins axis
        images_per (list<list<list<int>>>): the list of vectors resulting of a singlet displacement on vect with a hole orthogonal to the spins axis
    """
    # Configuration
    sites_restriction = configuration.sites_restriction
    bond_matrix = configuration.bond_matrix     

    images_par, images_per = [], []

    # In vect, we look for holes in the lattice, then for neighbouring singlets
    sites_holes = bC.get_indexes_of(configuration, vect, 0) # sites with holes
    for site_hole in sites_holes:
        
        sites_neigh = lF.get_neighbors(site_hole, bond_matrix, sites_restriction)
        sites_neigh_electrons = list(set(bC.get_indexes_of(configuration, vect, 1, 1) \
            + bC.get_indexes_of(configuration, vect, -1, 1)) & set(sites_neigh)) # indexes of neighboring electrons

        for site_neigh_elec in sites_neigh_electrons:

            spin_site = bC.get_spin_site(configuration, vect, site_neigh_elec)
            site_next_neigh = lF.get_neighbors(site_neigh_elec, bond_matrix, sites_restriction)
            sites_next_neigh_electrons = list(set(bC.get_indexes_of(configuration, vect, -2 * spin_site, 1)) \
                & set(site_next_neigh)) # index of electrons of opposed spin neighbouring the previous electron

            if sites_next_neigh_electrons:
                for site_next_neigh_elec in sites_next_neigh_electrons:
                    ind_hole = vect.index(site_hole)
                    ind_neigh_elec = vect.index(site_neigh_elec)
                    ind_next_neigh_elec = vect.index(site_next_neigh_elec)
                    
                    temp_1 = copy.deepcopy(vect)
                    temp_2 = copy.deepcopy(vect)
                    
                    temp_1[ind_hole] = site_next_neigh_elec
                    temp_1[ind_neigh_elec] = site_hole
                    temp_1[ind_next_neigh_elec] = site_neigh_elec
                    
                    temp_2[ind_hole] = site_next_neigh_elec
                    temp_2[ind_neigh_elec] = site_neigh_elec
                    temp_2[ind_next_neigh_elec] = site_hole
                    
                    if bC.is_bond_linear(configuration, site_next_neigh_elec, site_neigh_elec, site_hole): 
                        images_par += [[bC.reorder_basis_element(configuration, temp_1), bC.reorder_basis_element(configuration, temp_2)]]
                    else:
                        images_per += [[bC.reorder_basis_element(configuration, temp_1), bC.reorder_basis_element(configuration, temp_2)]]
    return (images_par, images_per)
                
def add_singlet_displacement(ham, basis, interactions): 
    """
    Fills the Hamiltonian ham with the singlet displacement terms
    Args:
        ham (scipy.sparse.coo_matrix): our Hamiltonian
        basis (class baseConstruction.Basis): the basis of our configuration
        interactions (class interactions.Interactions): the coefficients for any interaction in our Hamiltonian
    """
    # Configuration
    configuration = basis.configuration
    
    # Basis
    determinants = basis.determinants
    # Interactions
    hSDpar = interactions.hSDpar
    hSDper = interactions.hSDper

    for ind_vect, vect in tqdm(enumerate(determinants), total=len(determinants)):
        images_par, images_per = compute_singlet_displacement(configuration, interactions, vect)
        for temp_1, temp_2 in images_par:
            ind_temp_1, ind_temp_2 = bC.get_index_in_basis(configuration, temp_1), bC.get_index_in_basis(configuration, temp_2)
            hC.add_ham_coo(ham, hSDpar, ind_vect, ind_temp_1)
            hC.add_ham_coo(ham, -hSDpar, ind_vect, ind_temp_2)
        for temp_1, temp_2 in images_per:
            ind_temp_1, ind_temp_2 = bC.get_index_in_basis(configuration, temp_1), bC.get_index_in_basis(configuration, temp_2)
            hC.add_ham_coo(ham, hSDper, ind_vect, ind_temp_1)
            hC.add_ham_coo(ham, -hSDper, ind_vect, ind_temp_2)

# Hole repulsion

def compute_hole_repulsion(configuration, interactions, vect):
    """
    Compute the action of the static hole repulsion interaction on a Slater determinant vect \
        in the lattice defined by configuration
    Args:
        configuration (class baseConstruction.Configuration): the configuration of the lattice
        interactions (class interactions.Interactions): the coefficients for any interaction in the Hamiltonian
        vect (list<int>): the specified vector upon which acts the present operator
    Returns:
        hole_repulsion (float): the value of the static repulsion between holes
    """
    # Configuration
    nholes = configuration.nholes
    sites_restriction = configuration.sites_restriction
    bond_matrix = configuration.bond_matrix
    # Interactions
    VNN = interactions.VNN
    VNNNper = interactions.VNNNper

    hole_repulsion = 0
    if nholes > 1:
        sites_holes = bC.get_indexes_of(configuration, vect, 0)
        holes_bond_list = []
        holes_next_bond_list = []
        for ind_hole, site_hole in enumerate(sites_holes[:-1]):
            neighboring_holes = list(set(lF.get_neighbors(site_hole, bond_matrix, sites_restriction)).intersection(set(sites_holes[ind_hole:])))
            if neighboring_holes:
                holes_bond_list += [[site_hole, neighboring_hole] for neighboring_hole in neighboring_holes]
            diagonal_next_neighboring_holes = list(set(lF.get_diagonal_second_neighbors(configuration, site_hole)).intersection(set(sites_holes[ind_hole:])))
            if diagonal_next_neighboring_holes:
                holes_next_bond_list += [[site_hole, diagonal_next_neighboring_hole] for diagonal_next_neighboring_hole in diagonal_next_neighboring_holes]
        hole_repulsion = len(holes_bond_list) * VNN + len(holes_next_bond_list) * VNNNper
    return hole_repulsion
                                    
def add_hole_repulsion(ham, basis, interactions):
    """
    Fills the Hamiltonian ham with the hole-hole nearest-neighbor repulsion terms
    Args:
        ham (scipy.sparse.coo_matrix): our Hamiltonian
        basis (class baseConstruction.Basis): the basis of our configuration
        interactions (class interactions.Interactions): the coefficients for any interaction in our Hamiltonian
    """
    # Configuration
    configuration = basis.configuration
    # Basis
    determinants = basis.determinants
    for ind_vect, vect in tqdm(enumerate(determinants), total=len(determinants)):
        hole_repulsion = compute_hole_repulsion(configuration, interactions, vect)
        hC.add_ham_coo(ham, hole_repulsion, ind_vect, ind_vect)
            

# Energy

def compute_energy(configuration, interactions, det):
    """
    Computes the energy of the Slater determinant det in the lattice defined by configuration with \
        the Hamiltonian defined by interactions
    Args:
        configuration (class baseConstruction.Configuration): the configuration of the lattice
        interactions (class interactions.Interactions): the coefficients for any interaction in the Hamiltonian
        det (list<int>): the specified vector upon which acts the present operator
    Returns:
        energy (float): the energy of the determinant in this configuration
    """
    energy = compute_static_spin_interaction(configuration, interactions, det) \
        + compute_hole_repulsion(configuration, interactions, det)
    return energy

# Dynamical interactions

def compute_dynamic_interactions(configuration, interactions, bool_interactions, det):
    """
    Computes the dynamic interactions of the Slater determinant det in the lattice defined by configuration with \
        the others Slater determinants of the basis according to the Hamiltonian defined by interactions
    Args:
        configuration (class baseConstruction.Configuration): the configuration of the lattice
        interactions (class interactions.Interactions): the coefficients for any interaction in the Hamiltonian
        bool_interactions (class initialComputations.BoolInteractions): booleans representing the selection of implemented interactions
        det (list<int>): the specified vector upon which acts the present operator
    Returns:
        dynamic_interactions (dict): the determinants coupled with det via the Hamiltonian, as keys of the dictionnary, \
            with the coupling coefficient as the corresponding value
    """
    # Interactions
    t = interactions.t
    J = interactions.J
    Jhpar = interactions.Jhpar
    Jhper = interactions.Jhper
    hSDpar = interactions.hSDpar
    hSDper = interactions.hSDper
    tNNN = interactions.tNNN
    tnNNN = interactions.tnNNN
    # Interactions booleans
    bt = bool_interactions.bt
    bij = bool_interactions.bij
    bSD = bool_interactions.bSD
    btN = bool_interactions.btN

    dynamic_interactions = {}
    events = []
    if bt:
        t_coupled = compute_hopping_integral(configuration, interactions, det)
        events.append((t_coupled, t))
    if btN:
        tN_col_coupled, tN_ortho_coupled = compute_next_hopping_integral(configuration, interactions, det)
        events += [(tN_col_coupled, tnNNN), (tN_ortho_coupled, tNNN)]
        print("[itr] det = ", det)
        print("[itr] tN_col_coupled = ", tN_col_coupled)
        print("[itr] tN_ortho_coupled = ", tN_ortho_coupled)
    if bij:
        Jcol_coupled, Jortho_coupled, J_coupled = compute_spin_flip(configuration, interactions, det)
        events += [(Jcol_coupled, Jhpar), (Jortho_coupled, Jhper), (J_coupled, J)]
    if bSD:
        hpar_coupled, hper_coupled = compute_singlet_displacement(configuration, interactions, det)
        singlet_events = [(hpar_coupled, hSDpar), (hper_coupled, hSDper)]

    for couplings, inter in events:
        for det in couplings:
            key = tuple(det)
            if key not in dynamic_interactions.keys():
                dynamic_interactions[key] = inter
            else:
                dynamic_interactions[key] += inter
    if bSD:
        for singlet_couplings, inter in singlet_events:
            for det_plus, det_minus in singlet_couplings:
                key_plus = tuple(det_plus)
                key_minus = tuple(det_minus)

                if key_plus not in dynamic_interactions.keys():
                    dynamic_interactions[key_plus] = inter
                else:
                    dynamic_interactions[key_plus] += inter

                if key_minus not in dynamic_interactions.keys():
                    dynamic_interactions[key_minus] = -inter
                else:
                    dynamic_interactions[key_minus] += -inter
    return dynamic_interactions


### Spin operators

def compute_spin_z(configuration, vect): 
    """
    Return the operation of Sz on vect, with coordinates in basis
    Args:
        configuration (class baseConstruction.Configuration): the configuration of our lattice
        vect (np.array): a vector of our Hilbert space, with coordinates expressed in basis
    Returns:
        sz (float): the expected value of Sz on vect
        vect_return (np.array): representation of Sz.vect in basis
    """
    # Configuration
    nalpha = configuration.nalpha
    nbeta = configuration.nbeta 
    
    sz = (nalpha - nbeta)/2
    vect_return = sz * vect
    return sz, vect_return

def compute_spins_z_two_bodies_det(configuration, det, site_i, site_j): 
    """
    Compute the action of the operator Sz_i.Sz_j on a Slater determinant det \
        in the lattice defined by configuration
    Args:
        configuration (class baseConstruction.Configuration): the configuration of our lattice
        det (list<int>): the specified determinant
        site_i (int): a first site
        site_j (int): a second site
    Returns:
        sz (float): spin value between the sites site_i and site_j
    """
    return bC.get_spin_site(configuration, det, site_i) * bC.get_spin_site(configuration, det, site_j)

def compute_spins_z_two_bodies(configuration, vect, site_i, site_j): 
    """
    Compute the action of the operator Sz_i.Sz_j on a vector vect \
        in the lattice defined by configuration
    Args:
        configuration (class baseConstruction.Configuration): the configuration of our lattice
        vect (np.array): the specified vector
        site_i (int): a first site
        site_j (int): a second site
    Returns:
        vect_return (np.array): the result of the operation of Sz_{i}.Sz_{j} on vect
    """
    vect_return = np.copy(vect)
    vect_return.dtype = np.float_
    for ind in range(configuration.nb_conf):
        vect_return[ind] = vect[ind] * compute_spins_z_two_bodies_det(configuration, bC.get_hash_basis(configuration.nholes, configuration.nalpha, configuration.nbeta, ind), site_i, site_j)
    return vect_return
    
def compute_S_plus_S_minus_det(configuration, det, site_plus, site_minus): 
    """
    Computes the action of the operator Splus_(ind_plus).Sminus_(ind_minus) on a Slater determinant det \
        in the lattice defined by configuration
    Args:
        configuration (class baseConstruction.Configuration): the configuration of our lattice
        det (list<int>): a Slater determinant
        site_plus (ind): the site on which we apply S_plus
        site_minus (ind): the site on which we apply S_minus
    Returns:
        det_return (list<int>): the resulting Slater determinant
    """
    list_alpha = bC.get_indexes_of(configuration, det, 1)
    list_beta = bC.get_indexes_of(configuration, det, -1)
    
    if site_plus in list_beta:
        if site_minus in list_alpha:
            list_alpha.remove(site_minus)
            list_alpha.append(site_plus)
            
            list_beta.remove(site_plus)
            list_beta.append(site_minus)
            
            list_alpha.sort()
            list_beta.sort()
            
            det_return = bC.get_indexes_of(configuration, det, 0) + list_beta + list_alpha
            return det_return
    else:
        return

def compute_S_plus_S_minus(configuration, vect, site_plus, site_minus): 
    """
    Compute the action of the operator Splus_(ind_plus).Sminus_(ind_minus) on a vector vect \
        in the lattice defined by configuration
    Args:
        configuration (class baseConstruction.Configuration): the configuration of our lattice
        vect (np.array): the specified vector
        site_plus (int): the index of the "plus" site
        site_minus (int): the index of the "minus" site
    Returns:
        vect_return (np.array): the result of the operation of Sz_{i}.Sz_{j} on vect
    """
    vect_return = np.zeros(vect.shape)
    for ind in range(configuration.nb_conf):
        det = compute_S_plus_S_minus_det(configuration, bC.get_hash_basis(configuration.nholes, configuration.nalpha, configuration.nbeta, ind), site_plus, site_minus)
        if det:
            vect_return[bC.compute_hash_table(configuration, det),] += vect[ind,]
    return vect_return
    
def compute_S2(configuration, vect):
    """
    Return the operation of S^2 on a Slater determinant vect, with coordinates in basis
    Args:
        vect (np.array): a vector of our Hilbert space, with coordinates expressed in basis
    Returns:
        vect_return (np.array): representation of S^2|vect> in basis
    """
    # Configuration
    nsites = configuration.nsites
    nalpha = configuration.nalpha
    nbeta = configuration.nbeta

    vect_return = ((3/4) * (nalpha + nbeta) + (nalpha * (nalpha - 1) / 2 + nbeta * (nbeta - 1) / 2 - nalpha * nbeta)/2) * vect

    for site_0 in range(nsites - 1):
        for site_1 in range(site_0 + 1, nsites):
            vect_return += compute_S_plus_S_minus(configuration, vect, site_0, site_1) + compute_S_plus_S_minus(configuration, vect, site_1, site_0)
    return vect_return
    
def compute_S2_exp_val(configuration, vect):
    """
    Returns the expectation value <vect|S^2|vect>
    Args:
        configuration (class baseConstruction.Configuration): the configuration of our lattice
        vect (np.array): a vector of our Hilbert space, with coordinates expressed in basis
    Returns:
        expval (float): the expectation value <vect|S^2|vect>
    """
    if lin.norm(vect) != 0:
        vect_return = compute_S2(configuration, vect)
        if isinstance((np.dot(vect.T, vect_return) / lin.norm(vect)**2), float):
            return (np.dot(vect.T, vect_return) / lin.norm(vect)**2)
        else:
            return (np.dot(vect.T, vect_return) / lin.norm(vect)**2)[0, 0]
    else:
        return 0

def compute_spin_number(configuration, vect): 
    """
    Computes the value of the spin number of vect, with coordinates expressed in basis
    Args:
        configuration (class baseConstruction.Configuration): the configuration of our lattice
        vect (np.array): a vector of our Hilbert space, with coordinates expressed in basis
    Returns:
        spin_number (float): the spin number
    """
    try :
        expval = compute_S2_exp_val(configuration, vect) # x = <vect|S2|vect>
        coeff = [1, 1, -expval] # Compute positive solution of s(s+1) = <vect|S2|vect>
        (s1, s2) = np.roots(coeff)
        if s1 >= 0:
            # Round s1 to one decimal
            return round(2 * s1, 0) / 2 # s1 should be half-integer, probem with round(s1, 1) if s1=0.95 : return 0,9
        else:
            return round(2 * s2, 0) / 2
    except IndexError:
        return "-"
        
def compute_spin_multiplicity(spin_number):
    """
    Given the spin number of an eigenstate of S^2, returns the name associated to the multiplicity of this eigenstate
    Args:
        spin_number (float): the spin number
    Returns:
        multiplicity (str): multiplicity of the eigenstate
    """
    if spin_number == "-":
        return "-"
    else:
        mult = 2 * spin_number + 1
        if mult == 1:
            return "singlet"
        elif mult == 2:
            return "doublet"
        elif mult == 3:
            return "triplet"
        elif mult == 4:
            return "quartet"
        elif mult == 5:
            return "quintet"
        elif mult > 5:
            return "multiplet"

                
# Test S2 H944

# bond_matrix = lF.build_rectangular_bond_matrix(nbh=1, nbv=1, nblocs_side=3)
# configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=1, nbeta=4, nbh=1, nbv=1, nblocs_side=3)
# print("[itr] Building basis...")
# basis = bC.construct_basis(configuration)
# print("[itr] done.")
# determinants = basis.determinants

# vect = np.zeros((configuration.nb_conf, 1))
# vect[20, 0] = 1
# print("[itr] {} : {} \n{}".format(20, 1, bC.get_displayable_rectangular_lattice(configuration, determinants[20])))
# vect_s2 = compute_S2(configuration, vect)
# for row in range(configuration.nb_conf):
#     if vect_s2[row, 0] != 0:
#         print("[itr] {} : {}\n{}".format(row, vect_s2[row, 0], bC.get_displayable_rectangular_lattice(configuration, determinants[row])))
# print("[itr] <S^2> = ", compute_S2_exp_val(configuration, vect))

### CSR Tests
# 
# data=[1,4,3,8,2]
# indices=[1,0,3,0,1]
# indptr=[0,0,1,3,5,5]
# H=[data,indices,indptr]
# 
# print("[itr] H = ",H)
# 
# B=sp.csr_matrix(tuple(H),shape=(5,4))
# print("[itr] B = \n",B.toarray())
# 
# H=addHamCsr(H,1,1,2)
# H=addHamCsr(H,7,0,0)
# H=addHamCsr(H,4,1,1)
# H=addHamCsr(H,-1,4,3)
# H=addHamCsr(H,10,4,0)
# Ham=sp.csr_matrix(tuple(H),shape=(5,4))
# print("[itr] Ham = \n",Ham.toarray())

### Vhole Tests

# nbh=2
# nbv=2
# nblocs_side=2
# nalpha=6
# nbeta=6
# Basis = bC.baseConstruction(nbh,nbv,nblocs_side,nalpha,nbeta)
# 
# V=1.77
# VN=0.79
# 
# vect=[5,6,10,15,0,2,7,9,11,12,1,3,4,8,13,14]
# 
# bondMatrix=lF.buildBondMatrix(nbh,nbv,nblocs_side)
# 
# v=0
# indHoles=bC.indexesOf(Basis,vect,0)
# bondList=lF.buildBondList(nbh,nbv,nblocs_side,bondMatrix)
# holesBondList=[]
# holesNBondList=[]
# for iHole in range(len(indHoles)-1):
#     indHole=indHoles[iHole]
#     neighHole=lF.getNeighbors(indHole,nbh,nbv,nblocs_side,bondMatrix)
#     
#     for iNHole in range(iHole+1,len(indHoles)):
#         indNHole=indHoles[iNHole]
#             # VNN
#         if (indNHole in neighHole) and ([indHole,indNHole] not in holesBondList):
#             holesBondList.append([indHole,indNHole])
#         # VNNNper
#         neighNHole=lF.getNeighbors(indNHole,nbh,nbv,nblocs_side,bondMatrix)
#         for neigh in neighHole:
#             if (neigh in neighNHole) and ([indHole,indNHole] not in holesNBondList):
#                 holesNBondList.append([indHole,indNHole])
# v = len(holesBondList)*V+len(holesNBondList)*VN
# print("[itr] holesBondList = ",holesBondList) #####
# print("[itr] holesNBondList = ",holesNBondList) #####

### Spin operators tests

# #Test sur les vecteurs propres théoriques de (4,2,1)
# Basis=bC.baseConstruction(1,1,2,2,1)
# # Trou en d
# Abc=np.array([0,1,0,0,0,0,0,0,0,0,0,0])
# Abc.shape=(12,1)
# #print("[itr] Abc = ",bC.vectDisplay(Basis,bC.hashBasis(1,2,1,1)))
# 
# aBc=np.array([0,0,1,0,0,0,0,0,0,0,0,0])
# aBc.shape=(12,1)
# #print("[itr] aBc = ",bC.vectDisplay(Basis,bC.hashBasis(1,2,1,2)))
# 
# abC=np.array([1,0,0,0,0,0,0,0,0,0,0,0])
# abC.shape=(12,1)
# #print("[itr] abC = ",bC.vectDisplay(Basis,bC.hashBasis(1,2,1,0)))
# 
# Qd=(1/mp.sqrt(3))*(Abc+aBc+abC)
# DdS=(1/mp.sqrt(2))*(Abc-abC)
# DdA=(1/mp.sqrt(6))*(Abc-2*aBc+abC)
# 
# #Trou en b
# Acd=np.array([0,0,0,0,0,0,0,0,0,0,0,1])
# Acd.shape=(12,1)
# #print("[itr] Acd = ",bC.vectDisplay(Basis,bC.hashBasis(1,2,1,11)))
# 
# acD=np.array([0,0,0,0,0,0,0,0,0,1,0,0])
# acD.shape=(12,1)
# #print("[itr] acD = ",bC.vectDisplay(Basis,bC.hashBasis(1,2,1,9)))
# 
# aCd=np.array([0,0,0,0,0,0,0,0,0,0,1,0])
# aCd.shape=(12,1)
# #print("[itr] aCd = ",bC.vectDisplay(Basis,bC.hashBasis(1,2,1,10)))
# 
# Qb=(1/mp.sqrt(3))*(Acd+acD+aCd)
# DbS=(1/mp.sqrt(2))*(Acd-aCd)
# DbA=(1/mp.sqrt(6))*(Acd-2*acD+aCd)
# 
# # Doublets fondamentaux
# 
# DbdSS=(1/mp.sqrt(2))*(DbS+DdS)
# DbdSA=(1/mp.sqrt(2))*(DbS-DdS)
# DbdAS=(1/mp.sqrt(2))*(DbA+DdA)
# DbdAA=(1/mp.sqrt(2))*(DbA-DdA)
# 
# # Tests of Sz
# 
# print("[itr] <Dd1|Sz|Dd1> = ",SzExpVal(DdS,Basis))
# print("[itr] <Dd2|Sz|Dd2> = ",SzExpVal(DdA,Basis))
# print("----------------------------------------------------------")
# print("[itr] <DbdSS|Sz|DbdSS> = ",SzExpVal(DbdSS,Basis))
# print("[itr] <DbdSA|Sz|DbdSA> = ",SzExpVal(DbdSA,Basis))
# print("[itr] <DbdAS|Sz|DbdAS> = ",SzExpVal(DbdAS,Basis))
# print("[itr] <DbdAA|Sz|DbdAA> = ",SzExpVal(DbdAA,Basis))
# print("----------------------------------------------------------")
# #print("[itr] Sz|Qd> =\n",Sz(Qd,Basis))
# print("[itr] <Qd|Sz|Qd> = ",SzExpVal(Qd,Basis))
# print("----------------------------------------------------------")
# # Tests of S2
# print("[itr] DdS =\n",DdS)
# print("[itr] <DdS|S2+S3-|DdS> =\n",SplusSmoins(DdS,Basis,2,3))
# print("[itr] <DdS|S2|DdS> = ",S2ExpVal(DdS,Basis))
# print("[itr] <Qd|S2|Qd> = ",S2ExpVal(Qd,Basis))
# print("----------------------------------------------------------")
# # Tests of Spin number
# print("[itr] spinNumber(DdS) = ",spinNumber(DdS,Basis))
# print("[itr] spinNumber(Qd) = ",spinNumber(Qd,Basis))


# print("[itr] <Qd|S2|Qd> = ",S2ExpVal(Qd,Basis))
# print("[itr] <Dd1|S2|Dd1> = ",S2ExpVal(Dd1,Basis))
# print("[itr] <Dd2|S2|Dd2> = ",S2ExpVal(Dd2,Basis))
# print("----------------------------------------------------------")
# print("[itr] <Qb|S2|Qb> = ",S2ExpVal(Qb,Basis))
# print("[itr] <Db1|S2|Db1> = ",S2ExpVal(Dd1,Basis))
# print("[itr] <Db2|S2|Db2> = ",S2ExpVal(Dd2,Basis))
# 
# 
# print("[itr] vect =\n",vect)
# print("[itr] vect = ",bC.vectDisplay(Basis,bC.hashBasis(1,2,1,2)))

# print("[itr] <vect|Sz_3|vect> = ",bC.spinSite(Basis,bC.hashBasis(1,2,1,2),3))
#print("[itr] Sz_3 Sz_2|vect> =\n",Sz(vect,Basis,3,2))

#print("[itr] Splus_3 Smoins_2|vect> =\n",SplusSmoins(vect,Basis,3,2))

#print("[itr] S2i|vect> =\n",S2i(vect,Basis,3))

# print("[itr] S2|vect> =\n",S2(vect,Basis))

# Test sur les vecteurs propres théoriques de (4,0,2)
# Basis=bC.baseConstruction(1,1,2,0,2)
# nsites=4
# nalpha=0
# nbeta=2
# nbConfig=lF.nbConfig(nsites, nalpha, nbeta)
# for i in range(nbConfig):
#     vect=np.zeros((nbConfig,1))
#     vect[i,0]=1
#     print("[itr] vect =\n",vect) #####
#     print("[itr] spinNumber(vect) = ",spinNumber(vect,Basis)) #####

# Test compute_spin_number large sparse matrix
# bond_matrix = lF.build_rectangular_bond_matrix(nbh=2, nbv=1, nblocs_side=3)
# configuration = bC.Configuration(bond_matrix=bond_matrix, nholes=2, nbeta=8, nbh=2, nbv=1, nblocs_side=3)
# vect = np.random.rand(configuration.nb_conf, 1)
# print(compute_spin_number(configuration, vect))
