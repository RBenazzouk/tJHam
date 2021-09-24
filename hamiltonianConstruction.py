import latticeFunctions as lF
import numpy as np
import interactions as itr
import scipy.sparse as sp

class HamiltonianCOO:
    """
    Describes the Hamiltonian, in a COO format, as its three defining lists, \
        to be able to construct it iteratively as a COO sparse matrix.
    """
    def __init__(self, data, row, col):
        """
        Args:
            data (list<float>): the data in the Hamiltonian
            row (list<int>): the row of each element in data
            col (list<int>): the column of each element in data
        """
        self.data = data
        self.row = row
        self.col = col
    
def build_hamiltonian(configuration, interactions, basis, bool_interactions, sites_restriction=[], verbose=False): 
    """
    Builds the t-J extended Hamiltonian which implements all selected interactions, defined above
    Args:
        configuration (class latticeFunctions.Configuration): the configuration of our system
        interactions (class interactions.Interactions): the coefficients for any interaction in our Hamiltonian
        basis (class baseConstruction. basis): the  basis of Slater determinants of our system
        bool_interactions (class initialComputations.BoolInteractions): booleans representing the selection of implemented interactions
        sites_restriction (list<int>): the list of sites on which we allow the interactions, the others cannot be affected\
            (default to [], which stands for no restriction, so sites_restriction is automatically redefined as the set\
                containing all the sites)
        verbose (bool): prints messages in the shell if True, does nothing if False
    Returns:
        ham (scipy.sparse.csr_matrix): a csr sparse matrix representation of our Hamiltonian, in the mentionned basis (default to False)

    """
    # Configuration
    nsites = configuration.nsites
    nalpha = configuration.nalpha
    nbeta = configuration.nbeta
    nsites = configuration.nsites
    if not sites_restriction:
        sites_restriction = list(range(nsites))

    dim = lF.compute_dimension(nsites,nalpha,nbeta)
    # Interactions booleans
    bt = bool_interactions.bt
    bz = bool_interactions.bz
    bij = bool_interactions.bij
    bSD = bool_interactions.bSD
    btN = bool_interactions.btN
    bV = bool_interactions.bV
    
    print_verbose("--Hamiltonian initialization...", verbose)
    ham = HamiltonianCOO([], [], [])
    if bt==True:
        print_verbose("--t-hoping integral...", verbose)
        itr.add_hopping_integral(ham, basis, interactions)
    if bz==True:
        print_verbose("--J/4-spin interaction...", verbose)
        itr.add_static_spin_interaction(ham, basis, interactions)
    if bij==True:
        print_verbose("--J/2-spin-flip interaction...", verbose)
        itr.add_spin_flip(ham, basis, interactions)
    if bSD==True:
        print_verbose("--singlet-displacement term...", verbose)
        itr.add_singlet_displacement(ham, basis, interactions)
    if btN==True:
        print_verbose("--Next neighbour t-hopping integral...", verbose)
        itr.add_next_hopping_integral(ham, basis, interactions)
    if bV==True:
        print_verbose("--Hole-hole repulsion...", verbose)
        itr.add_hole_repulsion(ham, basis, interactions)
    ham = sp.coo_matrix((ham.data, (ham.row, ham.col)), shape=(dim, dim))
    print_verbose("--done", verbose)
    return ham.tocsr()

# Utilities

def add_ham_coo(ham, elt, ind_row, ind_col):
    """
    Add the element elt to the ind_row row and ind_col column of the matrix of ham, as a spare COO matrix
    Args:
        ham (scipy.sparse.coo.coo_matrix or class HamiltonianCOO): a COO sparse matrix representation of our Hamiltonian
        elt (float): the matrix element to add
        ind_row (int): the row where we add the element
        ind_col (int): the column where we add the element
    Returns:
        None (if ham is HamiltonianCOO) or ham_new (if ham is scipy.sparse.coo.coo_matrix)\
             (scipy.sparse.coo.coo_matrix): the updated COO representation of our Hamiltonian
    """
    if isinstance(ham.data, list):
        ham.data.append(elt)
        ham.row.append(ind_row)
        ham.col.append(ind_col)
        
    elif isinstance(ham.data, np.ndarray):
        ham_temp = HamiltonianCOO(list(ham.data), list(ham.row), list(ham.col))
        ham_temp.data.append(elt)
        ham_temp.row.append(ind_row)
        ham_temp.col.append(ind_col)

        return sp.coo_matrix((ham_temp.data, (ham_temp.row, ham_temp.col)), shape=ham.shape)

def print_verbose(string, verbose):
    """
    Prints the message string in the shell if and only if verbose is True
    Args:
        string (str): a message to print
        verbose (bool): select if the message will be printed in the shell or not
    """
    if verbose:
        print(string)
