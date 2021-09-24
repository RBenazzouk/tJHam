import latticeFunctions as lF
import baseConstruction as bC
import hamiltonianConstruction as hC
import davidson as dav
import interactions as itr

import math as mp
import copy
import sys
import os
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import json

import scipy.sparse as sp
import time

from tqdm import tqdm, trange

class BoolInteractions :
    """
    Booleans to decide which interaction to include in the Hamiltonian.
    """
    def __init__(self, bt, bz, bij, bSD, btN, bV, bV_embedding=False, b_spin_multipicities=True):
        """
        Args:
            bt (bool): nearest neighbor hopping integral
            bz (bool): static spins interactions
            bij (bool): spin flip interactions
            bSD (bool): singlet displacement
            btN (bool): next nearest neighbor hopping integral
            bV (bool): hole-hole repulsion
            b_spin_multiplicities (bool): spin multiplicities
        """
        self.bt = bt
        self.bz = bz
        self.bij = bij
        self.bSD = bSD
        self.btN = btN
        self.bV = bV
        self.b_spin_multipicities = b_spin_multipicities

class Times :
    """
    A class wrapping the computation times for basis, Hamiltonian and eigenspaces computations
    """
    def __init__(self, tbasis, thamiltonian, teigenstates):
        """
        Args:
            tbasis (float): the computation time of the basis
            thamiltonian (float): the computation time of the hamiltonian
            teigenspaces (float): the computation time of the eigenstates
        """
        self.tbasis = tbasis
        self.thamiltonian = thamiltonian
        self.teigenstates = teigenstates

class Data:
    """
    Definition of the computations
    """
    def __init__(self, configuration, interactions, diagonalization, bool_interactions):
        """
        Args:
            configuration (class baseDefinition.Configuration): the configuration of our system
            interactions (class interactions.Interactions): the coefficients of any interaction in our Hamiltonian
            diagonalization (class davidson.Diagonalization): the inputs for diagonalization
            bool_interactions (class bool_interactions): the booleans to select the interactions to implement in our Hamiltonian
        """
        self.configuration = configuration
        self.interactions = interactions
        self.diagonalization = diagonalization
        self.bool_interactions = bool_interactions

class DataComputations :
    def __init__(self, data, determinants, hamiltonian, eigenstates, times=None):
        self.data = data
        self.determinants = determinants
        self.hamiltonian = hamiltonian
        self.eigenstates = eigenstates
        self.times = times

# Computation functions

def perform_initial_computations(data): 
    """
    Computes and diagonalize the Hamiltonian according to the instructions in data.
    Args:
        data (class Data): definition of the computations
    Returns:
        data_computation (class DataComputation): the results of the computations    
    """
    # Configuration
    configuration = data.configuration
    # Interactions
    interactions = data.interactions
    # Diagonalization
    diagonalization = data.diagonalization
    nb_val, stp, tol, dim_max, initial_guess = diagonalization.nb_val, diagonalization.stp, diagonalization.tol, \
        diagonalization.dim_max, diagonalization.initial_guess
    # Booleans Interactions
    bool_interactions = data.bool_interactions
    
    # Compute basis
    start_basis = time.time()
    print("Computing basis...")
    basis = bC.construct_basis(configuration)
    print("done")
    stop_basis = time.time()
    duration_basis = stop_basis - start_basis    

    # Compute Hamiltonian
    start_ham = time.time()
    print("Building hamiltonian...")
    ham = hC.build_hamiltonian(configuration=configuration, interactions=interactions, basis=basis, bool_interactions=bool_interactions, sites_restriction=[], verbose=True)
    print("done")
    stop_ham = time.time()
    duration_ham = stop_ham - start_ham
    
    # Compute eigenstates
    start_eig = time.time()
    print("Computing eigenvalues and eigenvectors...")
    eigenstates = dav.modified_davidson_algorithm(ham, nb_val, initial_guess, stp, tol, dim_max, True)
    eigvects = eigenstates.eigvects
    spins = eigenstates.spins
    if data.bool_interactions.b_spin_multipicities:
        print("Computing spin multiplicities...")
        for col in trange(eigvects.shape[1]):
            vect = eigvects[:,[col]]
            spins.append(itr.compute_spin_number(configuration, vect))
    stop_eig = time.time()
    duration_eig = stop_eig - start_eig

    times = Times(duration_basis, duration_ham, duration_eig)

    data_computations = DataComputations(data, basis.determinants, ham, eigenstates, times)

    return data_computations


def save_data(data_computations):
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

    configuration_dict = configuration.__to_json__()
    # BoolInteractions
    bool_interactions = data_computations.data.bool_interactions
    bt = bool_interactions.bt
    bz = bool_interactions.bz
    bij = bool_interactions.bij
    bSD = bool_interactions.bSD
    btN = bool_interactions.btN
    bV = bool_interactions.bV

    # Diagonalization
    data_computations.data.diagonalization.initial_guess = None # Discarding initial guess

    # File Name
    file_name = "tJHam-{}{}{}".format(nsites, nalpha, nbeta)
    if bt and bz and bij and bSD and btN and bV:
        file_name += "Full.json"
    else:
        extensions = {"bt":"t", "bij":"J", "bSD":"SD", "btN":"tN", "bV":"V"}
        for name in dir(bool_interactions):
            if name in extensions and getattr(bool_interactions, name):
                file_name += extensions[name]
        file_name += ".json"
 
    # Saving results
    with open("./initialComputations/{}".format(file_name), "w") as savefile:
        savefile.write(json.dumps({
            "configuration" : configuration_dict, 
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
                },
            "times" : data_computations.times.__dict__
        }))

def recover_results(configuration, interactions, diagonalization, bool_interactions, determinants, hamiltonian, eigenstates, **kwargs):
    """
    Returns the saved data, collected with load_data, as objects in the corresponding class defined in this program
    Args:
        configuration (dict): a serialized version of the configuration of our system
        interactions (dict): a serialized version of the class describing \
            the coefficients of any interaction in our Hamiltonian
        diagonalization (dict): a serialized version of the class describing the inputs for diagonalization
        bool_interactions (dict): a serialized version of the class describing \
            the booleans to select the interactions implemented in our Hamiltonian
        determinants (list<list<int>>): list of every Slater determinant in the basis, encoded by listing first \
                the indices of the sites containing holes, then spins down, and finally spins up 
        hamiltonian (dict): a serialized version of the csr sparse matrix representation of our Hamiltonian
        eigenstates (dict): a serialized version of the class describing \
            the eigenvalues, spin numbers and correponding eigenvectors of the Hamiltonian
        kwargs : times (dict): a serialized version of the class containing the computation times
    Returns:
        data_computations (class DataComputations): the data computations wrapped in the class DataComputations
    """
    configuration_class = bC.Configuration(bond_matrix=np.array(configuration["bond_matrix"]), 
        nholes=configuration["nholes"], 
        nbeta=configuration["nbeta"])
    for key in ["sites_restriction", "nbh", "nbv", "nblocs_side"]:
        if key in configuration.keys():
            setattr(configuration_class, key, configuration[key])
    if "times" in kwargs.keys():
        return DataComputations(\
            Data(
                configuration_class, 
                itr.Interactions(**interactions),
                dav.Diagonalization(**diagonalization), 
                BoolInteractions(**bool_interactions), 
                ), 
            determinants, 
            sp.csr_matrix((hamiltonian["data"], hamiltonian["indices"], hamiltonian["indptr"]), \
                shape=(configuration["nb_conf"], configuration["nb_conf"])), 
            dav.Eigenstates(eigenstates["eigvals"], eigenstates["spins"], np.array(eigenstates["eigvects"])), 
            Times(**kwargs["times"])
            )
    else:
        return DataComputations(\
            Data(
                configuration_class, 
                itr.Interactions(**interactions),
                dav.Diagonalization(**diagonalization), 
                BoolInteractions(**bool_interactions), 
                ), 
            determinants, 
            sp.csr_matrix((hamiltonian["data"], hamiltonian["indices"], hamiltonian["indptr"]), \
                shape=(configuration["nb_conf"], configuration["nb_conf"])), 
            dav.Eigenstates(eigenstates["eigvals"], eigenstates["spins"], np.array(eigenstates["eigvects"]))
            )


def load_data(file_name):
    """
    Load data from the file ./initialComputations/file_name.
    Args:
        file_name (str): the name of the file in which we have saved the computation data
    Returns:
        recovered_results (class DataComputations): the data computations wrapped in the class DataComputations
    """
    with open("./initialComputations/{}".format(file_name), "r") as savefile:
        return recover_results(**json.loads(savefile.read()))

    
def perform_configurations_computations(data):
    """
    Performs the computations for every configuration of a given lattice, and saves the data in a JSON format.
    Args:
        data (class Data): the data defining the lattice
    """

    # Configuration
    configuration = data.configuration
    nsites = configuration.nsites
    nholes = configuration.nholes
    bond_matrix = configuration.bond_matrix

    for nalpha in range(nsites - nholes + 1):

        nbeta = nsites - nholes - nalpha
        # Configuration update
        configuration = bC.Configuration(bond_matrix=bond_matrix, 
            nholes=nholes, 
            nbeta=nbeta, 
            sites_restriction=configuration.sites_restriction, 
            nbh=configuration.nbh, 
            nbv=configuration.nbv, 
            nblocs_side=configuration.nblocs_side)
        ms = configuration.spin_number
        doping = configuration.doping
        nb_conf = configuration.nb_conf
        data.configuration = configuration

        print("---Configuration--------------------------------------------------------------------------------------------------")
        print("--nsites = ", nsites)
        print("--nalpha = ", nalpha)
        print("--nbeta = ", nbeta)
        print("--nholes = ", nholes)
        print("--ms = ", ms)
        print("--doping = ", doping)
        print("--nb_conf = ", nb_conf)
        print("----------------")
        print("\n")
        
        data_computations = perform_initial_computations(data)
        print("Eigenstates")
        if data.bool_interactions.b_spin_multipicities:
            for ind, eigval in enumerate(data_computations.eigenstates.eigvals):
                print("- {} : {}".format(eigval, data_computations.eigenstates.spins[ind]))
        print("Saving data...")
        save_data(data_computations)

        print("done.")


def get_most_dominant_determinants(determinants, vect, nb_det):
    """
    Yields the most dominant determinants in the input vector.
    Args:
        determinants (list<list<determinants>>): the list of determinants that forms the basis of the Hilbert space
        vect (np.array): the vector in the basis of determinants
        nb_det (int): number of most dominant determinants to yield
    Returns:
        most_dominant_dets (dict<float : list<list<int>>>): a dictionnary that contains the most dominant determinants, indexed by their coefficients in vect
    """
    most_dominant_dets = {}
    amplitudes = abs(vect).tolist()
    coefficients = vect.tolist()
    indices_memory = []
    for amplitude in list(reversed(sorted(amplitudes)))[:nb_det]:
        index = 0
        while amplitudes[index] != amplitude or index in indices_memory:
            index += 1
        indices_memory.append(index)
        most_dominant_dets[index, coefficients[index]] = determinants[index]
    return most_dominant_dets
