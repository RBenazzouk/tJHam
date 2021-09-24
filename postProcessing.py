import baseConstruction as bC
import interactions as itr
import hamiltonianConstruction as hC
import davidson as dav
import initialComputations as iC

import math as mp
import numpy as np
import numpy.linalg as lin

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.lines import Line2D

import scipy.interpolate as interp
from scipy.integrate import simps

import copy
import os

from tqdm import trange

class InputPlotting :
    def __init__(self, nb_excited, nb_plot, bool_spectrum=False, bool_holes=False, bool_electrons=False, bool_determinants=False, bool_neel_embedding=False, bool_static_embedding=False, bool_dynamic_embedding=False, bool_neighbor_embedding=False):
        """
        Selects the plots to perform.
        Args:
            nb_excited (int): number of states for which we compute the electron and spin distributions
            nb_plot (int): number of states for which we compute the charge distributions
            bool_spectrum (bool): True to plot the spectrum of the configuration
            bool_holes (bool):  True to plot the charge distribution in the lattice
            bool_electrons (bool): True to plot the electrons and spin distributions in the lattice
            bool_determinants (bool): True to plot the repartition of energy amongst determinants
            bool_neel_embedding (bool): True to plot the influence of the static embedding within an autocoherent peripheral Neel lattice on the energy of the determinants
            bool_static_embedding (bool): True to plot the influence of the static embedding within neighboring blocks on the energy of the determinants
            bool_dynamic_embedding (bool): True to plot the influence of the dynamic embedding within neighboring blocks on the energy of the determinants
            bool_neighbor_embedding (bool): True to plot the influence of the embedding next to a single neighboring block on the energy of the determinants
        """
        self.nb_excited = nb_excited
        self.nb_plot = nb_plot
        self.bool_spectrum = bool_spectrum
        self.bool_holes = bool_holes
        self.bool_electrons = bool_electrons
        self.bool_determinants = bool_determinants
        self.bool_neel_embedding = bool_neel_embedding
        self.bool_static_embedding = bool_static_embedding
        self.bool_dynamic_embedding = bool_dynamic_embedding
        self.bool_neighbor_embedding = bool_neighbor_embedding

def generate_plots_configuration(data, input_plotting):
    """
    Generates the different plots defined below for a certain configuration.
    Args:
        data (class initialComputations.Data): the data defining the computations
        input_plotting (class InputPlotting): defines which plots to perform
    """
    # Configuration
    configuration = data.configuration
    nsites = configuration.nsites
    nholes = configuration.nholes
    nbeta = configuration.nbeta
    nalpha = configuration.nalpha
    ms = configuration.spin_number
    doping = configuration.doping
    nb_conf = configuration.nb_conf
    # Bool Interactions
    bool_interactions = data.bool_interactions
    bt = bool_interactions.bt
    bz = bool_interactions.bz
    bij = bool_interactions.bij
    bSD = bool_interactions.bSD
    btN = bool_interactions.btN
    bV = bool_interactions.bV
    # Plotting and printing inputs
    nb_excited = input_plotting.nb_excited
    nb_plot = input_plotting.nb_plot
    bool_spectrum = input_plotting.bool_spectrum
    bool_holes = input_plotting.bool_holes
    bool_electrons = input_plotting.bool_electrons
    bool_determinants = input_plotting.bool_determinants
    bool_neel_embedding = input_plotting.bool_neel_embedding
    bool_static_embedding = input_plotting.bool_static_embedding
    bool_dynamic_embedding = input_plotting.bool_dynamic_embedding
    bool_neighbor_embedding = input_plotting.bool_neighbor_embedding

    # File Name
    file_name = "tJHam-{}{}{}".format(nsites, nalpha, nbeta)
    file_name_emb = "tJHam"
    if bt and bz and bij and bSD and btN and bV:
        file_name += "Full.json"
    else:
        extensions = {"bt":"t", "bij":"J", "bSD":"SD", "btN":"tN", "bV":"V"}
        for name in dir(bool_interactions):
            if name in extensions and getattr(bool_interactions, name):
                file_name += extensions[name]
        file_name += ".json"

    # Loading data
    print("Loading data...")
    data_computations = iC.load_data(file_name)

    if bool_neel_embedding or bool_static_embedding or bool_dynamic_embedding:

        if bool_dynamic_embedding:
            file_name_emb += "-dyn"
        if bool_neel_embedding:
            file_name_emb += "-neel"
        if bool_neighbor_embedding:
            file_name_emb += "-neigh"
        file_name_emb += "-{}{}{}".format(nsites, nalpha, nbeta)
        if bt and bz and bij and bSD and btN and bV:
            file_name_emb += "Full.json"
        else:
            extensions = {"bt":"t", "bij":"J", "bSD":"SD", "btN":"tN", "bV":"V"}
            for name in dir(bool_interactions):
                if name in extensions:
                    file_name_emb += extensions[name]
            file_name_emb += ".json"
        data_computations_embedding = iC.load_data("Embedding_blocks/" + file_name_emb)
    print("done.")

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

    if bool_spectrum:
        print("--Plotting spectrum...")
        save_spectrum(data_computations)
    if nholes > 0 and bool_holes:
        print("--Plotting hole density...")
        save_hole_probability_lattice(data_computations, nb_plot)
    if bool_electrons and (nalpha * nbeta > 0):
        print("--Plotting electrons and charge density...")
        for ind_state in trange(nb_excited):
            save_electron_probability_lattice(data_computations, ind_state)
    if bool_determinants:
        print("--Plotting determinants amplitudes...")
        plot_determinants_amplitude_repartition(data_computations, nb_plot)
    if bool_neel_embedding:
        print("--Plotting influence of the Néel embedding on determinants energy")
        plot_embedding_effect_on_determinants(data_computations, data_computations_embedding, min(nb_conf, 30))
    if bool_static_embedding or bool_dynamic_embedding:
        print("--Plotting influence of the blocks-embedding on determinants energy")
        plot_embedding_effect_on_determinants(data_computations, data_computations_embedding, min(nb_conf, 30))

def generate_plots(data, input_plotting):
    """
    Generate the different plots defined below for all of the configurations of a certain lattice.
    Args:
        data (class initialComputations.Data): the data defining the computations
        input_plotting (class InputPlotting): defines which plots to perform
    """
    # Configuration
    configuration = data.configuration
    bond_matrix = configuration.bond_matrix
    nsites = configuration.nsites
    nholes = configuration.nholes


    for nalpha in range(nsites - nholes + 1):
        nbeta = nsites - nholes - nalpha
        # Configuration update
        configuration = bC.Configuration(bond_matrix, nholes, nbeta)
        nsites = configuration.nsites
        nholes = configuration.nholes
        data.configuration = configuration

        generate_plots_configuration(data, input_plotting)

      
### Phase space

def plot_spectrum_dependance(data): # Not complete, problem with colors of overlapping energy pliots
    """
    Plots the dependance of the spectrum on the quotient t/J.

    Args:
        data (class initialComputations.Data): the data defining the computations
    """
    # Configuration
    configuration_tJ = data.configuration
    # Interactions
    interactions = data.interactions
    # Diagonalization
    diagonalization = data.diagonalization

    tJ_range = np.linspace(0.3, 10, 1000)

    spectrums_tJ = [[] for _ in range(diagonalization.nb_val)]
    spin_multiplicities_tJ = [[] for _ in range(diagonalization.nb_val)]
    degeneracies_tJ = [[] for _ in range(diagonalization.nb_val)]

    spectrums_tJ_extended = [[] for _ in range(diagonalization.nb_val)]
    spin_multiplicities_tJ_extended = [[] for _ in range(diagonalization.nb_val)]
    degeneracies_tJ_extended = [[] for _ in range(diagonalization.nb_val)]


    for tJ in tJ_range:
        t = interactions.t
        J = abs(t)/tJ
        Jhpar = interactions.Jhpar * J / interactions.J
        Jhper = interactions.Jhper * J / interactions.J
        hSDper = interactions.hSDper * J / interactions.J
        hSDpar = interactions.hSDpar * J / interactions.J
        tNNN = interactions.tNNN
        tnNNN = interactions.tnNNN
        VNN = interactions.VNN
        VNNNper = interactions.VNNNper
        interactions_tJ = itr.Interactions(t=t, J=J, Jhpar=Jhpar, Jhper=Jhper, hSDper=hSDper, hSDpar=hSDpar, \
        tNNN=tNNN, tnNNN=tnNNN, VNN=VNN, VNNNper=VNNNper)
        bool_interactions_tJ = iC.BoolInteractions(bt=True, bz=True, bij=True, bSD=False, btN=False, bV=False, bV_embedding=False, b_spin_multipicities=True)
        bool_interactions_tJ_extended = iC.BoolInteractions(bt=True, bz=True, bij=True, bSD=True, btN=True, bV=True, bV_embedding=False, b_spin_multipicities=True)

        data_tJ = iC.Data(configuration=configuration_tJ, interactions=interactions_tJ, diagonalization=diagonalization, bool_interactions=bool_interactions_tJ)
        data_tJ_extended = iC.Data(configuration=configuration_tJ, interactions=interactions_tJ, diagonalization=diagonalization, bool_interactions=bool_interactions_tJ_extended)

        data_computation_tJ = iC.perform_initial_computations(data_tJ)
        eigvals_tJ = [round(energy, 6) for energy in data_computation_tJ.eigenstates.eigvals]
        multiplicities_tJ = [itr.compute_spin_multiplicity(spin_number) for spin_number in data_computation_tJ.eigenstates.spins]
        degeneracy_tJ = [eigvals_tJ.count(eigval_tJ) for eigval_tJ in eigvals_tJ]

        data_computation_tJ_extended = iC.perform_initial_computations(data_tJ_extended)
        eigvals_tJ_extended = [round(energy, 6) for energy in data_computation_tJ_extended.eigenstates.eigvals]
        multiplicities_tJ_extended = [itr.compute_spin_multiplicity(spin_number) for spin_number in data_computation_tJ_extended.eigenstates.spins]
        degeneracy_tJ_extended = [eigvals_tJ_extended.count(eigval_tJ_extended) for eigval_tJ_extended in eigvals_tJ_extended]

        for ind, (eigval_tJ, eigval_tJ_extended) in enumerate(zip(sorted(eigvals_tJ), sorted(eigvals_tJ_extended))):
            spectrums_tJ[ind].append(eigval_tJ)
            spin_multiplicities_tJ[ind].append(multiplicities_tJ[ind])
            degeneracies_tJ[ind].append(degeneracy_tJ[ind])

            spectrums_tJ_extended[ind].append(eigval_tJ_extended)
            spin_multiplicities_tJ_extended[ind].append(multiplicities_tJ_extended[ind])
            degeneracies_tJ_extended[ind].append(degeneracy_tJ_extended[ind])

    # Reorganizing lists to make the spectra monotonous
    colors_multiplicities = {"doublet" : 'b', 'quartet' : 'g'}
    linestyles_multiplicities = {1 : '--', 2 : '-'}

    ind_spectrum = 0
    while ind_spectrum < diagonalization.nb_val - 1:
        multiplicity_tJ, degeneracy_tJ = spin_multiplicities_tJ[ind_spectrum], degeneracies_tJ[ind_spectrum]
        nb_points = len(degeneracy_tJ)

        ind = 0
        m_tJ = multiplicity_tJ[0]
        d_tJ = degeneracy_tJ[0]
        while ind < nb_points - 1 and m_tJ == multiplicity_tJ[ind + 1] and d_tJ == degeneracy_tJ[ind + 1]:
            ind += 1
            m_tJ = multiplicity_tJ[ind]
            d_tJ = degeneracy_tJ[ind]
        if ind == nb_points - 1:
            ind_spectrum += 1
        else:
            new_ind_spectrum = ind_spectrum + 1

            while new_ind_spectrum < diagonalization.nb_val \
                and ((spin_multiplicities_tJ[ind_spectrum][ind] != spin_multiplicities_tJ[new_ind_spectrum][ind + 1] or spin_multiplicities_tJ[new_ind_spectrum][ind] != spin_multiplicities_tJ[ind_spectrum][ind + 1]) \
                    or (degeneracies_tJ[ind_spectrum][ind] != degeneracies_tJ[new_ind_spectrum][ind + 1] or degeneracies_tJ[new_ind_spectrum][ind] != degeneracies_tJ[ind_spectrum][ind + 1])):
                new_ind_spectrum += 1
            if new_ind_spectrum < diagonalization.nb_val:
                #print(ind_spectrum, new_ind_spectrum)
                #print("m : ", spin_multiplicities_tJ[ind_spectrum][ind], spin_multiplicities_tJ[ind_spectrum][ind + 1])
                #print("m : ", spin_multiplicities_tJ[new_ind_spectrum][ind], spin_multiplicities_tJ[new_ind_spectrum][ind + 1])
                #print("d : ", degeneracies_tJ[ind_spectrum][ind], degeneracies_tJ[ind_spectrum][ind + 1])
                #print("d : ", degeneracies_tJ[new_ind_spectrum][ind], degeneracies_tJ[new_ind_spectrum][ind + 1])
                e_ind_m, e_ind, e_ind_p = spectrums_tJ[ind_spectrum][ind - 1], spectrums_tJ[ind_spectrum][ind], spectrums_tJ[ind_spectrum][ind + 1]
                new_e_ind_m, new_e_ind, new_e_ind_p = spectrums_tJ[new_ind_spectrum][ind - 1], spectrums_tJ[new_ind_spectrum][ind], spectrums_tJ[new_ind_spectrum][ind + 1]
                if ind > 1 and abs(abs(e_ind_p - e_ind) - abs(e_ind - e_ind_m)) < abs(abs(new_e_ind_p - e_ind) - abs(e_ind - e_ind_m)) \
                    and abs(abs(new_e_ind_p - new_e_ind) - abs(new_e_ind - new_e_ind_m)) < abs(abs(e_ind_p - new_e_ind) - abs(new_e_ind - new_e_ind_m)):

                    #print(ind_spectrum, new_ind_spectrum, " : ", ind)
                    #print("{} = | |{} - {}| - |{} - {}| | < | |{} - {}| - |{} - {}| | = {}".format(abs(abs(e_ind_p - e_ind) - abs(e_ind - e_ind_m)), e_ind_p, e_ind, e_ind, e_ind_m, new_e_ind_p, e_ind, e_ind, e_ind_m, abs(abs(new_e_ind_p - e_ind) - abs(e_ind - e_ind_m))))
                    #print("{} = | |{} - {}| - |{} - {}| | < | |{} - {}| - |{} - {}| | = {}\n".format(abs(abs(new_e_ind_p - new_e_ind) - abs(new_e_ind - new_e_ind_m)), new_e_ind_p, new_e_ind, new_e_ind, new_e_ind_m, e_ind_p, new_e_ind, new_e_ind, new_e_ind_m, abs(abs(e_ind_p - new_e_ind) - abs(new_e_ind - new_e_ind_m))))

                    spectrums_tJ[ind_spectrum][ind + 1 :], spectrums_tJ[new_ind_spectrum][ind + 1 :] = spectrums_tJ[new_ind_spectrum][ind + 1 :], spectrums_tJ[ind_spectrum][ind + 1 :]
                    spin_multiplicities_tJ[ind_spectrum][ind + 1 :], spin_multiplicities_tJ[new_ind_spectrum][ind + 1 :] = spin_multiplicities_tJ[new_ind_spectrum][ind + 1 :], spin_multiplicities_tJ[ind_spectrum][ind + 1 :]
                    degeneracies_tJ[ind_spectrum][ind + 1 :], degeneracies_tJ[new_ind_spectrum][ind + 1 :] = degeneracies_tJ[new_ind_spectrum][ind + 1 :], degeneracies_tJ[ind_spectrum][ind + 1 :]    
                else:
                    ind_spectrum += 1
            else:
                ind_spectrum += 1

    ind_spectrum_extended = 0
    while ind_spectrum_extended < diagonalization.nb_val - 1:
        multiplicity_tJ_extended, degeneracy_tJ_extended = spin_multiplicities_tJ_extended[ind_spectrum_extended], degeneracies_tJ_extended[ind_spectrum_extended]
        nb_points = len(degeneracy_tJ_extended)

        ind = 0
        m_tJ = multiplicity_tJ_extended[0]
        d_tJ = degeneracy_tJ_extended[0]
        while ind < nb_points - 1 and m_tJ == multiplicity_tJ_extended[ind + 1] and d_tJ == degeneracy_tJ_extended[ind + 1]:
            ind += 1
            m_tJ = multiplicity_tJ_extended[ind]
            d_tJ = degeneracy_tJ_extended[ind]
        if ind == nb_points - 1:
            ind_spectrum_extended += 1
        else:
            new_ind_spectrum_extended = ind_spectrum_extended + 1

            while new_ind_spectrum_extended < diagonalization.nb_val \
                and ((spin_multiplicities_tJ_extended[ind_spectrum_extended][ind] != spin_multiplicities_tJ_extended[new_ind_spectrum_extended][ind + 1] or spin_multiplicities_tJ_extended[new_ind_spectrum_extended][ind] != spin_multiplicities_tJ_extended[ind_spectrum_extended][ind + 1]) \
                    or (degeneracies_tJ_extended[ind_spectrum_extended][ind] != degeneracies_tJ_extended[new_ind_spectrum_extended][ind + 1] or degeneracies_tJ_extended[new_ind_spectrum_extended][ind] != degeneracies_tJ_extended[ind_spectrum_extended][ind + 1])):
                new_ind_spectrum_extended += 1
            if new_ind_spectrum_extended < diagonalization.nb_val:
                e_ind_m, e_ind, e_ind_p = spectrums_tJ_extended[ind_spectrum_extended][ind - 1], spectrums_tJ_extended[ind_spectrum_extended][ind], spectrums_tJ_extended[ind_spectrum_extended][ind + 1]
                new_e_ind_m, new_e_ind, new_e_ind_p = spectrums_tJ_extended[new_ind_spectrum_extended][ind - 1], spectrums_tJ_extended[new_ind_spectrum_extended][ind], spectrums_tJ_extended[new_ind_spectrum_extended][ind + 1]
                if ind > 1 and abs(abs(e_ind_p - e_ind) - abs(e_ind - e_ind_m)) < abs(abs(new_e_ind_p - e_ind) - abs(e_ind - e_ind_m)) \
                    and abs(abs(new_e_ind_p - new_e_ind) - abs(new_e_ind - new_e_ind_m)) < abs(abs(e_ind_p - new_e_ind) - abs(new_e_ind - new_e_ind_m)):

                    spectrums_tJ_extended[ind_spectrum_extended][ind + 1 :], spectrums_tJ_extended[new_ind_spectrum_extended][ind + 1 :] = spectrums_tJ_extended[new_ind_spectrum_extended][ind + 1 :], spectrums_tJ_extended[ind_spectrum_extended][ind + 1 :]
                    spin_multiplicities_tJ_extended[ind_spectrum_extended][ind + 1 :], spin_multiplicities_tJ_extended[new_ind_spectrum_extended][ind + 1 :] = spin_multiplicities_tJ_extended[new_ind_spectrum_extended][ind + 1 :], spin_multiplicities_tJ_extended[ind_spectrum_extended][ind + 1 :]
                    degeneracies_tJ_extended[ind_spectrum_extended][ind + 1 :], degeneracies_tJ_extended[new_ind_spectrum_extended][ind + 1 :] = degeneracies_tJ_extended[new_ind_spectrum_extended][ind + 1 :], degeneracies_tJ_extended[ind_spectrum_extended][ind + 1 :]    
                else:
                    ind_spectrum_extended += 1
            else:
                ind_spectrum_extended += 1

    custom_lines = [Line2D([0], [0], color='b', linestyle='-', label='doublet 2x degenerate'),
                    Line2D([0], [0], color='b', linestyle='--', label='doublet non-degenerate'),
                    Line2D([0], [0], color='g', linestyle='-', label='quartet 2x degenerate'),
                    Line2D([0], [0], color='g', linestyle='--', label='quartet non-degenerate')]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    for ind in range(diagonalization.nb_val):
        ax1.plot(tJ_range, np.array(spectrums_tJ[ind]), color=colors_multiplicities[spin_multiplicities_tJ[ind][0]], linestyle=linestyles_multiplicities[degeneracies_tJ[ind][0]])
    ax1.vlines(0.55/0.123, ymin=-2.6, ymax=1.3, color='k', linestyles='dashed')
    ax1.legend(handles=custom_lines)
    ax1.set_xscale('log')
    ax1.set_xlabel("t/J")
    ax1.set_ylabel("E")

    try :
        plt.savefig("./tJ-HamCsr/tJ-H{}/ParametrizedSpectrum/tJ-H{}{}{}.png".format(configuration_tJ.nsites, configuration_tJ.nsites, configuration_tJ.nalpha, configuration_tJ.nbeta))
        plt.savefig("./tJ-HamCsr/tJ-H{}/ParametrizedSpectrum/tJ-H{}{}{}.pdf".format(configuration_tJ.nsites, configuration_tJ.nsites, configuration_tJ.nalpha, configuration_tJ.nbeta))
    except OSError:
        os.mkdir(path = "./tJ-HamCsr/tJ-H{}/ParametrizedSpectrum/".format(configuration_tJ.nsites))
        plt.savefig("./tJ-HamCsr/tJ-H{}/ParametrizedSpectrum/tJ-H{}{}{}.png".format(configuration_tJ.nsites, configuration_tJ.nsites, configuration_tJ.nalpha, configuration_tJ.nbeta))
        plt.savefig("./tJ-HamCsr/tJ-H{}/ParametrizedSpectrum/tJ-H{}{}{}.pdf".format(configuration_tJ.nsites, configuration_tJ.nsites, configuration_tJ.nalpha, configuration_tJ.nbeta))


    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    for ind in range(diagonalization.nb_val):
        ax2.plot(tJ_range, np.array(spectrums_tJ_extended[ind]), color=colors_multiplicities[spin_multiplicities_tJ_extended[ind][0]], linestyle=linestyles_multiplicities[degeneracies_tJ_extended[ind][0]])
    ax2.vlines(0.55/0.123, ymin=-2.6, ymax=1.3, color='k', linestyles='dashed')
    ax2.legend(handles=custom_lines)
    ax2.set_xscale('log')
    ax2.set_xlabel("t/J")
    ax2.set_ylabel("E")

    try :
        plt.savefig("./tJ-HamCsr/tJ-H{}/ParametrizedSpectrum/tJ-H{}{}{}-extended.png".format(configuration_tJ.nsites, configuration_tJ.nsites, configuration_tJ.nalpha, configuration_tJ.nbeta))
        plt.savefig("./tJ-HamCsr/tJ-H{}/ParametrizedSpectrum/tJ-H{}{}{}-extended.pdf".format(configuration_tJ.nsites, configuration_tJ.nsites, configuration_tJ.nalpha, configuration_tJ.nbeta))
    except OSError:
        os.mkdir(path = "./tJ-HamCsr/tJ-H{}/ParametrizedSpectrum/".format(configuration_tJ.nsites))
        plt.savefig("./tJ-HamCsr/tJ-H{}/ParametrizedSpectrum/tJ-H{}{}{}-extended.png".format(configuration_tJ.nsites, configuration_tJ.nsites, configuration_tJ.nalpha, configuration_tJ.nbeta))
        plt.savefig("./tJ-HamCsr/tJ-H{}/ParametrizedSpectrum/tJ-H{}{}{}-extended.pdf".format(configuration_tJ.nsites, configuration_tJ.nsites, configuration_tJ.nalpha, configuration_tJ.nbeta))


def save_phase_space(data, nb_state, nb_t, nb_v):
    """
    Plots the spin multiplicity of the nb_state -th excited state depending on the ratio t/J
    Args:
        data (class initialComputations.Data): the data defining the computation
        nb_state (int): the index of the desired excited state
        nb_t (int): the number of values for t to plot
        nb_v (int): the number of values of VNN to plot
    """
    
    # Configuration
    configuration = data.configuration
    nsites = configuration.nsites
    nholes = configuration.nholes

    # Interactions
    interactions = data.interactions

    J = interactions.J

    # Diagonalization
    diagonalization = data.diagonalization

    stp, tol, dim_max = diagonalization.stp, diagonalization.tol, diagonalization.dim_max

    # Interactions booleans
    bool_interactions = data.bool_interactions


    # Definition of the configuration
    for nalpha in range(nsites - nholes + 1):
        nbeta = nsites - nholes - nalpha
        
        configuration.nalpha = nalpha
        configuration.nbeta = nbeta

        basis = bC.construct_basis(configuration)
        
        print("[iC] Configuration : nsites = {}, nholes = {}, nalpha = {}, nbeta = {}".format(nsites, nholes, nalpha, nbeta))
        
        # Definition of the grid
        
        x = np.linspace(0, 15, nb_v)
        y = np.linspace(-5, 0, nb_t)
        X, Y = np.meshgrid(x, y)
        
        # Computation of the plotted values
        
        Zval = np.zeros((len(Y),len(X)))
        Zspin = np.zeros((len(Y),len(X)))
        
        for iX in range(len(X)):
            for iY in range((len(Y))):
                # Hamiltonian construction
                t = -iY*J
                interactions.t = t
                interactions.tNNN = -13 * t / 55
                interactions.tnNNN = 36 * t / 55
                VNN = iX*J
                interactions.VNN = VNN
                interactions.VNNNper = (0.79/1.77)*VNN
                
                verbose = False
                ham = hC.build_hamiltonian(configuration, interactions, basis, bool_interactions, verbose)
                
                try :
                    eigenstates = dav.modified_davidson_algorithm(ham, nb_state, stp, tol, dim_max)
                    eigval = eigenstates.eigvals[nb_state]
                    eigvect = eigenstates.eigvects[:,[nb_state]]
                    Zval[iY, iX] = eigval
                    spin = itr.compute_spin_number(basis, eigvect)
                    if spin != '-':
                        Zspin[iY, iX] = spin
                    else :
                        Zspin[iY, iX] = 0
                except np.linalg.LinAlgError :
                    Zval[iY, iX] = 0
                    Zspin[iY, iX] = 0
        
        fig, (ax0, ax1) = plt.subplots(2, 1)
        
        im0 = ax0.pcolormesh(x, y, Zval, cmap = 'cool', shading = 'auto')
        fig.colorbar(im0, ax=ax0)
        ax0.set_xlabel('V/J')
        ax0.set_ylabel('t/J')
        ax0.set_title("Groud-state energy")
        
        
        cmapSpin = ListedColormap(["deepskyblue", "aquamarine", "lightgreen", "orange","tomato"])

        im1 = ax1.pcolormesh(x, y, Zspin, cmap = cmapSpin, shading = 'auto')
        fig.colorbar(im1, ax=ax1)
        ax1.set_xlabel('V/J')
        ax1.set_ylabel('t/J')
        ax1.set_title("Spin multiplicity")
        
        plt.tight_layout()
        
        plt.savefig("./tJ-HamCsr/tJ-H{}/PhaseDiagrams/tJ-H{}{}{}.png".format(str(nsites),str(nsites),str(nalpha),str(nbeta)), 
            dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            metadata=None)

        plt.close()
     
### Spectrum

def save_spectrum(data_computation):
    """
    Saves a visual representation of the spectrum from data_computations
    Args:
        data_computation (class initialComputations.DataComputation): the computation results
    """
    
    # Configuration
    configuration = data_computation.data.configuration
    nsites = configuration.nsites
    nalpha = configuration.nalpha
    nbeta = configuration.nbeta
        
    # Spectrum
    eigenstates = data_computation.eigenstates
    eigvals = eigenstates.eigvals
    
    # Building the list of eigenvalues with their degeneracy
    
    vals = copy.deepcopy(eigvals)
    degeneracy = [1 for _ in vals]
    ind = 0
    while ind < len(vals)-1:
        rest = vals[ind + 1:]
        if vals[ind] in rest:
            rest.remove(vals[ind])
            vals = vals[:ind + 1]+rest
            degeneracy[ind] += 1
        else :
            ind += 1
    degeneracy = degeneracy[:len(vals)]

    # Building the list orbitals convenient to plot the spectrum given
    
    orbitals = []
    for ind_val, val in enumerate(vals):
        for ind in range(degeneracy[ind_val]):
            orbitals.append([[ind + 1, ind + 1.5],[val, val]])

    # Plotting the spectrum
    
    fig = plt.figure()
    for orbital in orbitals :
        plt.plot(orbital[0], orbital[1], color = 'black')
    plt.ylabel("Energy of eigenstates")
    
    # Saving the spectrum
    
    fig.savefig("./tJ-HamCsr/tJ-H{}/Spectrum/tJ-H{}{}{}.png".format(str(nsites),str(nsites),str(nalpha),str(nbeta)))

    plt.close()
    
### Densities

def save_hole_probability_lattice(data_computation,nb_subplots):
    """
    Plots the probability of presence of the hole on the lattice.
    Args:
        data_computations (class initialComputations.DataComputation): the data from computations
        nb_subplots (int): the number of subplots
    """

    nb_plot = int(mp.floor(mp.sqrt(nb_subplots))**2)
    nb_row = int(mp.sqrt(nb_plot))
    fig, ax = plt.subplots(nb_row,nb_row)

    # Configuration
    configuration = data_computation.data.configuration
    nbh = configuration.nbh
    nbv = configuration.nbv
    nblocs_side = configuration.nblocs_side
    nsites = configuration.nsites
    nalpha = configuration.nalpha
    nbeta = configuration.nbeta

    basis = bC.Basis(configuration, data_computation.determinants)
        
    # Spectrum
    eigenstates = data_computation.eigenstates
    eigvects = eigenstates.eigvects


    # Computation of the probability of presence of a hole on each site of the lattice
    for ind_vect in range(min(nb_plot, eigvects.shape[1])):
        vect = eigvects[:,[ind_vect]]
        proba = bC.get_probability_spin_rectangular_lattice(basis, vect, 0)

    # Interpolation of the probability density function
        x = list(range(nblocs_side * nbh))
        y = list(range(nblocs_side * nbv))
        z = proba.tolist()

        f = interp.interp2d(x, y, z)

        lattice =[[],[]]
        for ex in x:
            for ey in y:
                lattice[0].append(ex)
                lattice[1].append(ey)
    
    # Plotting and saving the result

        xnew = np.arange(-0.1,nblocs_side*nbh-1+0.1,0.01)
        ynew = np.arange(-0.2,nblocs_side*nbv-1+0.2,0.01)
        znew = f(xnew, ynew)


        
        ax[ind_vect // nb_row, ind_vect % nb_row].pcolormesh(xnew, ynew, znew, cmap = 'magma', shading = 'auto')
        ax[ind_vect // nb_row, ind_vect % nb_row].scatter(lattice[0], lattice[1], marker = 'o', c = 'r', s=15**2)
        ax[ind_vect // nb_row, ind_vect % nb_row].set_title("{}-th lowest eigenstate".format(ind_vect))

    fig.suptitle("Probability density of the holes in the lattice ({}x{})".format(nblocs_side * nbh, nblocs_side * nbv))

    plt.tight_layout()

    # File Name
    bool_interactions = data_computation.data.bool_interactions
    bt = bool_interactions.bt
    bz = bool_interactions.bz
    bij = bool_interactions.bij
    bSD = bool_interactions.bSD
    btN = bool_interactions.btN
    bV = bool_interactions.bV
    bV_embedding = bool_interactions.bV_embedding
    file_name = "./tJ-HamCsr/tJ-H{}/HolePresenceProbability/tJ-H{}{}{}".format(nsites, nsites, nalpha, nbeta)
    if bt and bz and bij and bSD and btN and bV and bV_embedding:
        file_name += "Full.png"
    else:
        extensions = {"bt":"t", "bij":"J", "bSD":"SD", "btN":"tN", "bV":"V", "bV_embedding":"emb"}
        for name in dir(bool_interactions):
            if name in extensions and getattr(bool_interactions, name):
                file_name += extensions[name]
        file_name += ".png"

    try :
        plt.savefig(file_name, 
            dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            metadata=None)
    except FileNotFoundError :
        os.mkdir("./tJ-HamCsr/tJ-H{}/HolePresenceProbability/".format(str(nsites)))
        plt.savefig(file_name, 
            dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            metadata=None)
    plt.close()

def save_electron_probability_lattice(data_computation, nb_state): 
    """
    Plot the alpha, beta and charge distribution in the lattice for the nb_state-th lowest eigenstate
    Args:
        data_computations (class initialComputations.DataComputation): the data from computations
        nb_states (int): the index of the excited state to plot
    """

    fig, (axA, axB, axC) = plt.subplots(3,1)

    # Configuration
    configuration = data_computation.data.configuration
    nbh = configuration.nbh
    nbv = configuration.nbv
    nblocs_side = configuration.nblocs_side
    nsites = configuration.nsites
    nalpha = configuration.nalpha
    nbeta = configuration.nbeta

    basis = bC.Basis(configuration, data_computation.determinants)
        
    # Spectrum
    eigenstates = data_computation.eigenstates
    eigvects = eigenstates.eigvects


    # Computation of the probability of presence of a hole on each site of the lattice
    if nb_state < eigvects.shape[1]:
        vect = eigvects[:,[nb_state]]
        proba_alpha = bC.get_probability_spin_rectangular_lattice(basis, vect, 1)
        proba_beta = bC.get_probability_spin_rectangular_lattice(basis, vect, -1)

    # Interpolation of the probability density function
        x = list(range(nblocs_side*nbh))
        y = list(range(nblocs_side*nbv))
        
        z_alpha = proba_alpha.tolist()
        z_beta = proba_beta.tolist()
        z_charge = []
        for ind_row in range(len(z_alpha)):
            row = []
            for ind_col in range(len(z_alpha[0])):
                row.append(z_alpha[ind_row][ind_col] - z_beta[ind_row][ind_col])
            z_charge.append(row)

        f_alpha = interp.interp2d(x, y, z_alpha)
        f_beta = interp.interp2d(x, y, z_beta)
        f_charge = interp.interp2d(x, y, z_charge)

        lattice =[[],[]]
        for ex in x:
            for ey in y:
                lattice[0].append(ex)
                lattice[1].append(ey)
    
    # Plotting and saving the result

        xnew = np.arange(-0.1, nblocs_side * nbh - 1 + 0.1, 0.01)
        ynew = np.arange(-0.2, nblocs_side * nbv - 1 + 0.2, 0.01)
        
        z_alpha_new = f_alpha(xnew, ynew)
        z_beta_new = f_beta(xnew, ynew)
        z_charge_new = f_charge(xnew, ynew)
        
        # Alpha electrons
        imA = axA.pcolormesh(xnew, ynew, z_alpha_new, cmap='autumn', shading='auto')
        axA.scatter(lattice[0], lattice[1], marker='o', c='black', s=10**2)
        axA.set_title("Alpha electrons density")
        fig.colorbar(imA, ax=axA)
        # Beta electrons
        imB = axB.pcolormesh(xnew, ynew, z_beta_new, cmap='winter', shading='auto')
        axB.scatter(lattice[0], lattice[1], marker='o', c='black', s=10**2)
        axB.set_title("Beta electrons density")
        fig.colorbar(imB, ax=axB)
        # Charge density
        imC = axC.pcolormesh(xnew, ynew, z_charge_new, cmap='coolwarm', shading='auto')
        axC.scatter(lattice[0], lattice[1], marker='o', c='black', s=10**2)
        axC.set_title("Charge density")
        fig.colorbar(imC, ax=axC)

        fig.suptitle("{}-th lowest state in the lattice ({}x{})".format(nb_state, nblocs_side * nbh, nblocs_side * nbv))

        plt.tight_layout()
        if nb_state == 0:
            try:
                plt.savefig("./tJ-HamCsr/tJ-H{}/SpinDistribution/GroundState/tJ-H{}{}{}.png".format(\
                    str(nsites), str(nsites), str(nalpha), str(nbeta)), 
                    dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    metadata=None)
            except FileNotFoundError :
                os.mkdir(path = "./tJ-HamCsr/tJ-H{}/SpinDistribution/GroundState".format(nsites))
                plt.savefig("./tJ-HamCsr/tJ-H{}/SpinDistribution/GroundState/tJ-H{}{}{}.png".format(\
                    str(nsites), str(nsites), str(nalpha), str(nbeta)), 
                    dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    metadata=None)

        else:
            try :
                plt.savefig("./tJ-HamCsr/tJ-H{}/SpinDistribution/Excited{}/tJ-H{}{}{}.png".format(\
                    str(nsites), str(nb_state), str(nsites), str(nalpha), str(nbeta)), 
                    dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    metadata=None)
            except FileNotFoundError :
                os.mkdir("./tJ-HamCsr/tJ-H{}/SpinDistribution/Excited{}".format(str(nsites), str(nb_state)))
                plt.savefig("./tJ-HamCsr/tJ-H{}/SpinDistribution/Excited{}/tJ-H{}{}{}.png".format(\
                    str(nsites), str(nb_state), str(nsites), str(nalpha), str(nbeta)), 
                    dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    metadata=None)
        plt.close()

### Amplitudes

def plot_determinants_amplitude_repartition(data_computation, nb_subplots): 
    """
    Returns the inverse cumulative distribution function \
        for the coefficients of the lowests eigenstates F(x) = Card(vect[i]>=x)
    Args:
        data_computations (class initialComputations.DataComputation): the data from computations
        nb_subplots (int): the number of subplots
    """
    
    nb_plot = int(mp.floor(mp.sqrt(nb_subplots))**2)
    nb_row = int(mp.sqrt(nb_plot))
    fig, ax = plt.subplots(nb_row, nb_row)

    # Configuration
    configuration = data_computation.data.configuration
    nsites = configuration.nsites
    nalpha = configuration.nalpha
    nbeta = configuration.nbeta
        
    # Spectrum
    eigenstates = data_computation.eigenstates
    eigvects = eigenstates.eigvects

    # Computation of the inverse cumulative distribution function for each eigenstate
    for ind_vect in range(min(nb_plot, eigvects.shape[1])):
        vect = np.sort(np.absolute(eigvects[:, [ind_vect]]), 0)
        max_coeff = np.amax(vect)
        x = np.arange(-5 * 10**-4, max_coeff + 5 * 10 ** -4, 10 ** -4)
        y = np.zeros(x.shape)
        dim = vect.shape[0]
        for ind_i in range(y.shape[0]):
            ind_j = 0
            while ind_j < dim and abs(vect[ind_j, 0]) < x[ind_i]:
                ind_j += 1
            y[ind_i] = dim - ind_j

    # Plotting and saving the result
      
        ax[ind_vect // nb_row, ind_vect % nb_row].plot(x, y)
        ax[ind_vect // nb_row, ind_vect % nb_row].set_title("{}-th lowest eigenstate".format(ind_vect))

    fig.suptitle("Number of coefficients in the eigenstates superior to the value plotted against")

    plt.tight_layout()
        
    plt.savefig("./tJ-HamCsr/tJ-H{}/DeterminantsAmplitude/tJ-H{}{}{}.png".format(\
        str(nsites), str(nsites), str(nalpha), str(nbeta)), 
        dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        metadata=None)

    plt.close()

### Embedding

def plot_embedding_effect_on_determinants(data_computations_cluster, data_computations_embedding, nb_det):
    """
    Compares the effects of the embedding on the prevalence of Slater determinants in the ground state of the cluster.
    Args:
        data_computations_cluster (class initialComputations.DataComputations): the data from the computations on the isolated cluster
        data_computations_embedding (class initialComputations.DataComputations): the data from the computations on the embedded cluster
        nb_det (int): the number of determinants highest in energy to display
    """

    _, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 15)) #(ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))

    color_cluster = 'tomato'
    color_embedding = 'skyblue'

    # Isolated cluster

    configuration = data_computations_cluster.data.configuration
    nsites = configuration.nsites
    nbeta = configuration.nbeta
    nalpha = configuration.nalpha

    ground_state = np.power(abs(data_computations_cluster.eigenstates.eigvects[:,0]), 2).tolist()
    highests_det = {ind : ground_state.index(ndet) for ind, ndet in enumerate(list(reversed(sorted(ground_state)))[:nb_det])}

    # Embedded cluster

    ground_state_emb = np.power(abs(data_computations_embedding.eigenstates.eigvects[:,0]), 2).tolist()
    highests_det_emb = {ind : ground_state_emb.index(ndet) for ind, ndet in enumerate(list(reversed(sorted(ground_state_emb)))[:nb_det])}
    
    print("[emb] Highest determinant in embedding, with amplitude {} :\n\n{}".format(abs(data_computations_embedding.eigenstates.eigvects[:,0][highests_det_emb[0]])**2, bC.get_displayable_rectangular_lattice(data_computations_embedding.data.configuration, data_computations_embedding.determinants[highests_det_emb[0]])))

    # Domain
    #max_coeff = max(max(ground_state), max(ground_state_emb))

    # Plotting highest determinants amplitudes

    indices = range(nb_det)
    coeffs = [ground_state[highests_det[ind]] for ind in indices]
    coeffs_emb = [ground_state_emb[highests_det_emb[ind]] for ind in indices]

    ax1.bar(x=indices, height=coeffs, width=0.6, align='center', color=color_cluster, alpha=0.6, zorder=sum(coeffs), label='cluster')
    ax1.bar(x=indices, height=coeffs_emb, width=0.5, align='center', color=color_embedding, alpha=0.6, zorder=sum(coeffs_emb), label='embedding')
        
    ax1.set_xlabel("Determinants with highest amplitude")
    ax1.set_ylabel("Amplitude")
    ax1.legend()

    # Plotting influence on the density of determinants amplitudes
    #nb_stp = 100
    #amplitudes = np.linspace(start=0, stop=max_coeff * (1 + nb_stp) / nb_stp, num=nb_stp + 1) 
    #density_cluster = [len([coeff for coeff in ground_state if (coeff >= amplitude and coeff < amplitudes[ind + 1])]) / nb_stp for ind, amplitude in enumerate(amplitudes[:-1])]
    #density_embedding = [len([coeff for coeff in ground_state_emb if (coeff >= amplitude and coeff < amplitudes[ind + 1])]) / nb_stp for ind, amplitude in enumerate(amplitudes[:-1])]

    #ax2.plot(amplitudes[:-1], density_cluster, color=color_cluster, label='cluster', linestyle='--')
    #ax2.plot(amplitudes[:-1], density_embedding, color=color_embedding, label='embedding', linestyle='-')
    
    #ax2.set_xlabel("Amplitude")
    #ax2.set_ylabel("Density of determinants")
    #ax2.legend()

    #ax2_inset = ax2.inset_axes([0.2, 0.2, 0.7, 0.7])
    #ax2_inset.plot(amplitudes[:-1], density_cluster, color=color_cluster, label='cluster', linestyle='--')
    #ax2_inset.plot(amplitudes[:-1], density_embedding, color=color_embedding, label='embedding', linestyle='-')
    #ax2_inset.set_ylim(-0.01, 0.05)
    

    #fig.suptitle("Influence of the static embedding in a Néel lattice on the amplitudes of the Slater determinants.")

    try :
        plt.savefig("./tJ-HamCsr/tJ-H{}/AmplitudesStaticEmbeddingNeel/tJ-H{}{}{}.png".format(\
            str(nsites), str(nsites), str(nalpha), str(nbeta)))
        plt.savefig("./tJ-HamCsr/tJ-H{}/AmplitudesStaticEmbeddingNeel/tJ-H{}{}{}.pdf".format(\
            str(nsites), str(nsites), str(nalpha), str(nbeta)))
    except OSError:
        os.mkdir(path = "./tJ-HamCsr/tJ-H{}/AmplitudesStaticEmbeddingNeel/".format(nsites))
        plt.savefig("./tJ-HamCsr/tJ-H{}/AmplitudesStaticEmbeddingNeel/tJ-H{}{}{}.png".format(\
            str(nsites), str(nsites), str(nalpha), str(nbeta)))
        plt.savefig("./tJ-HamCsr/tJ-H{}/AmplitudesStaticEmbeddingNeel/tJ-H{}{}{}.pdf".format(\
            str(nsites), str(nsites), str(nalpha), str(nbeta)))

