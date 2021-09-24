import math as mp
import numpy as np
import numpy.linalg as lin
np.set_printoptions(linewidth=1000)
import scipy.sparse as sp
import scipy.sparse.linalg as slin

import copy

import interactions as itr

from tqdm import tqdm, trange # Install tqdm via pip

class Diagonalization :
    """
    Class containing parameters for the diagonalization procedure.
    """
    def __init__(self, nb_val, stp, tol, dim_max, initial_guess=None):
        """
        Args:
            nb_val (int): the number of lowest eigenstates to compute
            stp (int): maximum number of steps
            tol (float): tolerance for the stopping criteria
            dim_max (int): limit to the dimension of the search space
            initial_guess (np.array): the initial guess for the eigenstates
        """
        self.nb_val = nb_val
        self.stp = int(stp)
        self.tol = tol
        self.dim_max = dim_max
        self.initial_guess = initial_guess

class Eigenstates :
    """
    Class describing the elements of a list of eigenstates.
    """
    def __init__(self, eigvals, spins, eigvects):
        """
        Args:
            eigvals (list<float>): the list of eigenvalues
            spins (list<float>): the list of spins of the corresponding eigenvectors
            eigvects (list<np.array>): the list of corresponding eigenvectors
        """
        self.eigvals = eigvals
        self.spins = spins
        self.eigvects = eigvects

### Davidson algorithm - projections onto unexplored eigenspaces

def davidson_algorithm(A,stp,tol,prec):
    """
    Davidson algorithm, yield the lowest eigenstate of the matrix A.
    Args:
        A (np.array): diagonalizable matrix
        stp (int): maximal number of steps
        tol (float): tolerance for the stopping criteria
        prec (float): expected precision of the final eigenstate
    """
    n = A.shape[0]
    v0 = np.random.random((n,1))
    V = np.array(v0)
    H = np.matmul(V.T, np.matmul(A, V))
    
    (eigval, eigvect) = lin.eigh(H)
    val = list(eigval)
    ind_lowest = val.index(min(val))
    lambdaj = eigval[ind_lowest]
    yj = eigvect[:, [ind_lowest]]
    
    uj = np.dot(V, yj)
    rj = np.dot(A, uj) - lambdaj*uj
    if lin.norm(rj) < tol :
        return (lambdaj, uj)
    else:
        Mj = lin.inv(np.diag(np.diag(A)) - lambdaj*np.eye(n))
        tj = np.dot(Mj, rj)
        V = perform_modified_gram_schmidt(np.append(V, tj, axis=1))
    while lin.norm(rj) > tol:
        for _ in range(stp):
            H = np.matmul(V.T, np.matmul(A, V))
            
            (eigval, eigvect) = lin.eigh(H)
            val = list(eigval)
            ind_lowest = val.index(min(val))
            lambdaj = eigval[ind_lowest]
            yj = eigvect[:, [ind_lowest]]
            
            uj = np.dot(V, yj)
            rj = np.dot(A, uj) - lambdaj*uj
            if lin.norm(rj) < tol:
                return (round(lambdaj, prec), uj)
            else:
                #Mj=lin.inv(np.diag(np.diag(A))-lambdaj*np.eye(n)) # Jacobi preconditionner
                Mj = np.eye(n) # Lanczos method
                tj = np.dot(Mj, rj)
                V = perform_modified_gram_schmidt(np.append(V, tj, axis=1))

    return (round(lambdaj, prec), uj)


def get_lowests_eig(eigvals, eigvects, nb_eig): 
    """
    Return the nb_eig lowests eigenvalues and associated eigenvectors
    Args:
        eigvals (list<float> or np.array): the list of eigenvalues
        eigvects (np.array): the corresponding eigenvectors
    Returns:
        val_sorted (list<float>): the nb_eig lowests eigenvalues
        vect_sorted (np.array): the corresponding eigenvectors
    """
    eigvals_temp = copy.deepcopy(eigvals)
    sorted_eigvals = list(sorted(eigvals_temp))[:nb_eig]
    val_sorted = sorted(list(set(sorted_eigvals)))
    sorted_indexes = []
    for val in val_sorted:
        val_indices = [ind for ind, elt in enumerate(eigvals_temp) if elt == val]
        val_indices = val_indices[: min(len(val_indices), nb_eig - len(sorted_indexes))]
        sorted_indexes += val_indices
    vect_sorted = eigvects[:, sorted_indexes]
    return (sorted_eigvals, vect_sorted)
    
def get_meaningful_corrections(krylov_basis, temp_corrections, droptol, eta, nu, dim_max):
    """
    Computes the meaningful corrections to perform in modified_davidson_algorithm
    Args:
        krylov_basis (np.array): the basis of the search subspace
        temp_corrections (np.array): computed corrections, residues multiplied by the preconditionning matrix
        droptol (float): drop-tolerance for the meaningful corrections
        eta (float): threshold in the precision for the corrections
        nu (int): the number of eigenstates yet to compute
        dim_max (int): limit to the dimension of the search space
    Returns:
        corrections (np.array): the meaningful corrections to perform
    """
    dim = temp_corrections.shape[0]
    corrections = np.copy(krylov_basis)
    block_size_new = 0
    max_orth = 2
    for corr in range(nu):
        ind_k = 0
        while True :
            old = lin.norm(temp_corrections[:, corr])
            for ind_j in range(corrections.shape[1]):
                temp_corrections[:, corr] = np.dot(np.eye(dim) - \
                    np.dot(corrections[:,ind_j][:,None], corrections[:,ind_j][None,:]), temp_corrections[:, corr])
            ind_k += 1
            if (ind_k == max_orth - 1) or lin.norm(temp_corrections[:, corr]) > eta*old:
                break
        if lin.norm(temp_corrections[:, corr]) > droptol:
            block_size_new += 1
            correction = temp_corrections[:, [corr]]/lin.norm(temp_corrections[:, corr])
            corrections = np.append(corrections, correction.tolist(), axis=1)
    return corrections[:,-block_size_new:]


def modified_davidson_algorithm_opt(A, nb_val, init_block_size, stp=100, tol=1e-3, droptol=1e-6, eta=0.1, dim_max=0):
    """
    Modified Davidson algorithm for simultaneous computation of the nb_val lowests eigenvectors of csr sparse matrix A0,\
        using the Davidson preconditionners (C = (diag(A) - lambda_i*I)^-1)
    Args:
        A (scipy.sparse.csr_matrix): the matrix we seek to compute the eigenstates
        nb_val (int): the number of eigenstates to compute
        init_block_size (int): initial dimension of the search subspace
        stp (int): maximum number of steps (default to 100)
        tol (float): tolerance for the stopping criteria on residues (default to 1e-3)
        eta (float): threshold in the precision for the corrections (0 << eta < 1, default to 0.1)
        droptol (float): drop-tolerance for the meaningful corrections (default to 1e-6)
        dim_max (int): limit to the dimension of the search space (default to the dimension of A)
    Returns:
        eigenstates (class Eigenstates): the eigenvalues and correponding eigenvectors
    """
    dim = A.shape[0]
    if dim_max == 0:
        dim_max = dim

    nu_0 = max(init_block_size, nb_val)
    nb_found = 0
    block_size = nu_0

    eigvals = []
    eigvects = np.zeros((dim, nb_val))

    krylov_basis = perform_modified_gram_schmidt(np.random.rand(dim, block_size))

    for _ in range(stp):
        # Krylov vectors
        mat_w = A.dot(krylov_basis) 
        # Rayleigh matrix
        mat_rayleigh = np.matmul(krylov_basis.T, mat_w)
        nu = min(mat_rayleigh.shape[0], nu_0 - nb_found)
        # Eigenpairs
        eigvals_temp, eigvects_temp = get_lowests_eig(*lin.eigh(mat_rayleigh), nu)
        # Ritz vectors
        ritz_vectors = np.matmul(krylov_basis, eigvects_temp)
        # Residues
        residues = np.zeros((dim, nu))
        for col in range(nu):
            residues[:,col] = eigvals_temp[col] * ritz_vectors[:,col] - np.dot(mat_w, eigvects_temp[:,col])
        conv_residues = [lin.norm(residues[:,col]) <= tol for col in range(nu)]
        nc = sum(conv_residues)
        # Deflation
        for col, res_ind in enumerate(conv_residues): # Adding converging eigenvector to the solutions
            nbc = 0
            if res_ind == True and nb_found < nb_val:
                eigvals.append(eigvals_temp[col])
                eigvects[:,nb_found + nbc] = ritz_vectors[:,col]
                ritz_vectors[:, col], ritz_vectors[:, nbc] = ritz_vectors[:, nbc], ritz_vectors[:, col]
                nbc += 1
        nb_found += nc
        if nb_found >= nb_val:
            return Eigenstates(eigvals, [], eigvects)
        # Correction of unsatisfactory residues
        temp_corrections = np.zeros((dim, nu))
        for ind in range(nc, nu):
            temp_corrections[:,ind] = np.dot(lin.inv(np.diag(A.diagonal()) - eigvals_temp[ind]*np.eye(dim)), \
            residues[:,ind])
        # Meaningful corrections
        corrections = get_meaningful_corrections(krylov_basis, temp_corrections, droptol, eta, nu, dim_max)
        block_size_new = corrections.shape[1]
        # Iteration
        if (krylov_basis.shape[0] > dim_max - block_size_new) or (nc != 0) or (block_size_new == 0) : # restart
            krylov_basis = perform_modified_gram_schmidt(np.concatenate((ritz_vectors[:,nc:], corrections), 1))
            block_size = nu - nc + block_size_new
        else : # expansion of the search subspace
            krylov_basis = perform_modified_gram_schmidt(np.concatenate((krylov_basis, corrections), 1))
            block_size = block_size_new
    return Eigenstates(eigvals, [], eigvects)

def modified_davidson_algorithm(A, nb_val, initial_guess=None, stp=100, tol=1e-3, dim_max=0, verbose=False):
    """
    Modified Davidson algorithm for simultaneous computation of the nb_val lowests eigenvectors of csr sparse matrix A0,\
        using the Davidson preconditionners (C = (diag(A) - lambda_i*I)^-1)
    Args:
        A (scipy.sparse.csr_matrix): the matrix we seek to compute the eigenstates
        nb_val (int): the number of eigenstates to compute
        initial_guess (np.array): the initial guesses for the eigenvectors
        stp (int): maximum number of steps (default to 100)
        tol (float): tolerance for the stopping criteria on residues (default to 1e-3)
        dim_max (int): limit to the dimension of the search space (default to the dimension of A)
        verbose (bool): Tells if we want to display information about the progression of computations in the shell
    Returns:
        eigenstates (class Eigenstates): the eigenvalues and correponding eigenvectors
    """
    dim = A.shape[0]
    if dim_max == 0:
        dim_max = dim
    nb_val = min(nb_val, dim_max)

    if initial_guess is None:
        initial_guess = np.random.rand(dim, nb_val)
    
    # Krylov basis (basis of the search space)
    if nb_val > 1 :
        krylov_basis = perform_modified_gram_schmidt(initial_guess)
    else:
        krylov_basis = initial_guess

    for _ in trange_verbose(stp, verbose):
        # Krylov vectors
        mat_w = A.dot(krylov_basis) 
        # Rayleigh matrix
        mat_rayleigh = np.matmul(krylov_basis.T, mat_w)
        # Eigenpairs
        eigvals_temp, eigvects_temp = get_lowests_eig(*lin.eigh(mat_rayleigh), nb_val)
        # Ritz vectors
        ritz_vectors = np.matmul(krylov_basis, eigvects_temp)
        # Residues
        residues = np.zeros((dim, nb_val))
        for col in range(nb_val):
            residues[:,col] = eigvals_temp[col] * ritz_vectors[:,col] - np.dot(mat_w, eigvects_temp[:,col])
        conv = True
        for col in range(nb_val):
            conv *= lin.norm(residues[:,col]) <= tol
        if conv == True:
            print_verbose("Norm of residues : {}".format([lin.norm(residues[:,col]) for col in range(nb_val)]), verbose)
            return Eigenstates(eigvals_temp, [], ritz_vectors)
        # Correction of unsatisfactory residues
        corrections = np.zeros((dim, nb_val))
        for ind in range(nb_val):
            preconditionner = sp.csr_matrix(\
                ([1 / (diag - eigvals_temp[ind]) for diag in A.diagonal()], (range(dim), range(dim)) ), \
                shape=(dim, dim))
            corrections[:,ind] = preconditionner.dot(residues[:,ind])
        # Iteration
        if krylov_basis.shape[0] > dim_max - nb_val :
            krylov_basis = perform_modified_gram_schmidt(np.concatenate((ritz_vectors, corrections), 1))
        else : # expansion of the search subspace
            krylov_basis = perform_modified_gram_schmidt(np.concatenate((krylov_basis, corrections), 1))
    print_verbose("Maximum number of iterations exceeded, Davidson procedure could not converge.", verbose)
    return Eigenstates(eigvals_temp, [], ritz_vectors)
    
    
### Gram-Schmidt procedures
    
def perform_modified_gram_schmidt(V):
    """
    Performs the modified Gram Schmidt algorithm on a set of vectors vects
    Args:
        V (np.array): a set of vectors
    Returns:
        Q (np.array): the orthonormalized set of vectors from vects
    """
    Q, _ = lin.qr(V) 
    return Q

    
def gramSchmidt(V): 
    """
    Gram-Schmidt orthonormalization procedure, orthonormalize the columns of the matrix V
    Args:
        V (np.array): the set of vectors to orthonormalize
    Returns:
        A (np.array): the orthonormalized set of vectors
    """
    dim = V.shape[0]
    A = np.zeros(V.shape) # Orthonormalized columns of V
    A[:,[0]] = V[:,[0]] / lin.norm(V[:,[0]])
    for col in range(1, dim):
        vect = V[:,[col]]
        for prec in range(col):
            vect_prec = A[:,[prec]]
            vect = vect - (np.dot(vect.T, vect_prec) / np.dot(vect_prec.T, vect_prec)) * vect_prec
        A[:,[col]] = vect / lin.norm(vect)
    return A
    
def perform_matrix_orthogonal_projection(A, vects): # Not complete
    """
    Return the matrix A projected and restrained onto the orthogonal space to Span(vects)
    Args:
        A (np.array): A matrix
        vects (np.array): an orthonormal basis of the subspace along which we want to project A
    Returns:
        A_proj (np.array): the projection and restriction of the matrix A onto :math:`vects^{\\perp}`
    """
    dim = A.shape[1]
    dim_sub = vects.shape[1]
    comp_basis = np.ones((dim, dim - dim_sub))
    basis = np.append(vects, comp_basis, axis=1)
    projector = perform_modified_gram_schmidt(basis) # orthogonal projector associated to the direct sum : Span(vects) + vects^{\perp}
    A_proj = np.matmul(projector.T, np.matmul(A, projector))[dim_sub:,dim_sub:]
    return A_proj
    
### Eigenvectors properties

def compute_spin_and_multiplicity(basis, eigvects): # Not complete
    """
    Computes the expectation value of S^2 for each computed eigenvector
    Args:
        basis (class baseConstruction.Basis): the basis of our configuration
        eigvect (np.array): the eigenvectors, expressed in this basis, that we want to calculate the spin multiplicity
    Returns:
        spins
        multiplicity
    """
    nb_val = eigvects.shape[1]
    spins = []
    multiplicity = []

    for ind in range(nb_val):
        spin = itr.compute_S2_exp_val(basis, eigvects[:,[ind]])
        spins.append(spin)
        multiplicity.append(itr.compute_spin_multiplicity(spin))
    return (spins,multiplicity)
    

### Utilities

def print_verbose(string, verbose=False):
    """
    Prints the string in the shell only if verbose is True.
    Args:
        string (str): the string to print in the shell
        verbose (bool): tells if we print or not string in the shell
    """
    if verbose:
        print("[dav] ", string)

def trange_verbose(steps, verbose=False):
    """
    Uses the tqdm package only if verbose == True
    Args:
        steps (int): our range
        verbose (bool): tells if we want or not to print informations about the progression of computations in the shell
    Returns:
        range_iter (range or trange): either a range or trange object depending on verbose
    """
    if verbose:
        return trange(steps)
    else:
        return range(steps)
    
