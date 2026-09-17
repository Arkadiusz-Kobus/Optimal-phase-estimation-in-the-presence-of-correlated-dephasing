Fig. 4
#!/usr/bin/env python

from __future__ import annotations
from warnings import warn
import cvxpy as cp
import numpy as np
import math
import mosek
import matplotlib.pyplot as plt
from quantro import *
from scipy.linalg import sqrtm
from scipy.special import binom
from scipy.optimize import minimize
from scipy.linalg import expm



def matrix_sqrt(A):
    
    eigenvalues, eigenvectors = np.linalg.eig(A)

    
    sqrt_eigenvalues = np.diag(np.sqrt(eigenvalues))  # Take sqrt of diagonal matrix


   
    A_sqrt = eigenvectors @ sqrt_eigenvalues @ np.linalg.inv(eigenvectors)

    return A_sqrt


def directsumdiag(A, B):
    rowsA, colsA = A.shape
    rowsB, colsB = B.shape

    rowsResult = rowsA + rowsB
    colsResult = colsA + colsB

    sum_matrix = np.zeros((rowsResult, colsResult), dtype=A.dtype)

    sum_matrix[:rowsA, :colsA] = A
    sum_matrix[rowsA:, colsA:] = B

    return sum_matrix


def directsumoffdiag(A, B):
    rowsA, colsA = A.shape
    rowsB, colsB = B.shape

    rowsResult = rowsA + rowsB
    colsResult = colsA + colsB

    sum_matrix = np.zeros((rowsResult, colsResult), dtype=A.dtype)

    sum_matrix[:rowsA, colsResult - colsA:] = A
    sum_matrix[rowsA:, :colsB] = B

    return sum_matrix

def stationary(no_of_state, index, s):
    n = no_of_state

    lambda_value = (math.gamma(n) / (math.gamma((n - 1) - (index - 1) + 1) * math.gamma(index))) * pow(s, (n - index)) * pow((1 - s), (index - 1))

    return lambda_value


def tran_matrix(t2, no_of_state, p, q):
    for round in range(3, no_of_state + 1):
        m = t2.shape[0] + 1

        t3 = np.zeros((m, m), dtype=t2.dtype)
        tfinal = np.zeros((m, m), dtype=t2.dtype)
        zero = np.zeros((1, 1), dtype=t2.dtype)

        t3 = (p * directsumdiag(t2, zero)) + ((1 - p) * directsumoffdiag(t2, zero)) + ((1 - q) * directsumoffdiag(zero, t2)) + (q * directsumdiag(zero, t2))

        for i in range(m):
            for j in range(m):
                if i != 0 and i != m - 1:
                    tfinal[i, j] = t3[i, j] / 2.0
                else:
                    tfinal[i, j] = t3[i, j]


        t2 = tfinal

    return t2

def anti_diagonal_unit_matrix(n):
    return np.fliplr(np.eye(n))

def basis_vector(n, k):
    state = np.zeros(n)  # Create an array of zeros of size n
    state[k] = 1         # Set the k-th element to 1 (the basis state)
    return state


def krauses_t_neg(c, t2, p, q, s, no_of_state):
    kraus_neg = []

    for j in range(no_of_state):
        for k in range(no_of_state):
            proj = np.outer(basis_vector(no_of_state, k), basis_vector(no_of_state, j).conj())
            value = tran_matrix(t2, no_of_state, p, q)[j, k]
            
            if abs(value) < 1e-13:
                value = 0.0
                                                      
                kraus_neg.append(np.sqrt(value)*(proj@anti_diagonal_unit_matrix(no_of_state)))
            else:
                (kraus_neg.append(np.sqrt(value)*(proj@anti_diagonal_unit_matrix(no_of_state))))





    return kraus_neg



def krauses_t(c, t2, p, q, s, no_of_state):
    kraus = []

    for j in range(no_of_state):
        for k in range(no_of_state):
            proj = np.outer(basis_vector(no_of_state, k), basis_vector(no_of_state, j).conj())
            value = tran_matrix(t2, no_of_state, p, q)[j, k]
           
            if abs(value) < 1e-13:
                value = 0.0
                                                     
                kraus.append(np.sqrt(value)*(proj))
            else:
                (kraus.append(np.sqrt(value)*(proj)))





    return kraus


def local_kraus(eta):


    sigma_z = np.array([[1, 0], [0, -1]])
    loc_kra=[]
   

    loc_kra.append(np.kron(np.identity(no_of_state), np.sqrt(eta) * np.eye(2, 2)))
    loc_kra.append(np.kron(np.identity(no_of_state),np.sqrt(1 - eta) * sigma_z))

    return loc_kra


def is_valid_kraus(kraus, no_of_state):
    identity = np.eye(no_of_state, dtype=np.complex128)  # Identity matrix
    kraus_sum = np.zeros((no_of_state, no_of_state),dtype=np.complex128)  # Initialize sum

    for K in kraus:
        # print(K)
        kraus_sum += K.conj().T @ K  # Compute sum of K_i† K_i
    # print(kraus_sum)

    return np.allclose(kraus_sum, identity, atol=1e-10)  # Check if it's close to identity




def unitary(psi, increment, no_of_state):
    sz = np.array([[1, 0], [0, -1]])
    U = expm(-1j * 0.5 * (- psi[0]) * sz)
    U_incre = expm(-1j * 0.5 * increment * sz)
    uni = []
    for i in range (no_of_state):
        U1 = U
        uni.append(U1)
        U = U1@U_incre

    return uni



no_of_state = 5 # We have discretized the Gaussian noise by using the Rouwenhorst method. So this 5 states are approximately demonstrating the Gaussian noise as discussed in the main text
D = 1.0
c = 0.4 # Here one can change the values of the correlation parameter to make the plots for other values of c
asy_bound = []
X =[]
Y1 =[]
Y2 =[]
Y3 =[]
p = (1+np.sqrt(abs(c)))/2
q = (1+np.sqrt(abs(c)))/2
s = (1.0-q)/(2-(p+q))
t2 = np.array([[p, 1-p],
               [1-q, q]])
sz = np.array([[1,0],[0,-1]])
#psi = [4.9272135623727085, 0.0, -4.9272135623727085]
#psi = [17.165150807567176, 5.721716935855726, -5.721716935855724, -17.165150807567173]
psi = [14.52500000002643, 7.262500000013215, 0.0, -7.262500000013215, -14.52500000002643]
increment = (2*psi[0])/(no_of_state-1)
eta = np.exp(-D / 2)
kraus_list = krauses_t(c, t2, p, q, s, no_of_state)
kraus_list_neg = krauses_t_neg(c, t2, p, q, s, no_of_state)


dkrauses = [np.zeros((no_of_state, no_of_state)), ] * (no_of_state**2)
vtheta = np.zeros((2*no_of_state, 2*no_of_state))
dvtheta = np.zeros((2*no_of_state, 2*no_of_state))
for k in range (no_of_state):


    basis_proj = np.outer(basis_vector(no_of_state, k), basis_vector(no_of_state, k).conj())
    vtheta = vtheta + np.kron(basis_proj,unitary(psi, increment, no_of_state)[k])
    dvtheta = dvtheta +   np.kron(basis_proj,(-1j * 0.5) *(sz@unitary(psi, increment, no_of_state)[k]))

V_channel = ParamChannel(krauses = [vtheta,], dkrauses = [dvtheta,], env_dim= no_of_state)
t_channel = ParamChannel(krauses=krauses_t(c, t2, p, q, s, no_of_state), dkrauses=dkrauses, env_dim=no_of_state)
#t_channel_neg = ParamChannel(krauses=krauses_t_neg(c, t2, p, q, s, no_of_state), dkrauses=dkrauses, env_dim=no_of_state)
#full_channel = t_channel_neg *  V_channel * t_channel
full_channel = t_channel *  V_channel * t_channel
full_channel = ParamChannel(choi=full_channel.choi(), dchoi=full_channel.dchoi(), env_dim=no_of_state)




all_X = []
all_Y1 = []
for N in range (3,21,3):
    
    bound_1 = ad_bounds_correlated(full_channel, nmax= N,  block_size=3)
    



    X, Y1 = bound_1
    
    print(X)
    print(Y1)

    

all_X.extend(X)
all_Y1.extend(Y1/X)
print(all_X)
print(all_Y1)
data_to_save = np.column_stack((all_X, all_Y1))
np.savetxt("new_phase_5_gaussian_bound_finite_block_3_correction_c_0pt4.dat", data_to_save, comments='')
print("Data saved to data.dat")



















Fig. 3 (For calculating the bounds)
#!/usr/bin/env python

from __future__ import annotations


from warnings import warn
import cvxpy as cp
import numpy as np
import math
import mosek
import matplotlib.pyplot as plt
from quantro import *
from scipy.linalg import sqrtm
from scipy.special import binom
from scipy.optimize import minimize
from scipy.linalg import expm



def matrix_sqrt(A):
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Compute square root of eigenvalues
    sqrt_eigenvalues = np.diag(np.sqrt(eigenvalues))  # Take sqrt of diagonal matrix


    # Reconstruct the square root matrix
    A_sqrt = eigenvectors @ sqrt_eigenvalues @ np.linalg.inv(eigenvectors)

    return A_sqrt


def directsumdiag(A, B):
    rowsA, colsA = A.shape
    rowsB, colsB = B.shape

    rowsResult = rowsA + rowsB
    colsResult = colsA + colsB

    sum_matrix = np.zeros((rowsResult, colsResult), dtype=A.dtype)

    sum_matrix[:rowsA, :colsA] = A
    sum_matrix[rowsA:, colsA:] = B

    return sum_matrix


def directsumoffdiag(A, B):
    rowsA, colsA = A.shape
    rowsB, colsB = B.shape

    rowsResult = rowsA + rowsB
    colsResult = colsA + colsB

    sum_matrix = np.zeros((rowsResult, colsResult), dtype=A.dtype)

    sum_matrix[:rowsA, colsResult - colsA:] = A
    sum_matrix[rowsA:, :colsB] = B

    return sum_matrix

def stationary(no_of_state, index, s):
    n = no_of_state

    lambda_value = (math.gamma(n) / (math.gamma((n - 1) - (index - 1) + 1) * math.gamma(index))) * pow(s, (n - index)) * pow((1 - s), (index - 1))

    return lambda_value


def tran_matrix(t2, no_of_state, p, q):
    for round in range(3, no_of_state + 1):
        m = t2.shape[0] + 1

        t3 = np.zeros((m, m), dtype=t2.dtype)
        tfinal = np.zeros((m, m), dtype=t2.dtype)
        zero = np.zeros((1, 1), dtype=t2.dtype)

        t3 = (p * directsumdiag(t2, zero)) + ((1 - p) * directsumoffdiag(t2, zero)) + ((1 - q) * directsumoffdiag(zero, t2)) + (q * directsumdiag(zero, t2))

        for i in range(m):
            for j in range(m):
                if i != 0 and i != m - 1:
                    tfinal[i, j] = t3[i, j] / 2.0
                else:
                    tfinal[i, j] = t3[i, j]


        t2 = tfinal

    return t2

def anti_diagonal_unit_matrix(n):
    return np.fliplr(np.eye(n))

def basis_vector(n, k):
    state = np.zeros(n)  # Create an array of zeros of size n
    state[k] = 1         # Set the k-th element to 1 (the basis state)
    return state


def krauses_t_neg(c, t2, p, q, s, no_of_state):
    kraus_neg = []

    for j in range(no_of_state):
        for k in range(no_of_state):
            proj = np.outer(basis_vector(no_of_state, k), basis_vector(no_of_state, j).conj())
            value = tran_matrix(t2, no_of_state, p, q)[j, k]
            # print(value)
            if abs(value) < 1e-13:
                value = 0.0
                # print(value)                                     
                kraus_neg.append(np.sqrt(value)*(proj@anti_diagonal_unit_matrix(no_of_state)))
            else:
                (kraus_neg.append(np.sqrt(value)*(proj@anti_diagonal_unit_matrix(no_of_state))))





    return kraus_neg



def krauses_t(c, t2, p, q, s, no_of_state):
    kraus = []

    for j in range(no_of_state):
        for k in range(no_of_state):
            proj = np.outer(basis_vector(no_of_state, k), basis_vector(no_of_state, j).conj())
            value = tran_matrix(t2, no_of_state, p, q)[j, k]
            # print(value)
            if abs(value) < 1e-13:
                value = 0.0
                # print(value)                                     
                kraus.append(np.sqrt(value)*(proj))
            else:
                (kraus.append(np.sqrt(value)*(proj)))





    return kraus


def local_kraus(eta):


    sigma_z = np.array([[1, 0], [0, -1]])
    loc_kra=[]
    loc_kra.append(np.kron(np.identity(no_of_state), np.sqrt(eta) * np.eye(2, 2)))
    loc_kra.append(np.kron(np.identity(no_of_state),np.sqrt(1 - eta) * sigma_z))

    return loc_kra


def is_valid_kraus(kraus, no_of_state):
    identity = np.eye(no_of_state, dtype=np.complex128)  # Identity matrix
    kraus_sum = np.zeros((no_of_state, no_of_state),dtype=np.complex128)  # Initialize sum

    for K in kraus:
        
        kraus_sum += K.conj().T @ K  # Compute sum of K_i† K_i
   

    return np.allclose(kraus_sum, identity, atol=1e-10)  # Check if it's close to identity




def unitary(psi, increment, no_of_state):
    sz = np.array([[1, 0], [0, -1]])
    U = expm(-1j * 0.5 * (- psi[0]) * sz)
    U_incre = expm(-1j * 0.5 * increment * sz)
    uni = []
    for i in range (no_of_state):
        U1 = U
        uni.append(U1)
        U = U1@U_incre

    return uni



no_of_state = 5
N = 30
D = 1.0

c_in = -1.0
c_final = 1.0
step = 0.01



X = []
X1 =[]
Y1 =[]
Y2 =[]
Y3 =[]
for c_loop in np.arange(c_in, c_final, step):
    c = round(c_loop, 2)
    
    p = (1+np.sqrt(abs(c)))/2
    q = (1+np.sqrt(abs(c)))/2
    s = (1.0-q)/(2-(p+q))
    t2 = np.array([[p, 1-p],
                   [1-q, q]])
    
    sz = np.array([[1,0],[0,-1]])
    
    #psi = [4.9272135623727085, 0.0, -4.9272135623727085]
    #psi = [17.165150807567176, 5.721716935855726, -5.721716935855724, -17.165150807567173]
    psi = [14.52500000002643, 7.262500000013215, 0.0, -7.262500000013215, -14.52500000002643]
    increment = (2*psi[0])/(no_of_state-1)
    eta = np.exp(-D / 2)
    kraus_list = krauses_t(c, t2, p, q, s, no_of_state)
    kraus_list_neg = krauses_t_neg(c, t2, p, q, s, no_of_state)
    
    
    dkrauses = [np.zeros((no_of_state, no_of_state)), ] * (no_of_state**2)
    vtheta = np.zeros((2*no_of_state, 2*no_of_state))
    dvtheta = np.zeros((2*no_of_state, 2*no_of_state))
    for k in range (no_of_state):
    
    
        basis_proj = np.outer(basis_vector(no_of_state, k), basis_vector(no_of_state, k).conj())
        vtheta = vtheta + np.kron(basis_proj,unitary(psi, increment, no_of_state)[k])
        dvtheta = dvtheta +   np.kron(basis_proj,(-1j * 0.5) *(sz@unitary(psi, increment, no_of_state)[k]))
    
    
    V_channel = ParamChannel(krauses = [vtheta,], dkrauses = [dvtheta,], env_dim= no_of_state)
    t_channel = ParamChannel(krauses=krauses_t(c, t2, p, q, s, no_of_state), dkrauses=dkrauses, env_dim=no_of_state)
    #t_channel_neg = ParamChannel(krauses=krauses_t_neg(c, t2, p, q, s, no_of_state), dkrauses=dkrauses, env_dim=no_of_state)
    #full_channel = t_channel_neg *  V_channel * t_channel
    full_channel = t_channel *  V_channel * t_channel
    full_channel = ParamChannel(choi=full_channel.choi(), dchoi=full_channel.dchoi(), env_dim=no_of_state)
    
    bound_1 = ad_bounds_correlated(full_channel, nmax= N,  block_size=3) #QCE upper bound calculation
        
    
    X1, Y_bound = bound_1

    X.append(c)
    Y1.append(Y_bound/X1)
    
    
data_to_save = np.column_stack((X, Y1))
np.savetxt("new_phase_5_gaussian_bound_finite_block_3_correction_N_30.dat", data_to_save, comments='')
print("Data saved to data.dat")



Fig.3 (For calculating the Fisher information)

#!/usr/bin/env python

from __future__ import annotations


from warnings import warn
import cvxpy as cp
import numpy as np
import math
import mosek
import matplotlib.pyplot as plt
from qmetro import *
#from qutip import basis
from scipy.linalg import sqrtm
from scipy.special import binom
from scipy.optimize import minimize
from scipy.linalg import expm
from scipy.linalg import eigvals

#Code for binary phase Fisher information: MPS
c = 0.4
p = 0.85
ep = 2.0*np.arccos(np.sqrt(p))

sx = np.array([[0,1],[1,0]])
sy = np.array([[0,-1j],[1j,0]])
sz = np.array([[1,0],[0,-1]])



plus_state = np.array([[1], [0]])
minus_state = np.array([[0], [1]])

plus_density_matrix = plus_state @ plus_state.T
minus_density_matrix = minus_state @ minus_state.T


minus_plus_density_matrix = minus_state @ plus_state.T
plus_minus_density_matrix = plus_state @ minus_state.T

U_eplust = expm(-1j*0.5*ep*sz)
U_eminust = expm(1j*0.5*ep*sz)

d_U_eplust = (-1j*0.5)*(sz@expm(-1j*0.5*ep*sz))
d_U_eminust = (-1j*0.5)*(sz@expm(1j*0.5*ep*sz))


k1 = np.sqrt((1+c)/2)*np.kron(plus_density_matrix,U_eplust)
k2 = np.sqrt((1-c)/2)*np.kron(minus_plus_density_matrix,U_eplust)
k3 = np.sqrt((1-c)/2)*np.kron(plus_minus_density_matrix,U_eminust)
k4 = np.sqrt((1+c)/2)*np.kron(minus_density_matrix,U_eminust)

# print(k1@k1.conj().T + k2@k2.conj().T+k3@k3.conj().T+k4@k4.conj().T)

dk1 = np.sqrt((1+c)/2)*np.kron(plus_density_matrix,d_U_eplust)
dk2 = np.sqrt((1-c)/2)*np.kron(minus_plus_density_matrix,d_U_eplust)
dk3 = np.sqrt((1-c)/2)*np.kron(plus_minus_density_matrix,d_U_eminust)
dk4 = np.sqrt((1+c)/2)*np.kron(minus_density_matrix,d_U_eminust)

chan1 = ParamChannel(krauses=[k1, k2, k3, k4], dkrauses=[dk1, dk2,dk3,dk4], env_dim=2)

X =[]
Y1 =[]
Y2 =[]
Y3 =[]
Y4 =[]
Y5 =[]
Y6 =[]
N = 30


    
max_qfi22 = 0.0
max_qfi24 = 0.0
max_qfi42 = 0.0
max_qfi44 = 0.0
for rep in range (5):


    fisher_22 = iss_tnet_parallel_qfi(chan1, number_of_channels=N, ancilla_dim=1, mps_bond_dim=2, measure_bond_dim=2)
    fisher_24 = iss_tnet_parallel_qfi(chan1, number_of_channels=N, ancilla_dim=1, mps_bond_dim=2, measure_bond_dim=4)
    fisher_42 = iss_tnet_parallel_qfi(chan1, number_of_channels=N, ancilla_dim=1, mps_bond_dim=4, measure_bond_dim=2)
    fisher_44 = iss_tnet_parallel_qfi(chan1, number_of_channels=N, ancilla_dim=1, mps_bond_dim=4, measure_bond_dim=4)
    
    max_qfi22 = max(fisher_22[0], max_qfi22)
    max_qfi24 = max(fisher_24[0], max_qfi24)
    max_qfi42 = max(fisher_42[0], max_qfi42)
    max_qfi44 = max(fisher_44[0], max_qfi44)



Y1.append(max_qfi22/N)
Y2.append(max_qfi24/N)
Y3.append(max_qfi42/N)
Y4.append(max_qfi44/N)

#----------------------------------------------------------------------------------------------------------------------

#Code for approximated Gaussian noise with arbitrary number of phases: Fisher information MPS

def directsumdiag(A, B):
    rowsA, colsA = A.shape
    rowsB, colsB = B.shape

    rowsResult = rowsA + rowsB
    colsResult = colsA + colsB

    sum_matrix = np.zeros((rowsResult, colsResult), dtype=A.dtype)

    sum_matrix[:rowsA, :colsA] = A
    sum_matrix[rowsA:, colsA:] = B

    return sum_matrix


def directsumoffdiag(A, B):
    rowsA, colsA = A.shape
    rowsB, colsB = B.shape

    rowsResult = rowsA + rowsB
    colsResult = colsA + colsB

    sum_matrix = np.zeros((rowsResult, colsResult), dtype=A.dtype)

    sum_matrix[:rowsA, colsResult - colsA:] = A
    sum_matrix[rowsA:, :colsB] = B

    return sum_matrix

def stationary(no_of_state, index, s):
    n = no_of_state

    lambda_value = (math.gamma(n) / (math.gamma((n - 1) - (index - 1) + 1) * math.gamma(index))) * pow(s, (n - index)) * pow((1 - s), (index - 1))

    return lambda_value

def tran_matrix(t2, no_of_state, p, q):
    for round in range(3, no_of_state + 1):
        m = t2.shape[0] + 1

        t3 = np.zeros((m, m), dtype=t2.dtype)
        tfinal = np.zeros((m, m), dtype=t2.dtype)
        zero = np.zeros((1, 1), dtype=t2.dtype)

        t3 = (p * directsumdiag(t2, zero)) + ((1 - p) * directsumoffdiag(t2, zero)) + ((1 - q) * directsumoffdiag(zero, t2)) + (q * directsumdiag(zero, t2))

        for i in range(m):
            for j in range(m):
                if i != 0 and i != m - 1:
                    tfinal[i, j] = t3[i, j] / 2.0
                else:
                    tfinal[i, j] = t3[i, j]


        t2 = tfinal




    return t2



def basis_vector(n, k):
    state = np.zeros(n)  # Create an array of zeros of size n
    state[k] = 1         # Set the k-th element to 1 (the basis state)
    return state


def unitary(psi, increment, no_of_state):

    sz = np.array([[1, 0], [0, -1]])
    U = expm(-1j * 0.5 * (- psi[0]) * sz)
    U_incre = expm(-1j * 0.5 * increment * sz)
    uni = []
    for i in range (no_of_state):
        U1 = U
        uni.append(U1)
        U = U1@U_incre

    return uni

def anti_diagonal_unit_matrix(n):
    return np.fliplr(np.eye(n))

def kraus_operators(no_of_state,s,eta):
    kraus= []
    for i in range (no_of_state):
        for j in range (no_of_state):
            basis_proj = np.outer(basis_vector(no_of_state, i), basis_vector(no_of_state, j).conj().T)
            kraus.append(np.sqrt(tran_matrix(t2, no_of_state, p, q)[i,j])*(np.kron(basis_proj,unitary(psi, increment, no_of_state)[j])))

    return kraus



def d_kraus_operators(no_of_state,s,eta):
    dkraus= []
    for i in range (no_of_state):
        for j in range (no_of_state):
            basis_proj = np.outer(basis_vector(no_of_state, i), basis_vector(no_of_state, j).conj().T)
            dkraus.append(np.sqrt(tran_matrix(t2, no_of_state, p, q)[i,j])*(np.kron(basis_proj,((-1j*0.5)*(sz@unitary(psi, increment, no_of_state)[j])))))

    return dkraus


no_of_state = 4
count = 1   #change this for the smoothness of the plot
c = 0.0
D = 1.0
p = (1+c)/2
q = (1+c)/2
s = (1.0-q)/(2-(p+q))
t2 = np.array([[p, 1-p],
               [1-q, q]])

eta = np.exp(-D / 2)

psi = [17.165150807567176, 5.721716935855726, -5.721716935855724, -17.165150807567173]
sz = np.array([[1,0],[0,-1]])
increment = (2*psi[0])/(no_of_state-1)
                                                                                                                                                                                                                                                            

kraus_op = kraus_operators(no_of_state,s,eta)
dkraus_op = d_kraus_operators(no_of_state,s,eta)


chan1 = ParamChannel(krauses=kraus_op, dkrauses=dkraus_op, env_dim=no_of_state)

X =[]
Y1 =[]
Y2 =[]
Y3 =[]
Y4 =[]
N = 30



    
max_qfi12 = 0.0
max_qfi22 = 0.0
max_qfi24 = 0.0
max_qfi22 = 0.0
max_qfi24 = 0.0
max_qfi44 = 0.0
max_qfi3 = 0.0
max_qfi42 = 0.0
max_qfi44 = 0.0
    for rep in range (count):

        # for adaptive strategy
       fisher_11 = iss_tnet_adaptive_qfi(chan1, number_of_channels= N, ancilla_dim=4)
       fisher_22 = iss_tnet_adaptive_qfi(chan1, number_of_channels=N, ancilla_dim=5)
       fisher_33 = iss_tnet_adaptive_qfi(chan1, number_of_channels=N, ancilla_dim=6)
        # for parallel strategy
       fisher_12 = iss_tnet_parallel_qfi(chan1, number_of_channels=N, ancilla_dim=1, mps_bond_dim=1, measure_bond_dim=2)
       fisher_22 = iss_tnet_parallel_qfi(chan1, number_of_channels=N, ancilla_dim=1, mps_bond_dim=2, measure_bond_dim=2)
       fisher_24 = iss_tnet_parallel_qfi(chan1, number_of_channels=N, ancilla_dim=1, mps_bond_dim=2, measure_bond_dim=4)
       fisher_44 = iss_tnet_parallel_qfi(chan1, number_of_channels=N, ancilla_dim=1, mps_bond_dim=4, measure_bond_dim=4)

       max_qfi12 = max(fisher_12[0], max_qfi12)
       max_qfi22 = max(fisher_22[0], max_qfi22)
       
       max_qfi24 = max(fisher_24[0], max_qfi24)
       max_qfi44 = max(fisher_44[0], max_qfi44)
       

    


    
    Y1.append(max_qfi12/N)
    Y2.append(max_qfi22/N)
    Y3.append(max_qfi24/N)
    Y4.append(max_qfi44/N)
    

    




file_path = 'para_c_0pt0_phase_4_gaussian_fisher_12_22_24_44_N_30.dat'

with open(file_path, 'w') as file:
    for x_value, y1_value, y2_value, y3_value, y4_value in zip(X, Y1,Y2,Y3,Y4):
          file.write(f"{x_value} {y1_value}  {y2_value}  {y3_value} {y4_value}\n")







