# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Definitions & Imports

# +
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display, Markdown
# -

## Custom functions
import FermionSystem as fst
import KitaevTools as kt


def printmd(item):
    display(Markdown(item._repr_markdown_()))


# ## Additional Functions

# %load_ext cython

# + language="cython"
# ## For speed, using cython here to define the hamiltonian function utilised
# ## by the sparse solver.
# ## Should ideally be placed inside the FermionSystem file but did not yet figure out
# ## how to generate the c code (platform independently) inside the .py file
#
# ## TO DO: find out how to use this when the values are complex
#
# import numpy as np
# cimport numpy as cnp
#
# def hamiltonian_matvec(
#     cnp.ndarray[double complex, ndim=1] x,
#     cnp.ndarray[int, ndim=1] rows,
#     cnp.ndarray[int, ndim=1] cols,
#     cnp.ndarray[double complex, ndim=1] vals
# ):
#     cdef int r, c, i
#     cdef double complex v
#     cdef cnp.ndarray[double complex, ndim=1] result = np.zeros_like(x)
#     cdef int n = rows.shape[0]
#
#     for i in range(n):
#         r = rows[i]
#         c = cols[i]
#         v = vals[i]
#         result[r] = result[r] + v * x[c]
#         result[c] = result[c] + v * x[r]
#     return result
# -

# # Basic Functions

# ## States and Operators

# +
## FermionSystem provides functions for acting with 'operators' on Fock states using binary operations
## where both states and operators are represented by integers.
## Requires specifying the number of fermionic sites N:
N = 2
fs = fst.FockSystemBase(N)

## Each integer represents a fock state in binary, where each site covers two bits (00 = 0, 01 = down, 10 = up, 11 = up+down)
fs.vis_state(0, displ=True)
fs.vis_state(1, displ=True)
fs.vis_state(2, displ=True)
fs.vis_state(3, displ=True)
fs.vis_state(0b100, displ=True)
fs.vis_state(0b111, displ=True)
fs.vis_state(0b1111, displ=True)

## The size of the system N is only relevant for the logic related to the operators
## Integers exceeding the largest possible state still just exist, but
## an operator acting on it will only be able to act on the first N sites
fs.vis_state(2000, displ=True)

# +
## Operators are similarly represented by integers, storing the position on the fock state to act on
# ## + a bit specifying if the operator is creation or annihilation.
## The largest bit sets the purpose, such that creation operators are always larger numbers than annihilation operators (for sorting in normal order)

# operator to create a spin up in site 1:
operator = fs.operator("creation", 1, "up")
display(
    Markdown(
        f"Created operator {fs.vis_oper(operator)}, represented by the number: {operator}"
    )
)
# -

# ## Acting on states

# +
# The action of operators on fock states is handled by 'act_oper', returning the new states and the signs
operator = fs.operator("creation", 1, "up")
state = 0b111

new_state, sign = fs.act_oper(operator, np.array([state]))
display(
    Markdown(
        f'{fs.vis_oper(operator)}{fs.vis_state(state)} = {"-" if sign[0] ==-1 else ""}{fs.vis_state(new_state[0])}'
    )
)

# +
# A sequence of operators is represented as a list of ints
# Applying the sequence to states uses fs.act_oper_list, which returns only states that are not destroyed by the sequence
fs = fst.FockSystemBase(3)

oper_1 = fs.operator("creation", 0, "up")
oper_2 = fs.operator("creation", 1, "up")
CAR_up_up = [oper_2, oper_1]

## Here: applying the CAR operator to the list of all possible Fock states
old_states, new_states, signs = fs.act_oper_list(CAR_up_up, np.arange(2**6))

## Show results of the operation
display(
    Markdown(f"Non-zero terms for operator {fs.vis_oper_list(CAR_up_up,displ=False)}")
)
for state, new_state, parity in zip(old_states, new_states, signs):
    if parity == 1:
        display(
            Markdown(
                f"{fs.vis_oper_list(CAR_up_up)}{fs.vis_state(state)} = {fs.vis_state(new_state)}"
            )
        )
    else:
        display(
            Markdown(
                f"{fs.vis_oper_list(CAR_up_up)}{fs.vis_state(state)} = -{fs.vis_state(new_state)}"
            )
        )
# -

# ## Convert between states/operators

# +
## Convert from a state to the creation operators

fs = fst.FockSystemBase(N=4)
state = 0b11011001

opers = fs.state_to_oper_list(state)
display(Markdown(f"{fs.vis_state(state)} = {fs.vis_oper_list(opers)}{fs.vis_state(0)}"))
# -

# ## Normal Ordering

# +
## Normal ordering of the states requires sorting the list of operators (integers) in size
## This is now done with a bubble sort, which straightforwardly allows tracking the number of swaps

fs = fst.FockSystemBase(N=3)

## Example of an non-ordered list of operators:
oper_list = [
    fs.operator("creation", 1, "up"),
    fs.operator("annihilation", 0, "down"),
    fs.operator("creation", 1, "down"),
    fs.operator("creation", 2, "down"),
]
display(Markdown(f"Original sequence: {fs.vis_oper_list(oper_list)} ({oper_list})"))

## Retrieving the list in order + the sign
normal_ordered, sign = fs.normal_order_naive(oper_list)
display(
    Markdown(
        f'Normal order: {"-" if sign < 1 else ""}{fs.vis_oper_list(normal_ordered)} ({normal_ordered})'
    )
)
# -

# ## Bra-Kets

# +
## <phi|(operators)|phi> matrices can be calculated with bra_oper_ket function
## Where (phi) is an array storign the weights of the system eigenvectors

fs = fst.FockSystemBase(N=2)
states = fs.fock_states

## Phi are the weights of the eigenstates. For this example just the I matrix
phi = np.zeros((len(states), len(states)))
np.fill_diagonal(phi, 1)

## Make an operator list. This simple example returns a matrix representing which Fock states are connected by CAR_down_down
CAR_down_down = [fs.operator("creation", 0, "down"), fs.operator("creation", 1, "down")]
operators = [fs.normal_order_naive(CAR_down_down)[0]]

## Calculate matrix
M = fs.bra_oper_ket(states, phi, operators)

## Visualize outcome
non_zero_terms = np.where(M != 0)
for row, col in zip(non_zero_terms[0], non_zero_terms[1]):
    display(
        Markdown(
            f'{fs.vis_oper_list(CAR_down_down)}{fs.vis_state(col)} = {"-" if M[row][col]<0 else ""}{fs.vis_state(row)}'
        )
    )


# -

# # OperSequence -> multiplication, addition etc

# +
## The OperSequence wrapper class is included for visualisation and construction of more complicated operators.
# -

# Wrapping simple operators in OperSequence allows for constructing more complicated obejcts
c_up = fst.FockOperSequence([[fs.op("cr", 0, "up")]])
c_dwn = fst.FockOperSequence([[fs.op("cr", 0, "dwn")]])
a_up = fst.FockOperSequence([[fs.op("a", 0, "up")]])
a_dwn = fst.FockOperSequence([[fs.op("a", 0, "dwn")]])

## Addition
maj = c_up + a_up
maj

## Subtraction
maj = c_up - a_up
maj

## Assigns weights
ex > [2, 2j]

## Shift all operators to another site
c_up >> 3

## Exponentation
ex = c_up + a_up
ex**2

## Multiplication
2 * c_up * c_dwn

## conjugation
c_up > 1j  # assign complex weight as example
~c_up

# ## More complex example

## Create in one go a more complex string of operators
an, cr, up, dwn = "a", "c", "u", "d"
weights = [1j, -1, 1, -1, 1j, -1j]
operators = [
    [fs.op(an, 0, up), fs.op(cr, 0, up), fs.op(an, 0, dwn)],
    [fs.op(an, 0, dwn), fs.op(cr, 0, dwn), fs.op(cr, 0, up)],
    [fs.op(an, 0, up)],
    [fs.op(an, 0, dwn), fs.op(cr, 0, dwn), fs.op(an, 0, up)],
    [fs.op(cr, 0, dwn)],
    [fs.op(an, 0, up), fs.op(cr, 0, up), fs.op(cr, 0, dwn)],
]
t = fst.FockOperSequence(operators, weights)

t

t + t

t**3

## Shift the site of all operators
t_shifted = t >> 1

t_shifted * (~t)

# +
## Sanity check: the operator ^4 = -1
# -

t**4

# # Constructing Hamiltonian

c_up = fst.FockOperSequence([[fs.op("cr", 0, "up")]])
c_dwn = fst.FockOperSequence([[fs.op("cr", 0, "dwn")]])
a_up = fst.FockOperSequence([[fs.op("a", 0, "up")]])
a_dwn = fst.FockOperSequence([[fs.op("a", 0, "dwn")]])

# Create down-down CAR and assign an initial value of 20
CAR_dd = c_dwn * (c_dwn >> 1) > 20
CAR_dd

## Create down-down ECT and assign an initial value of 20
ECT_dd = c_dwn * (a_dwn >> 1) > 20
ECT_dd

## Create mu and assign a weight of 0
mu_d = c_dwn * a_dwn > 0
mu_d

N = 2
H_base = [
    (CAR_dd, range(0, N - 1), "d_dd", "\u0394"),
    (ECT_dd, range(0, N - 1), "t_dd", "t"),
    (mu_d, range(0, N), "mu_d", "\u03bc"),
]


def create_H(H_base):
    H_vals, H_types, H_symbols = {}, {}, {}
    H_terms, H_signs = [], []
    for h_basis in H_base:
        base_opers = h_basis[0]
        for i in h_basis[1]:
            add_oper = base_opers >> i
            oper_as_str = fs.oper_list_to_str(add_oper.oper_list[0])
            H_vals[oper_as_str] = add_oper.weights[0]
            H_symbols[oper_as_str] = "$" + h_basis[3] + "^" + f"{i}" + "$"
            H_types[f"{h_basis[2]}_{i}"] = oper_as_str
            H_terms.append(add_oper.oper_list[0])

            sign = -1 if int(np.sign(add_oper.weights[0])) == -1 else 1
            H_signs.append(sign)
    return H_terms, H_signs, H_vals, H_types, H_symbols


# # Solving Hamiltonian: Effective Kitaev Model

# ## Constructing and solving system

# +
## Generating a Hamiltonian requires creating the list of operator-lists
## The function 'generate_kitaev_hamiltonian' generates all nearest neighbour interactions + chemical potentials + Us
N = 2
fs = fst.FockSystemBase(N, store_fock_states=False)
hamiltonian = kt.make_kitaev_hamiltonian(fs)
operator_list = hamiltonian[0]
display(Markdown(f"Hamiltonian representation: {operator_list}"))

## A mapping is needed to map  operators -> values and readable symbols, such that the H_params can be entered as readable
## A function 'map_H_params' maps both the operator lists and the H_params below to the same string representation.
H_params = {
    "d_dd": [0] * (N - 1),
    "d_uu": [0] * (N - 1),
    "d_ud": [20] * (N - 1),
    "d_du": [20] * (N - 1),
    "t_dd": [20] * (N - 1),
    "t_uu": [20] * (N - 1),
    "t_ud": [0] * (N - 1),
    "t_du": [0] * (N - 1),
    "mu": [0] * N,
    "Ez": [0] * N,
    "U": [0] * N,
}
H_vals, H_symbols, H_types = kt.map_H_params_kitaev(fs, H_params)

str = "H = "
for oper in operator_list:
    str += f"{H_symbols[fs.oper_list_to_str(oper)]}" + f"{fs.vis_oper_list(oper)}" + " + "
display(Markdown(str[:-2]))

# +
# An additional class (ParitySystem) is relevant for cases where a division of the fock states
# into odd and even parity sectors is desired (i.e.: the Kitaev chain)
N = 3

## Dictionary with all relevant parameters
## Parameter names used here are arbitrary, a function is required to link them to the correct 'operators' ('map_H_params_Kitaev' in this case)
H_params = {
    "d_dd": [0] * (N - 1),
    "d_uu": [0] * (N - 1),
    "d_ud": [20] * (N - 1),
    "d_du": [20] * (N - 1),
    "t_dd": [20] * (N - 1),
    "t_uu": [20] * (N - 1),
    "t_ud": [0] * (N - 1),
    "t_du": [0] * (N - 1),
    "mu": [0] * N,
    "Ez": [0] * N,
    "U": [0] * N,
}
fs = fst.FockSystemBase(N)

## Generate a list of all relevant operators for the Kitaev chain (with interactions)
## This is just a list with lists of integers representing the operator sequences
generate_kitaev_hamiltonian = partial(kt.make_kitaev_hamiltonian, fs)

## Links operators to understandable parameter names
generate_mapping = partial(kt.map_H_params_kitaev, fs, H_params)

## Create a ParitySystem (builds on FermionSystem): seperates the fock states into odd and even parity states
## and applies the provided Hamiltonian to calculate non-zero terms for later use
chain = fst.ParitySystem(
    N=N,
    H_generator=generate_kitaev_hamiltonian,
    H_mapping=generate_mapping,
    Ez_inf=True,
    U_inf=True,
    sparse_function=hamiltonian_matvec,
)
chain.gather_H()
## Once chain is constructed, can use solve_system to get eigenvalues and eigenvector
## Two methods are included:
## 'linalg' -> uses numpy's linalg.eigh, requires constructing the matrix
## 'sparse' -> uses scipy's LinearOperator class, does not require explicitly filling a matrix
##             returns the lowest N eigenvalues and eigenvectors (set by 'n_values')
##             The method is faster for N>4, if only a small number of eigenvalues are needed
display(Markdown("**Method: Sparse**"))
E_odd, E_even, phi_odd, phi_even = chain.solve_system(method="sparse", n_values=1)
print(f"Odd ground state energy: {np.round(E_odd[0],3)}")
print(f"Odd ground state: {chain.vis_state_list(chain.odd_states,phi_odd[0])}\n")
print(f"Even ground state energy: {np.round(E_even[0],3)}")
print(f"Even ground state: {chain.vis_state_list(chain.even_states,phi_even[0])}\n")
print("________________")
display(Markdown("**Method: Linalg**"))
E_odd, E_even, phi_odd, phi_even = chain.solve_system(method="linalg")
print(f"Odd ground state energy: {np.round(E_odd[0],3)}")
print(f"Odd ground state: {chain.vis_state_list(chain.odd_states,phi_odd[0])}\n")
print(f"Even ground state energy: {np.round(E_even[0],3)}")
print(f"Even ground state: {chain.vis_state_list(chain.even_states,phi_even[0])}\n")
# -

## To show all eigenstates and solutions:
chain = kt.make_kitaev_chain(
    2, H_params, Ez_inf=False, U_inf=True
)  # shortcut for the Kitaev chain construction
chain.eigenstates(only_ground_states=False, only_energies=False)

# +
## For small Hamiltonians and debugging: print a symbolic Hamiltonian (only shows non-diagonal terms)

chain = kt.make_kitaev_chain(
    2, H_params, Ez_inf=False, U_inf=True
)  # shortcut for the Kitaev chain construction
# chain.eigenstates(only_ground_states=False,only_energies=False)
chain.show_hamiltonian(parity="even")
# -

## The hamiltonian parameters can be updated without having to redo the entire matrix
chain = kt.make_kitaev_chain(
    2, H_params, Ez_inf=False, U_inf=True, make_arrays=True
)  # shortcut for the Kitaev chain construction
chain.update_H_param_list(["mu_d_0"], 5, update_matrix=True)
print("Updated to have value of 5 for mu_0_down:")
chain.show_hamiltonian(parity="even", numeric=True)


# ## Energy Spectrum

# +
## To calculate a simplified energy spectrum (for comparing to finite bias conductace measurements)
## the function'energy_spectrum' extracts transitions and probablities for transitions
## between even and odd ground states to excited states.
## This gives a simplified representation, but provides speed over solving the rate equation (below)
N = 2


## Create Figure
fig, axs = plt.subplots(ncols=N, figsize=(N * 2.2, 2.5))
for ax in axs:
    ax.grid(False)
    ax.set_ylim([-150, 150])
    ax.set_xlabel("$\\delta \\mu$")
    ax.set_ylabel("$E_{T}$")

## Set parameters
H_params = {
    "d_dd": [20] * (N - 1),
    "d_uu": [20] * (N - 1),
    "d_ud": [20] * (N - 1),
    "d_du": [20] * (N - 1),
    "t_dd": [20] * (N - 1),
    "t_uu": [20] * (N - 1),
    "t_ud": [0] * (N - 1),
    "t_du": [0] * (N - 1),
    "mu": [0] * N,
    "Ez": [0] * N,
    "U": [0] * N,
}

## Create chain
chain = kt.make_kitaev_chain(
    N,
    H_params,
    Ez_inf=True,
    U_inf=True,
    make_arrays=True,
    sparse_function=hamiltonian_matvec,
)

## Range and parameters to vary
mu_range = np.linspace(-100, 100, 100)
vary_params = [f"mu_d_{i}" for i in range(chain.N)] + [
    f"mu_u_{i}" for i in range(chain.N)
]  # All mu_down parameters int he chain

## PLot energy spectrum for all sites
timed = 0
kt.energy_spectrum(chain, vary_params, mu_range, range(N), fig, axs, plot=True)
plt.tight_layout()
# -

# ## Conductance from rate equation

lead_params = {"gammas": [0.001], "kBT": 0.002, "dV": 0.001}

# +
N = 5
H_params = {
    "d_dd": [20e-3] * (N - 1),
    "d_uu": [20e-3] * (N - 1),
    "d_ud": [0] * (N - 1),
    "d_du": [0] * (N - 1),
    "t_dd": [20e-3] * (N - 1),
    "t_uu": [20e-3] * (N - 1),
    "t_ud": [0] * (N - 1),
    "t_du": [0] * (N - 1),
    "mu": [0] * N,
    "Ez": [0] * N,
    "U": [0] * N,
}
chain = kt.make_kitaev_chain(
    N,
    H_params,
    Ez_inf=True,
    U_inf=True,
    make_arrays=True,
    sparse_function=hamiltonian_matvec,
)  # shortcut for the Kitaev chain construction

vary_params = [f"mu_d_{i}" for i in range(N)] + [f"mu_u_{i}" for i in range(N)]
param_range = np.linspace(-100e-3, 100e-3, 50)
bias_range = np.linspace(-100e-3, 100e-3, 50)

## When getting conductance, can truncate the number of eigenvectors to pass to the rate equation (given by n_values)
lead_params["gammas"] = [0.001] * N
Gs_dataset = kt.conductance_spectrum(
    chain,
    vary_params,
    param_range,
    bias_range,
    sites=[i for i in range(N)],
    lead_params=lead_params,
    plot=True,
    method="linalg",
    n_values=30,
)


# +
N = 2
H_params = {
    "d_dd": [0e-3] * (N - 1),
    "d_uu": [0e-3] * (N - 1),
    "d_ud": [20e-3] * (N - 1),
    "d_du": [20e-3] * (N - 1),
    "t_dd": [np.sqrt(2) * 20e-3] * (N - 1),
    "t_uu": [np.sqrt(2) * 20e-3] * (N - 1),
    "t_ud": [0] * (N - 1),
    "t_du": [0] * (N - 1),
    "mu": [0] * N,
    "Ez": [0] * N,
    "U": [0] * N,
}

chain = kt.make_kitaev_chain(
    N,
    H_params,
    Ez_inf=False,
    U_inf=True,
    make_arrays=True,
    sparse_function=hamiltonian_matvec,
)  # shortcut for the Kitaev chain construction

vary_params = ["mu_d_1", "mu_u_1"]
param_range = np.linspace(-100e-3, 100e-3, 80)
bias_range = np.linspace(-100e-3, 100e-3, 80)

## When getting conductance, can truncate the number of eigenvectors to pass to the rate equation (given by n_values)
lead_params["gammas"] = [0.001] * N
Gs_dataset = kt.conductance_spectrum(
    chain,
    vary_params,
    param_range,
    bias_range,
    sites=[i for i in range(N)],
    lead_params=lead_params,
    plot=True,
    method="linalg",
    n_values=50,
)


# +
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(4, 4))
for ax_row in axs:
    for ax in ax_row:
        ax.set_xlabel("$\\mu_0$")
        ax.set_ylabel("$\\mu_1$")

## Set up a Kitaev chain
N = 2
H_params = {
    "d_dd": [30e-3] * (N - 1),
    "d_uu": [30e-3] * (N - 1),
    "d_ud": [0] * (N - 1),
    "d_du": [0] * (N - 1),
    "t_dd": [30e-3] * (N - 1),
    "t_uu": [30e-3] * (N - 1),
    "t_ud": [0] * (N - 1),
    "t_du": [0] * (N - 1),
    "mu": [0] * N,
    "Ez": [0] * N,
    "U": [0] * N,
}
chain = kt.make_kitaev_chain(N, H_params, Ez_inf=True, U_inf=True, make_arrays=True)

## Select parameters and ranges
x_range = np.linspace(-100e-3, 100e-3, 80)
y_range = np.linspace(-100e-3, 100e-3, 80)

## Instruct to vary mu_0 vs mu_1
vary_x = ["mu_d_0"]
vary_y = ["mu_d_1"]
sites = [0, 1]

lead_params["gammas"] = [0.001] * N
results = kt.charge_stability_diagram(
    chain,
    vary_x,
    x_range,
    vary_y,
    y_range,
    sites=sites,
    lead_params=lead_params,
    method="linalg",
)
count = 0
for i in range(len(sites)):
    for j in range(len(sites)):
        if i != j:
            axs[i][j].pcolormesh(results[f"G_{sites[i]}{sites[j]}"], cmap="RdBu_r")
        else:
            vmax = np.max(results[f"G_{sites[i]}{sites[j]}"])
            axs[i][j].pcolormesh(
                results[f"G_{sites[i]}{sites[j]}"], cmap="RdBu_r", vmin=-vmax, vmax=vmax
            )

plt.tight_layout()

# -

# ## Odd-Even Phase Space

# +
N = 3
phi = np.arccos(-1 / 4)
delta = 30

H_params = {
    "d_dd": [0] * (N - 1),
    "d_uu": [0] * (N - 1),
    "d_ud": [delta, delta * (np.exp(-1j * phi))],
    "d_du": [delta, delta * np.exp(-1j * phi)],
    "t_dd": [40e-3] * (N - 1),
    "t_uu": [40e-3] * (N - 1),
    "t_ud": [0] * (N - 1),
    "t_du": [0] * (N - 1),
    "mu": [0] * N,
    "Ez": [0] * N,
    "U": [0] * N,
}
chain = kt.make_kitaev_chain(
    N,
    H_params,
    Ez_inf=False,
    U_inf=True,
    make_arrays=True,
    sparse_function=hamiltonian_matvec,
)
## Select parameters and ranges
mu_range = np.array(np.linspace(-60, 60, 150))
t_range = np.linspace(0, 2 * delta, 150)

## Instruct to vary all mu parameters and all t parameters
vary_mu = [f"mu_d_{i}" for i in range(N)] + [f"mu_u_{i}" for i in range(N)]
vary_t = [f"t_uu_{i}" for i in range(N - 1)] + [f"t_dd_{i}" for i in range(N - 1)]

## Calculate energy differences
results = np.array(
    kt.phase_space(chain, vary_t, t_range, vary_mu, mu_range, T=1, disable=False)[0]
)

res_array = np.zeros(np.shape(results))
threshhold = 0.001
pos = np.where(results > threshhold)
neg = np.where(results < -threshhold)
res_array[pos] = 1
res_array[neg] = -1
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4, 4))
ax.pcolormesh(t_range, mu_range, res_array, cmap="RdBu_r", shading="auto")
ax.set_title("Odd/Even GS phase diagram, $\\phi$=" + f"{phi}")
ax.set_xlabel("t/$\\Delta$")
ax.set_ylabel("$\\mu$/$\\Delta$")
ax.plot([0, 60], [0, 0], linestyle="dashed", color="black")
ax.plot(
    [np.sqrt(2) * delta, np.sqrt(2) * delta], [-60, 60], color="black", linestyle="dashed"
)
# -

# ## Block Diagonalize

# +
## Block diagonalize the matrix and visualize with graphs or with plt.matshow
## block_diagonalize function reorders the basis states and regenerates the hamiltonian
## If a state is nonzero but only couples to itself I.e. only shows up on the diagonal) it is removed from the basis

# +
N = 3
H_params = {
    "d_dd": [0] * (N - 1),
    "d_uu": [0] * (N - 1),
    "d_ud": [40] * (N - 1),
    "d_du": [40] * (N - 1),
    "t_dd": [40] * (N - 1),
    "t_uu": [40] * (N - 1),
    "t_ud": [0] * (N - 1),
    "t_du": [0] * (N - 1),
    "mu": [1] * N,
    "Ez": [0] * N,
    "U": [0] * N,
}
times = []

chain = kt.make_kitaev_chain(
    N,
    H_params,
    Ez_inf=False,
    U_inf=True,
    make_arrays=True,
    sparse_function=hamiltonian_matvec,
)

# -

blocks = chain.block_diagonalize(print_result=True, graph=True)
