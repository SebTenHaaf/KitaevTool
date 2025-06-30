from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import block_diag
from scipy.sparse import coo_array

from typing import Callable
from functools import partial
from collections import defaultdict, Counter
import copy


from IPython.display import display, Markdown

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import c.fermion_operations as fo

### OBSELETE CLASS ####
## Class below has been scavanged to create the above data structures that are more general and achieve the same goal
## Class can soon be removed after scavaging some further functions (block diagonalizing for example)
class ParitySystem(FockSystemBase):
    def __init__(
        self,
        N: int,
        H_generator: Callable,
        H_mapping: Callable,
        sparse_function: Callable = None,
        Ez_inf: bool = False,
        U_inf: bool = False,
    ):
        """
        Class for specifically handling systems where a division into odd and even states is useful
        Args:
            N (int): the number of fermionic sites
            H_generator (Callable): a function that creates the list of operators representing H
            H_mapping (Callable): a fuction that assigns readable labels to the operators
            sparse_function (Callable): the function of LinearOperator to use when using scipy's sparse solver
            Ez_inf (bool): whether to restrict the subspaces to infinite Ez
            U_inf (bool): whether to restrict the subspaces to infinite U
        """
        #super().__init__(N)
        self.N=N
        self.fock_states = np.arange(0,2**(2*N))
        ## Restrict and divide fock space
        self.odd_states, self.even_states = self.restrict_and_sort_fockspace(
            Ez_inf=Ez_inf, U_inf=U_inf
        )

        self.fock_states = np.append(self.even_states, self.odd_states)
        self.odd_states_hash = {num: idx for idx, num in enumerate(self.odd_states)}
        self.even_states_hash = {num: idx for idx, num in enumerate(self.even_states)}

        ## Initialise H formatting dictionary for readability and control purposes
        self.H, self.H_signs = H_generator()
        self.H_vals, self.H_symbols, self.H_types = H_mapping()
        self.sparse_function = sparse_function

    def restrict_and_sort_fockspace(self, Ez_inf: bool, U_inf: bool):
        """
        Sorts Fock states into 'even' and 'odd' parities (determined by count of 1's in binary)
        Optionally allows restricting the Fock space to infinite U or infinite Ez
        Args:
            Ez_inf (bool): If True, excludes states with a 'spin up' set
            U_inf (bool): If True, excludes states with both 'spin up' and 'spin down' set for a single site
        Returns:
            odd_states, even_states (nd.array): Nnon-excluded fock states sorted by parity
        """
        all_states = self.fock_states

        ## If Ez is infinite -> exclude all states with a 1 in the odd powers of 2 position
        if Ez_inf:
            for spin_up_in_state in [2 ** (2 * n + 1) for n in range(self.N)]:
                mask_states = (all_states & spin_up_in_state) == 0
                all_states = all_states[(mask_states)]

        ## If otherwise U is infinite -> exclude all states with both 1's set for a given site
        elif U_inf:
            for spin_up_in_state, spin_down_in_state in zip(
                [2 ** (2 * n) for n in range(self.N)],
                [2 ** (2 * n + 1) for n in range(self.N)],
            ):
                mask_states_odd = (all_states & spin_up_in_state) != 0
                mask_states_even = (all_states & spin_down_in_state) != 0
                mask_states = ~(mask_states_odd & mask_states_even)
                all_states = all_states[(mask_states)]

        ## Create masks for odd and even parity
        even_states_mask = hamming_weight(all_states) % 2 == 0
        odd_states_mask = ~even_states_mask

        ## Seperate the states into odd and even parity
        even_states = all_states[even_states_mask]
        odd_states = all_states[odd_states_mask]
        return odd_states, even_states

    def act_H_on_subspace(self, states: np.ndarray, states_hash: dict):
        """
        Applies the stored hamiltonian to a set of Fock states
        Args:
            states (ndarray[int]): list of Fock states to act on
            states_hash (dict): dictionary mapping a state to its list index in 'states'
        Returns:
            rows (ndarray[int]): row indices of non-zero terms
            cols (ndarray[int]): column indices of non-zero terms
            pars (ndarray[int]): relative signs of operators for the non-zero terms
            type (ndarray[str]): the types of operators giving rise to non-zero terms
        """
        rows, cols, pars, type = [], [], [], []
        for rel_sign, h in zip(self.H_signs, self.H):
            type_str = self.oper_list_to_str(h)
            old_states, new_states, parities = self.act_oper_list(
                h, states, rel_sign=rel_sign
            )
            subspace_filt = [state in states_hash for state in new_states]
            old_states = old_states[subspace_filt]
            new_states = new_states[subspace_filt]
            parities = parities[subspace_filt]

            type.extend([type_str] * len(parities))
            rows.extend([states_hash.get(state) for state in old_states])
            cols.extend([states_hash.get(state) for state in new_states])
            pars.extend(parities.tolist())
        return (
            np.array(rows, dtype=np.int32),
            np.array(cols, dtype=np.int32),
            np.array(pars, dtype=np.int32),
            np.array(type),
        )

    def gather_H(self):
        """
        Act the Hamiltonian on the odd and even subspaces to generate row and column lists for non-zero term

        To do:
            assert that the states and hamiltonian have been loaded correctly
        """
        ## Gather Odd Hamiltonian
        self.odd_terms = self.act_H_on_subspace(self.odd_states, self.odd_states_hash)

        list_1, list_2 = map(
            list,
            zip(
                *[
                    (b, a) if a > b else (a, b)
                    for a, b in zip(self.odd_terms[0], self.odd_terms[1])
                ]
            ),
        )
        self.odd_terms[0][:] = np.array(list_1)
        self.odd_terms[1][:] = np.array(list_2)
        self.odd_vals = np.array(
            [self.H_vals[type] for type in self.odd_terms[3]], dtype=complex
        )

        ## Gather Even Hamiltonian
        self.even_terms = self.act_H_on_subspace(self.even_states, self.even_states_hash)
        list_1, list_2 = zip(
            *[
                (b, a) if a > b else (a, b)
                for a, b in zip(self.even_terms[0], self.even_terms[1])
            ]
        )
        self.even_terms[0][:] = np.array(list_1)
        self.even_terms[1][:] = np.array(list_2)
        self.even_vals = np.array(
            [self.H_vals[type] for type in self.even_terms[3]], dtype=complex
        )

    def block_diagonalize(self, print_result: bool = False, graph: bool = True):
        """
        Once hamiltonian has been generated, block diagonalizing is done by
        obtaining connected graphs from the row/col lists and reordering the basis states
        The new hamiltonian is obtained and optionally visualized

        Args:
            print_resu
        """
        states_block_basis, idx_components, block_components = [], [], []
        if graph:
            title = ["Even", "Odd"]
            color = ["lightcoral", "royalblue"]
        i = 0
        for pos_data, vals, states_hash, states in zip(
            [self.even_terms, self.odd_terms],
            [self.even_vals, self.odd_vals],
            [self.even_states_hash, self.odd_states_hash],
            [self.even_states, self.odd_states],
        ):
            reverse_hash = []
            for key, val in states_hash.items():
                reverse_hash.append(key)

            rows = pos_data[0]
            cols = pos_data[1]
            filt_zero = np.where(vals != 0)
            rows = rows[filt_zero]
            cols = cols[filt_zero]
            vals = vals[filt_zero]

            # Create a graph from the non-zero row and column data
            G = nx.Graph()
            G.add_edges_from(zip(list(rows), list(cols)))
            components = list(nx.connected_components(G))
            block_components.append(
                [np.array([reverse_hash[i] for i in list(block)]) for block in components]
            )

            # From the graph components, reorder the basis
            new_states_order = []
            for j in range(len(components)):
                new_states_order.extend(list(components[j]))
            states_block_basis.append([reverse_hash[i] for i in new_states_order])
            if graph:
                fig, axs = plt.subplots(
                    ncols=len(components), figsize=(3.5 * len(components), 2.5)
                )

                fig.suptitle(f"{title[i]} sectors")
                for idx, comp in enumerate(components):
                    if len(components) > 1:
                        ax = axs[idx]
                    else:
                        ax = axs
                    subgraph = G.subgraph(comp)
                    subgraph = subgraph.copy()  # Make a copy to avoid modifying original
                    subgraph.remove_edges_from(
                        nx.selfloop_edges(subgraph)
                    )  # Remove self-loops
                    pos = nx.spring_layout(
                        G
                    )  # Position nodes using a force-directed layout
                    labels = {
                        i: self.vis_state(reverse_hash[i])[1:-1] for i in list(subgraph)
                    }
                    nx.draw(
                        subgraph,
                        pos,
                        ax=ax,
                        with_labels=True,
                        labels=labels,
                        node_color=color[i],
                        edge_color="black",
                        node_size=500,
                        font_size=8,
                        font_family="DejaVu Sans",
                    )
            i += 1
        self.even_states = np.array(states_block_basis[0])
        self.even_states_hash = {num: idx for idx, num in enumerate(self.even_states)}

        self.odd_states = np.array(states_block_basis[1])
        self.odd_states_hash = {num: idx for idx, num in enumerate(self.odd_states)}

        self.fock_states = np.append(self.even_states, self.odd_states)
        self.gather_H()
        self.H_to_array("odd")
        self.H_to_array("even")

        if print_result:
            print(f"Obtained {len(block_components[0])} Even blocks")
            for idx, block_states in enumerate(block_components[0]):
                display(Markdown(f"{idx+1}: {self.vis_state_list(block_states)}"))

            print(f"Obtained {len(block_components[1])} Odd blocks")
            for idx, block_states in enumerate(block_components[1]):
                display(Markdown(f"{idx+1}: {self.vis_state_list(block_states)}"))

            fig, axs = plt.subplots(ncols=2, figsize=(6, 3))
            axs[0].matshow(np.abs(self.H_even))
            axs[0].set_title("Even sector")
            axs[1].matshow(np.abs(self.H_odd))
            axs[1].set_title("Odd sector")

        return block_components

    def H_to_array(self, parity):
        """
        Converts lists of non-zero row/column indices to a 2D array
        Used for smaller systems to do np.linalg.eigh on the Hamiltonian in array form
        Args:
            parity (str): which parity sector to construct the hamiltonian for
        Returns:
            None
        """
        if parity == "odd":
            states = self.odd_states
            pos_data = self.odd_terms
            val_data = self.odd_vals
        else:
            states = self.even_states
            pos_data = self.even_terms
            val_data = self.even_vals

        arr = np.zeros((len(states), len(states)), dtype=complex)
        idx = 0
        rows = np.append(pos_data[0], pos_data[1])
        cols = np.append(pos_data[1], pos_data[0])
        pars = np.append(val_data * pos_data[2], np.conj(val_data) * pos_data[2])

        np.add.at(arr, (rows, cols), pars)
        # np.add.at(arr, (cols,rows), np.conj(pars*val_data))

        if parity == "odd":
            self.H_odd = arr
        else:
            self.H_even = arr

    def solve_system(
        self,
        method="linalg",
        n_values=None,
    ):
        """
        Calculate the eigenvalues and eigenstates of the odd and even parity sectors
        Args:
            method (str):
                'linalg': uses full matrix arrays and np.linalg.eigh to calculate values
                'sparse': uses scipys LinearOperator and the lists of non-zero terms to calculate values
                The latter becomes faster for larger N, when only requiring a small number of n_values
            n_values (int):
                Required in combination with method='sparse'-> specifies the number n of smallest eigenvalues
                and eigenvectors to calculate.
                method='linalg' will always return the full set of solutions
        Returns:
            E_odd: list of eigenvalues of the odd parity sector
            E_even: list of eigenvalues of the even parity sector
            E_phi_odd: array of normalized odd eigenvector weights
            E_phi_even: array of normalized even eigenvector weights
        """

        if method == "linalg":
            ## Check if the matrices have been constructed, otherwise generate them
            if not hasattr(self, "H_odd"):
                self.H_to_array("odd")
            if not hasattr(self, "H_even"):
                self.H_to_array("even")
            E_odd, E_even, E_phi_odd, E_phi_even = self.solve_linalg()

        elif method == "sparse":
            ## If no n_values is provided, get the maximum allowed number
            if n_values is None:
                n_values = max([len(self.odd_states - 1), len(self.even_states - 1)])
            E_odd, E_even, E_phi_odd, E_phi_even = self.solve_sparse(n_values)

        return E_odd, E_even, E_phi_odd, E_phi_even

    def solve_linalg(self):
        """
        Uses numpy's linalg.eigh to obtain all eigenvalues and eigenvectors
        """
        E_odd, E_phi_odd = np.linalg.eigh(self.H_odd, UPLO="U")
        E_phi_odd = np.transpose(E_phi_odd)
        E_phi_odd = np.round(E_phi_odd, 10)  ## Truncate numerical errors

        E_even, E_phi_even = np.linalg.eigh(self.H_even, UPLO="U")
        E_phi_even = np.transpose(E_phi_even)
        E_phi_even = np.round(E_phi_even, 10)  ## Truncate numerical errors

        ## obtain the ground states of the system (allow degeneracy within some treshhold)
        return E_odd.real, E_even.real, E_phi_odd, E_phi_even

    def solve_sparse(self, n_values):
        """
        Uses scipy's LinearOperator to calculate the n_values number of lowest eigenvalues and eigenvectors
        """
        ## Construct LinearOperator for odd parity sector
        M_odd = LinearOperator(
            (len(self.odd_states), len(self.odd_states)),
            matvec=partial(
                self.sparse_function,
                rows=self.odd_terms[0],
                cols=self.odd_terms[1],
                vals=self.odd_vals * self.odd_terms[2],
            ),
            dtype=complex,
        )

        ## Construct LinearOperator for even parity sector
        M_even = LinearOperator(
            (len(self.even_states), len(self.even_states)),
            matvec=partial(
                self.sparse_function,
                rows=self.even_terms[0],
                cols=self.even_terms[1],
                vals=self.even_vals * self.even_terms[2],
            ),
            dtype=complex,
        )

        ## n_values cannot be larger than dimension if system -1
        n_values_odd = min([n_values, len(self.odd_states) - 1])
        n_values_even = min([n_values, len(self.even_states) - 1])

        ## Obtain n lowest eigenvalues and eigenvectors (up to n_values)
        E_odd, phi_odd = eigsh(M_odd, k=n_values_odd, which="SA")
        E_even, phi_even = eigsh(M_even, k=n_values_even, which="SA")
        return E_odd, E_even, np.transpose(phi_odd), np.transpose(phi_even)

    @staticmethod
    def sel_ground_states(E_odd: list, E_even: list, threshold: float = 0.001):
        """
        Determine the idx of the lowest ground state, between the odd and even sector
        Selects multiple in case of degeneracies, within the given threshhold
        Args:
            E_odd (list): odd eigenvalues
            E_even (list): even eigenvalues
            thresshold (float): the threshold within which to consider a degeneracy
        Returns:
            odd_gs (list): indexes of odd groundstates
            even_gs (list): indexes of even groundstates
            min_E (float): the lowest eigenvalue
        """
        min_e_odd = E_odd[np.argmin(E_odd)]
        min_e_even = E_even[np.argmin(E_even)]

        min_e = min(min_e_odd, min_e_even)
        even_gs = []
        odd_gs = []
        for idx, e in enumerate(E_odd):
            if np.abs(e - min_e) < threshold:
                odd_gs.append(idx)
        for idx, e in enumerate(E_even):
            if np.abs(e - min_e) < threshold:
                even_gs.append(idx)
        return odd_gs, even_gs, min_e

    def eigenstates(self, only_ground_states=False, only_energies=False):
        """
        Calculate and display the eigenstates of the system, sorted by energy

        """
        E_odd, E_even, E_phi_odd, E_phi_even = self.solve_system(method="linalg")
        odd_gs, even_gs, E_lowest = self.sel_ground_states(
            E_odd, E_even
        )  ## which states to label as ground state

        ## Merge the odd and even states
        merged_list = []
        i = 0
        for E, phi, gs in zip(
            [E_odd, E_even], [E_phi_odd, E_phi_even], [odd_gs, even_gs]
        ):
            for idx in np.arange(len(E)):
                if idx in gs:
                    merged_list.append((E[idx], phi[idx], "ground", i))
                else:
                    merged_list.append((E[idx], phi[idx], "excited", i))
            i += 1
        sorted_merged_list = sorted(merged_list, key=lambda x: x[0])

        ## Print the sorted states
        states_list = [self.odd_states, self.even_states]
        parity_list = ["odd", "even"]
        for state in sorted_merged_list:
            label = f"({state[2]}) ({parity_list[state[3]]})"
            if only_ground_states:
                if state[2] != "ground":
                    continue
                else:
                    if only_energies:
                        print(f"{label:<20} E = {np.round(state[0],2)}")
                    else:
                        print(
                            f"{label:<20} E = {np.round(state[0],2)}  \u007c\u03c6\u3009= {self.vis_state_list(states_list[state[3]],  state[1])}"
                        )
            else:
                if only_energies:
                    print(f"{label:<20} E = {np.round(state[0],2):<10.2f}")
                else:
                    print(
                        f"{label:<20} E = {np.round(state[0],2):<10.2f} \u007c\u03c6\u3009= {self.vis_state_list(states_list[state[3]],  state[1])}"
                    )

    def show_hamiltonian_numeric(self, parity, width):
        """
        Show the filled Hamiltonian matrix
        """
        if parity == "odd":
            if not hasattr(self, "H_odd"):
                self.H_to_array("odd")
            H = self.H_odd
        else:
            if not hasattr(self, "H_even"):
                self.H_to_array("even")
            H = self.H_even
        # H = self.H_to_array(parity)
        fig, ax = plt.subplots(1, figsize=(width / 1.2, width / 1.2))
        ax.set_xlim([0, width])
        ax.set_ylim([width, 0])
        for i in np.arange(width):
            for j in np.arange(width):
                t = H[j][i]
                t = np.round(complex(t), 2)
                if t.imag == 0:
                    t = t.real
                ax.text(i, j, t, horizontalalignment="center")
        ax.axis("off")

    def show_hamiltonian(self, parity, numeric=False):
        """
        Show the symbolic Hamiltonian matrix
        To do: add the symbols on the diagonal
        """
        if parity == "odd":
            states = self.odd_states
            pos_data = self.odd_terms
            val_data = self.odd_vals
        else:
            states = self.even_states
            pos_data = self.even_terms
            val_data = self.even_vals

        strs = ""
        for state in states:
            strs += self.vis_state(state) + ", "
        print(f"Basis: {strs[:-2]}")
        if numeric:
            self.show_hamiltonian_numeric(parity, len(states))
            return
        width = len(states)
        fig, ax = plt.subplots(1, figsize=(width / 1.2, width / 1.2))
        ax.set_xlim([0, width])
        ax.set_ylim([width, 0])

        ## Fill in known non-diagonal terms
        for idx, r in enumerate(pos_data[0]):
            i = r
            j = pos_data[1][idx]
            if i == j:
                continue
            sign = pos_data[2][idx]
            type = pos_data[3][idx]
            if sign == 1:
                t_print = self.H_symbols[f"{type}"]
            else:
                t_print = "-" + self.H_symbols[f"{type}"]
            ax.text(i, j, t_print, horizontalalignment="center")
            ax.text(j, i, t_print, horizontalalignment="center")

        ## list of row and column pairs
        rc_pairs = [[r, c] for r, c in zip(pos_data[0], pos_data[1]) if r != c]

        ## Fill in remaining zeroes
        for i in range(len(states)):
            for j in range(i, len(states)):
                if [i, j] in rc_pairs:
                    continue
                else:
                    ax.text(i, j, "0", horizontalalignment="center")
                    if i != j:
                        ax.text(j, i, "0", horizontalalignment="center")

        ax.axis("off")

    def lowest_transitions_sorted(
        self, sites: int, method: str = "linalg", n_values: int = 1, threshold: float = 1
    ):
        """
        Set-up for calculating possible single-electron transitions between odd/even ground states and the excited states
        for adding holes/electrons to a given site
        Included for speed compared to solving the rate equation

        Args:
            site: the fermionic site to calculate transitions for
            method: method to use for solving the hamiltonian
            n_values: number of lowest eigenvalues to obtain, only relevant if method='sparse'
            thresshold: range within which to consider a groundstate degenerate
        """
        E_odd, E_even, phi_odd, phi_even = self.solve_system(
            method=method, n_values=n_values
        )
        odd_gs_idx, even_gs_idx, min_E = self.sel_ground_states(
            E_odd, E_even, threshold=threshold
        )
        rows_to_keep = []
        for idx in even_gs_idx:
            rows_to_keep.append(idx)
        for idx in odd_gs_idx:
            rows_to_keep.append(idx + len(E_even))
        rows_to_keep = np.array(rows_to_keep)
        T_all = [[] for i in range(len(sites))]
        weights_all = [[] for i in range(len(sites))]
        ## Merge the odd and even sections into a block diagonal matrix with only the lowest eigenvectors and eigenvalues
        E = np.append(E_even, E_odd)
        phi = block_diag(phi_even, phi_odd)

        Es_a, Es_b = np.meshgrid(E, E)
        Es_ba = Es_b - Es_a
        ## For each desired site, get transition rate matrix
        for idx, site in enumerate(sites):
            operators = [
                [operator("creation", site, "up")],
                [operator("creation", site, "down")],
            ]  ## Create spin-up and spin-down
            Tsq_plus = np.abs(self.bra_oper_ket(self.fock_states, phi, operators)) ** 2
            Tsq_minus = Tsq_plus.T

            T_all[idx].extend(Es_ba[rows_to_keep].flatten())
            T_all[idx].extend(-Es_ba[rows_to_keep].flatten())
            weights_all[idx].extend(Tsq_plus[rows_to_keep].flatten())
            weights_all[idx].extend(Tsq_minus[rows_to_keep].flatten())

        filtered_T_all = [[] for i in range(len(sites))]
        filtered_weights = [[] for i in range(len(sites))]
        for idx in range(len(sites)):
            filter_zeros = np.where(np.array(weights_all[idx]) > 0)[0]

            filtered_T_all[idx] = np.array(T_all[idx])[filter_zeros]
            filtered_weights[idx] = np.array(weights_all[idx])[filter_zeros]

        return filtered_T_all, filtered_weights

    def rate_equation(
        self, sites, bias_range, lead_params, truncate_lim=100, method="linalg"
    ):
        ## Solve for energies and wavefunctions
        E_odd, E_even, phi_odd, phi_even = self.solve_system(
            method=method, n_values=truncate_lim
        )

        ## Merge the odd and even sections into a block diagonal matrix with only the lowest eigenvectors and eigenvalues
        E_even_trunc, E_odd_trunc = self.N_lowest_states(
            E_even, E_odd, truncate_lim=truncate_lim
        )
        E = np.append(E_even[:E_even_trunc], E_odd[:E_odd_trunc])
        phi = block_diag(phi_even[:E_even_trunc], phi_odd[:E_odd_trunc])

        G_matrix = np.zeros((len(sites), len(sites), len(bias_range)))

        Es_a, Es_b = np.meshgrid(E, E)
        Es_ba = Es_b - Es_a
        Es_ab = -Es_ba
        Tsq_plus_list, Tsq_minus_list = [], []
        ## For each desired site, get transition rate matrix
        for site in sites:
            operators = [
                [operator("creation", site, "down")],
                [operator("creation", site, "down")],
            ]  ## Create spin-up and spin-down

            Tsq_plus = np.abs(self.bra_oper_ket(self.fock_states, phi, operators)) ** 2
            Tsq_minus = Tsq_plus.T
            Tsq_plus_list.append(Tsq_plus)
            Tsq_minus_list.append(Tsq_minus)

        ## These terms needed for solving rate equation are constant
        kBT = lead_params["kBT"]
        Nf_0_plus = n_F(Es_ba, 0, kBT)
        Nf_0_min = np.ones(np.shape(Es_ba)) - n_F(Es_ab, 0, kBT)

        ## Generate currents from transition rate matrix
        for i, Vbias in enumerate(bias_range):
            ## These terms are specific to each Vbias
            mu_minus = Vbias - lead_params["dV"]
            Nf_mu_plus_minus = n_F(Es_ba, mu_minus, kBT)
            Nf_mu_min_minus = np.ones(np.shape(Es_ba)) - n_F(Es_ab, mu_minus, kBT)

            mu_plus = Vbias + lead_params["dV"]
            Nf_mu_plus_plus = n_F(Es_ba, mu_plus, kBT)
            Nf_mu_min_plus = np.ones(np.shape(Es_ba)) - n_F(Es_ab, mu_plus, kBT)

            for j in range(len(sites)):
                mus = np.zeros(len(sites))
                mus[j] = mu_minus
                Is0 = get_Is_vectorized(
                    len(sites),
                    Tsq_plus_list,
                    Tsq_minus_list,
                    lead_params["gammas"],
                    mus,
                    Nf_0_plus,
                    Nf_0_min,
                    Nf_mu_plus_minus,
                    Nf_mu_min_minus,
                )

                mus = np.zeros(len(sites))
                mus[j] = mu_plus
                Is1 = get_Is_vectorized(
                    len(sites),
                    Tsq_plus_list,
                    Tsq_minus_list,
                    lead_params["gammas"],
                    mus,
                    Nf_0_plus,
                    Nf_0_min,
                    Nf_mu_plus_plus,
                    Nf_mu_min_plus,
                )

                gs = 2 * np.pi * (Is1 - Is0) / (2 * lead_params["dV"])
                G_matrix[:, j, i] = gs
        return G_matrix

    @staticmethod
    def N_lowest_states(E_even, E_odd, truncate_lim):
        sorted_E = np.sort(np.append(E_odd, E_even))
        trunc = min([truncate_lim, len(E_odd) + len(E_even)])
        lowest_E_vals = (sorted_E[:trunc])[::-1]

        E_odd_trunc = (
            np.where(E_odd == lowest_E_vals[np.argmax(np.isin(lowest_E_vals, E_odd))])[0][
                -1
            ]
            + 1
        )
        E_even_trunc = (
            np.where(E_even == lowest_E_vals[np.argmax(np.isin(lowest_E_vals, E_even))])[
                0
            ][-1]
            + 1
        )
        return E_even_trunc, E_odd_trunc
