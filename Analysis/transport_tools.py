import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import block_diag
from FockSystem.FockSystemBase import operator_verbose


from tqdm import tqdm
from pathlib import Path
import json
import xarray as xr


class DummyVariable():
    def __init__(self,label,fancy_label):
        self.label = label
        self.fancy_label = fancy_label


def n_F(E, mu, kBT):
    energy = (E - mu) / kBT
    n = (np.exp(energy) + 1) ** -1
    return n


def get_P_old(rate_total, N_state):
    rate_matrix = np.zeros((N_state + 1, N_state))
    rate_matrix[0:N_state, 0:N_state] = rate_total
    for k in range(N_state):
        rate_matrix[k, k] = -np.sum(rate_total[:, k])
    rate_matrix[N_state, :] = np.ones(N_state)

    right_vec = np.zeros((N_state + 1, 1))
    right_vec[N_state, 0] = 1
    P_vec = np.linalg.pinv(rate_matrix) @ right_vec
    return P_vec


## Minor speedup compared to above code
def get_P_vectorized(rate_total):
    N_states = np.shape(rate_total)[0]

    rate_matrix = np.zeros((N_states + 1, N_states))
    rate_matrix[0:N_states, 0:N_states] = rate_total
    np.fill_diagonal(rate_matrix, -np.sum(rate_total, axis=0))
    rate_matrix[N_states, :] = np.ones(N_states)

    right_vec = np.zeros((N_states + 1, 1))
    right_vec[N_states, 0] = 1
    P_vec = np.linalg.pinv(rate_matrix) @ right_vec
    return P_vec


def get_current(rate_plus_list, rate_minus_list, P_vec, num_of_leads):
    Is = np.zeros(num_of_leads)
    for j in range(num_of_leads):
        Is[j] = np.sum((rate_plus_list[j] - rate_minus_list[j]) @ P_vec)
    return Is


def get_Is(num_of_leads, Tsq_plus_list, Tsq_minus_list, gammas, mus, Es_ba, kBT):
    rate_plus_list = []
    rate_minus_list = []
    rate_total = 0
    for k in range(num_of_leads):
        rate_plus = gammas[k] * Tsq_plus_list[k] * n_F(Es_ba, mus[k], kBT)
        rate_minus = (
            gammas[k]
            * Tsq_minus_list[k]
            * (np.ones(np.shape(Es_ba)) - n_F(-Es_ba, mus[k], kBT))
        )
        rate_plus_list.append(rate_plus)
        rate_minus_list.append(rate_minus)
        rate_total += rate_plus + rate_minus

    #P_vec = get_P_old(rate_total=rate_total, N_state=np.shape(rate_total)[0])
    P_vec = get_P(rate_total=rate_total, N_state = np.shape(rate_total)[0])

    Is = get_current(rate_plus_list, rate_minus_list, P_vec, num_of_leads)
    return Is


### n_F(mu=0) is calculated a redundant amount of times above
### Below gives a (minimal) speed-up when number of leads gets larger
### By calculated the Nf terms only once and passing them to get_Is
def get_Is_vectorized(
    num_of_leads,
    Tsq_plus_list,
    Tsq_minus_list,
    gammas,
    mus,
    Nf_0_plus,
    Nf_0_min,
    Nf_mu_plus,
    Nf_mu_min,
):
    nF_plus_array = np.array(
        [
            Nf_mu_plus * gammas[i] if mus[i] != 0 else Nf_0_plus * gammas[i]
            for i in range(len(mus))
        ]
    )
    nF_min_array = np.array(
        [
            Nf_mu_min * gammas[i] if mus[i] != 0 else Nf_0_min * gammas[i]
            for i in range(len(mus))
        ]
    )

    rate_plus_list = Tsq_plus_list * nF_plus_array
    rate_minus_list = Tsq_minus_list * nF_min_array

    rate_total = np.sum(rate_plus_list, axis=0) + np.sum(rate_minus_list, axis=0)

    P_vec = get_P_vectorized(rate_total=rate_total)

    Is = get_current(rate_plus_list, rate_minus_list, P_vec, num_of_leads)
    return Is


########################################################################################
def grab_symbolic_variables(subseq):
    json_path = Path(__file__).parent.parent / "FockSystem\\operators_symbolic.json"
    with json_path.open("r", encoding="utf-8") as file:
        symbol_dict = json.load(file)
    symbol_string = ''
    for sequence in subseq.oper_list:
        if isinstance(sequence,list):
            symbol_string += symbol_dict.get(subseq.oper_list_to_str(sequence), '')
    return symbol_string

def wrap_in_xarray(coords, datasets):
    xarray_coord = {}

    # Process coordinates
    for idx, coord_pair in coords.items():
        coord_values, coord_data = coord_pair  # Unpacking for clarity
        
        # Ensure we correctly extract 'label'
        data_label = coord_values.label if hasattr(coord_values, 'label') else f'{idx}'
        if hasattr(coord_values,'fancy_label'):
            data_string = coord_values.fancy_label
        else:
            data_string = grab_symbolic_variables(coord_values)  # Function assumed to exist
        
        # Store in xarray_coord
        xarray_coord[data_label] = xr.DataArray(
            coord_data, coords={data_label:coord_data,}, dims=(data_label,), attrs={"long_name": data_string, "units": "-"}
        )

    xarray_datasets = {}
    # Process datasets
    for idx, data in datasets.items():
        var_name, long_name, data_values, coordinates = data  # Unpacking assuming (name, label, values, dims)
        # Ensure dims exist in xarray_coord
        valid_dims = [f'{dim}' for dim in coordinates if f'{dim}' in xarray_coord.keys() ]
        valid_dims.reverse()
        coords = {dim: xarray_coord[dim] for dim in valid_dims}

        xarray_datasets[var_name] = xr.DataArray(
            data_values,
            coords = coords,
            dims=valid_dims,  # Use extracted valid dims
            attrs={"long_name": long_name, "units": "-"},
        )

    # Create and return xarray dataset
    ds = xr.Dataset(
        data_vars=xarray_datasets,
        coords=xarray_coord,
    )
    return ds
def phase_diagram(H, odd_basis, even_basis, subseq_x, x_range, subseq_y, y_range, disable_timer=False):
    H & odd_basis
    H & even_basis

    result_data = []
    for y_value in tqdm(y_range, disable=disable_timer):
        H[subseq_y] = y_value
        for x_value in x_range:
            H[subseq_x] = x_value
            E_odd,phi_odd = np.linalg.eigh(H[odd_basis].to_array(), UPLO="U")
            E_even,phi_even = np.linalg.eigh(H[even_basis].to_array(),  UPLO="U")
            result_data.append(E_odd[0]-E_even[0])

    dataset = wrap_in_xarray({0:[subseq_x, x_range],1:[subseq_y, y_range],},
    {0: ['E','$E_{odd} - E_{even}$', np.reshape(result_data, (len(y_range),len(x_range))), [0,1]]}
    )
    return dataset

def lowest_transitions_sorted(
        H, basis: int,sites, lowest_n_values= 1,
    ):
        """
        Set-up for calculating possible single-electron transitions between odd/even ground states and the excited states
        for adding holes/electrons to a given site
        Included for speed compared to solving the rate equation
        """
        E,phi = np.linalg.eigh(H[basis].data_array,UPLO='U' )
        phi = np.transpose(phi)
        rows_to_keep = [i for i in range(lowest_n_values)]
        T_all = [[] for i in range(len(sites))]
        weights_all = [[] for i in range(len(sites))]
        Es_a, Es_b = np.meshgrid(E, E)
        Es_ba = Es_b - Es_a
        ## For each desired site, get transition rate matrix
        for idx, site in enumerate(sites):
            operators = [
                [operator_verbose("creation", site, "up")],
                [operator_verbose("creation", site, "down")],
            ]  ## Create spin-up and spin-down
            Tsq_plus = np.abs(basis.bra_oper_ket(basis.states, phi, operators)) ** 2
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

def energy_spectrum(H,basis, param_seq, param_range, sites, fig, axs, plot=True):
    all_energies, all_weights, all_xvars = (
        [[] for i in range(len(sites))],
        [[] for i in range(len(sites))],
        [[] for i in range(len(sites))],
    )
    for v_idx in tqdm(np.arange(len(param_range))):
        H[param_seq] = param_range[v_idx]

        energies, weights = lowest_transitions_sorted(H,basis,sites, lowest_n_values=1)
        for i in range(len(sites)):
            all_energies[i].extend(energies[i])
            all_weights[i].extend(weights[i])
            all_xvars[i].extend(np.full(len(weights[i]), param_range[v_idx]))

    if plot:
        for ax in axs:
            ax.set_xlabel(grab_symbolic_variables(param_seq))
        for i in range(len(sites)):
            plot_energy_spectrum(
                fig,
                axs[i],
                all_xvars[i],
                all_energies[i],
                np.array(all_weights[i]),
                param_range[v_idx],
                sites[i],
            )
    else:
        return all_xvars, all_energies, all_weights


def plot_energy_spectrum(fix, ax, mu, energies, weights, xval, site):
    weights = np.abs(weights)
    weights = np.minimum(weights, 1)
    ax.scatter(mu, energies, alpha=np.abs(weights), s=3, color="black")
    ax.set_title(f"Spectum site {site}")
    ax.set_ylabel(f"E")


def rate_equation(
        H,basis, sites, bias_range, lead_params, truncate_lim=100,
    ):
        ## Solve for energies and wavefunctions
        E,phi = np.linalg.eigh(H[basis].to_array(), UPLO='U')
        phi = np.transpose(phi)
        G_matrix = np.zeros((len(sites), len(sites), len(bias_range)))

        Es_a, Es_b = np.meshgrid(E, E)
        Es_ba = Es_b - Es_a
        Es_ab = -Es_ba
        Tsq_plus_list, Tsq_minus_list = [], []
        ## For each desired site, get transition rate matrix
        for site in sites:
            operators = [
                [operator_verbose("creation", site, "down")],
                [operator_verbose("creation", site, "down")],
            ]  ## Create spin-up and spin-down

            Tsq_plus = np.abs(H.bra_oper_ket(basis.states, phi, operators)) ** 2
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

def conductance_spectrum(
    H,
    basis,
    subsequence,
    param_range,
    bias_range,
    sites=[0, 1],
    lead_params={},
    n_values = 100
):
    n_sites = len(sites)
    Gs = [[[] for i in range(n_sites)] for j in range(n_sites)]
    for v_idx in tqdm(np.arange(len(param_range))):
        H[subsequence] = param_range[v_idx]

        G_matrix = rate_equation(
            H,basis, sites, bias_range, lead_params, truncate_lim=n_values
        )

        for i in range(n_sites):
            for j in range(n_sites):
                Gs[i][j].append(G_matrix[i][j])

    coord_data = {
        f'bias_{s}':[DummyVariable(f'bias_{s}','$V_'+f'{s}'+'$'),bias_range] for s in sites
    }
    coord_data['param'] = [subsequence, param_range]

    datasets_data={}
    for i in range(n_sites):
        for j in range(n_sites):
            datasets_data[f'{i}{j}'] = ["G_" + f"{sites[i]}{sites[j]}","G_" + f"{sites[i]}{sites[j]}",np.reshape(Gs[i][j], (len(param_range), len(bias_range))), [f"bias_{sites[j]}",'param',]]

    dataset = wrap_in_xarray(coord_data,datasets_data )
    return dataset.transpose()

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