import FermionSystem as fst
import numpy as np
from functools import partial
from tqdm import tqdm


def get_chain_terms(fs,dist,types, spins):
    all_terms = []
    for i in range(fs.N-dist):
        oper_0 = fs.operator(types[0], i, spins[0])
        oper_1 = fs.operator(types[1], i + dist, spins[1])
        all_terms.append(fs.normal_order([oper_0, oper_1])[0])
    return all_terms

def make_kitaev_hamiltonian(fs):
    kitaev_terms = []
    sign_terms = []
    spin_combis = [['up','up'],['up','down'],['down','up'],['down','down']]
    rel_signs = [1,-1,1,1]
    hopping = ['annihilation','creation']
    pairing = ['creation','creation']
    
    ## Get all next nearest-neighbour interactions
    for rel_sign,spin in zip(rel_signs,spin_combis):
        hop_terms = get_chain_terms(fs,1, hopping, spin)
        for t in hop_terms:
            kitaev_terms.append(t)
            sign_terms.append(rel_sign)
            
        pair_terms = get_chain_terms(fs,1, pairing,spin)
        for t in pair_terms:
            kitaev_terms.append(t)
            sign_terms.append(rel_sign)

    ## Chemical Potential
    for spin in ['up','down']:
        for i in range(fs.N):
            oper_0 = fs.operator('creation', i, spin)
            oper_1 = fs.operator('annihilation', i, spin)
            term = fs.normal_order([oper_0, oper_1])[0] 
            kitaev_terms.append(term)
            sign_terms.append(1)

    ## Charging Energy
    for i in range(fs.N):
        oper_0 = fs.operator('creation', i, 'up')
        oper_1 = fs.operator('creation', i, 'down')
        oper_2 = fs.operator('annihilation', i, 'up')
        oper_3 = fs.operator('annihilation', i, 'down')
        term = fs.normal_order([oper_0, oper_1,oper_2,oper_3])[0] 
        kitaev_terms.append(term)
        sign_terms.append(1)
    return kitaev_terms,sign_terms

def map_H_params_kitaev(fs,H_params):
    H_vals = {}
    H_symbols = {}
    H_types = {}
    spin_map = {'u':['up','\u2191'],'d':['down','\u2193']}

    for param in H_params.keys():
        if param[0] == 'd':
            type = ['creation','creation']
            spin_0 = spin_map[param[-2]][0]
            spin_1 = spin_map[param[-1]][0]
            base_symb = '\u0394'
            spin_symb = f'{spin_map[param[-2]][1]}{spin_map[param[-1]][1]}'
        elif param[0] == 't':
            type = ['annihilation','creation']
            spin_0 = spin_map[param[-2]][0]
            spin_1 = spin_map[param[-1]][0]
            base_symb = 't'
            spin_symb = f'{spin_map[param[-2]][1]}{spin_map[param[-1]][1]}'
        else:
            continue
        for idx,val in enumerate(H_params[param]):
            oper_0 = fs.operator(type[0], idx, spin_0)
            oper_1 = fs.operator(type[1], idx + 1, spin_1)
            oper_type = fs.oper_list_to_str(fs.normal_order([oper_0, oper_1])[0])
            H_vals[oper_type] = val
            H_symbols[oper_type] = '$' + f'{base_symb}_'+'{' + f'{spin_symb}' + '}^{' + f'{idx}' + '}' + '$'
            H_types[f'{param}_{idx}'] = oper_type
            
    for param in H_params.keys():
        if param[0] == 'm':
            for idx,val in enumerate(H_params[param]):
                oper_0 = fs.operator('creation',idx, 'up')
                oper_1 = fs.operator('annihilation',idx,'up')
                oper_type = fs.oper_list_to_str(fs.normal_order([oper_0, oper_1])[0])
                value = val + H_params['Ez'][idx]
                H_vals[oper_type] = value/2 # halfvalue due to duplicate in matvec
                H_types[f'{param}_u_{idx}'] = oper_type

                oper_0 = fs.operator('creation',idx,'down')
                oper_1 = fs.operator('annihilation',idx,'down')
                oper_type = fs.oper_list_to_str(fs.normal_order([oper_0,oper_1])[0])
                value = val - H_params['Ez'][idx]
                H_vals[oper_type] = value/2
                H_types[f'{param}_d_{idx}'] = oper_type

        if param[0] == 'U':
            for idx,val in enumerate(H_params[param]):
                oper_0 = fs.operator('creation',idx, 'up')
                oper_1 = fs.operator('annihilation',idx,'up')
                oper_2 = fs.operator('creation',idx, 'down')
                oper_3 = fs.operator('annihilation',idx, 'down')
                oper_type = fs.oper_list_to_str(fs.normal_order([oper_0, oper_1,oper_2,oper_3])[0])
                value = val/2
                H_vals[oper_type] = value
                H_types[f'U_{idx}'] = oper_type
    return H_vals,H_symbols,H_types

def phase_space(chain, vary_params_x, x_vals,  vary_params_y,y_vals, T=3):
    EoddEeven = []
    Eexp = []
    shape = (len(y_vals), len(x_vals))
    for y_val in tqdm(y_vals):
    
        chain.update_H_param_list(vary_params_y, y_val, update_matrix=False)

        for x_val in x_vals:
            chain.update_H_param_list(vary_params_x, x_val,update_matrix=False)

            Eodd,Eeven = chain.solve_system(n_values=1, method='sparse')[:2]
            Eexp.append(np.exp(-(np.abs(Eodd[0]-Eeven[0]))/T))
    
            EoddEeven.append(Eodd[0]-Eeven[0])
    EoddEeven = np.reshape(EoddEeven, shape)
    Eexp = np.reshape(Eexp,shape)
    return EoddEeven,Eexp

def energy_spectrum(chain, params, param_range, site, fig, ax, plot=True):
	all_energies,all_weights,all_xvars = [],[],[]
	for v_idx in tqdm(np.arange(len(param_range))):
		chain.update_H_param_list(params, param_range[v_idx], update_matrix=True) 

		energies,weights = chain.transitions_sorted(site)
		all_energies.extend(energies)
		all_weights.extend(weights)
		all_xvars.extend(np.full(len(weights), param_range[v_idx]))
	if plot:
		plot_energy_spectrum(fig, ax, all_xvars, all_energies, all_weights,param_range[v_idx],site)
	else:
		return all_xvars,all_energies,all_weights

       
def plot_energy_spectrum(fix, ax,mu, energies,weights, xval,site):
    ax.scatter(mu,energies, alpha=  np.abs(weights) , s=3, color = 'black')
    ax.format(ylabel = '$E_{T}$')
    ax.format(title = f'Spectum site {site}', fontsize = 7)

def make_kitaev_chain(N,H_params, Ez_inf = True, U_inf = True, make_arrays=False, sparse_function = None):
    fs = fst.FermionSystem(N)
    generate_kit = partial(make_kitaev_hamiltonian,fs)
    generate_map = partial(map_H_params_kitaev,fs,H_params)
    chain = fst.ParitySystem(N = N, H_generator=generate_kit,H_mapping = generate_map,sparse_function=sparse_function, Ez_inf = Ez_inf, U_inf=U_inf)
    if make_arrays:
        chain.H_to_array('odd')
        chain.H_to_array('even')
    return chain