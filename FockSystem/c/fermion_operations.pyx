
import numpy as np
cimport numpy as cnp
from cython cimport boundscheck, wraparound

def merge_terms_cython(list weights, list oper_list):
    cdef dict merged_dict = {}
    cdef list merged_weight = []
    cdef list merged_list = []
    cdef int idx = 0
    cdef seq
    cdef weight

    # Loop through the oper_list and weights
    for idx in range(len(weights)):
        seq = oper_list[idx]
        weight = weights[idx]
        # Convert lists to tuples (if seq is a list) for hashability
        if isinstance(seq, list):
            seq = tuple(seq)

        if seq not in merged_dict:
            merged_dict[seq] = weight            
        else:
            merged_dict[seq] += weight

    # Now build the merged list and weights based on merged_dict
    for key in merged_dict.keys():
        # Convert tuple back to list if it was a tuple before
        if isinstance(key, tuple):
            merged_list.append(list(key))
        else:
            merged_list.append(key)
        merged_weight.append(merged_dict[key])

    return merged_weight, merged_list


cdef int hamming_weight(unsigned int x) nogil:
    cdef int count = 0
    while x:
        count += x & 1
        x >>= 1
    return count

cpdef tuple act_oper(int oper, int state):
    """Optimized version of act_oper using Cython"""
    cdef int check_bit = oper & 0b1
    cdef int flip_bit_pos = oper >> 1
    cdef int flip_bit = 1 << flip_bit_pos

    # Check if the operation annihilates the state, early exit otherwise
    cdef bint destroyed = ((flip_bit & state) == 0) == check_bit
    if destroyed:
        return -1, 1

    # Compute relative signs
    cdef int parity_bits = state & (flip_bit - 1)
    cdef int signs = (hamming_weight(state & parity_bits) & 0b1) * -2 + 1

    # Flip bit
    cdef int new_state = state ^ flip_bit

    return new_state, signs


cpdef list apply_act_oper_to_all(int oper, int N):
    """
    Applies act_oper to all integers from 0 to N and stores results in a list.
    """
    cdef int i
    cdef tuple result
    cdef list results = []
    for i in range(N + 1):  # Loop over all numbers from 0 to N
        result = act_oper(oper, i)
        #results.append(result)  # Store results in a Python list

    return results


def sparse_matvec(
    cnp.ndarray[double complex, ndim=1] x,
    cnp.ndarray[int, ndim=1] rows,
    cnp.ndarray[int, ndim=1] cols,
    cnp.ndarray[double complex, ndim=1] vals,
):
    cdef int r, c, i
    cdef double complex v
    cdef cnp.ndarray[double complex, ndim=1] result = np.zeros_like(x)
    cdef int n = rows.shape[0]

    for i in range(n):
        r = rows[i]
        c = cols[i]
        v = vals[i]
        result[r] = result[r] + v * x[c]
    return result

cpdef cnp.ndarray very_sparse_matvec(
    cnp.ndarray[double complex, ndim=1] x,
    cnp.ndarray[int, ndim=2] Hamiltonian,
    cnp.ndarray[double complex,ndim=1] vals,
    ):
    cdef int N = x.shape[0]
    cdef cnp.ndarray[double complex,ndim=1] result = np.zeros(N, dtype=complex)
    cdef int n, h, o,sign, new_sign, new_state,operator
    cdef int H = Hamiltonian.shape[0]  # Number of rows
    cdef int O = Hamiltonian.shape[1]  # Number of columns
    for n in range(N):
        for h in range(H):
            sign = 1 ## reset the sign
            new_state = n ## set the new_state to the counter
            v = vals[h]
            for o in range(O):
                operator  = Hamiltonian[h,o] ## Grab the operator
                if operator != -1 and new_state!=-1:
                    new_state,new_sign = act_oper(operator, new_state)
                    sign *= new_sign
                else:
                    break

            if new_state != -1:
                result[n] = result[n] + vals[h] * x[new_state] * sign
                if (n!=new_state):
                    result[new_state] = result[new_state] + vals[h] * x[n] * sign   
    return result

def multiplication_basis(
    list weights_1,  # List of complex floats
    list oper_list_1,  # Nested list of integers
    list weights_2,  # List of complex floats
    list oper_list_2  # Nested list of integers
):
    cdef list oper_products = []  # List of lists of integers
    cdef list product_weights = []  # List of complex floats
    cdef int idx_1, idx_2

    cdef oper_seq_1, oper_seq_2
    cdef list temp

    for idx_1, oper_seq_1 in enumerate(oper_list_1):
        for idx_2, oper_seq_2 in enumerate(oper_list_2):
            if isinstance(oper_seq_1, list) and isinstance(oper_seq_2, list):
                temp = oper_seq_2.copy()
                temp.extend(oper_seq_1)
                oper_products.append(temp)
                product_weights.append(weights_1[idx_1] * weights_2[idx_2])
            elif isinstance(oper_seq_1, list):
                oper_products.append(oper_seq_1.copy())
                product_weights.append(weights_1[idx_1] * oper_seq_2 * weights_2[idx_2])
            elif isinstance(oper_seq_2, list):
                oper_products.append(oper_seq_2.copy())
                product_weights.append(weights_2[idx_2] * oper_seq_1 * weights_1[idx_1])
            else:
                oper_products.append(oper_seq_1 * oper_seq_2)
                product_weights.append(weights_2[idx_2] * weights_1[idx_1])

    return product_weights, oper_products



cpdef bint is_normal_ordered(list oper_list):
    """Check if oper_list is in normal order."""
    cdef int idx, i, n
    cdef list oper_seq

    for idx in range(len(oper_list)):  # Loop using index
        if isinstance(oper_list[idx], list):
            oper_seq = oper_list[idx]
            n = len(oper_seq)
            for i in range(n - 1):
                if (oper_seq[i] % 2) < (oper_seq[i + 1] % 2):
                    return False
                elif (oper_seq[i] < oper_seq[i + 1]) and (
                    (oper_seq[i] % 2) < (oper_seq[i + 1] % 2)
                ):
                    return False
    return True

cpdef tuple normal_order(list weights, list oper_list):
    cdef int seq_idx, i, j, n
    cdef bint flag_swap, is_normal_ordered
    cdef list oper_seq
    is_normal_ordered = False
    while not is_normal_ordered:
        seq_idx = 0
        is_normal_ordered = True
        while seq_idx < len(oper_list):
            if not isinstance(oper_list[seq_idx], list): 
                seq_idx += 1
                continue
            oper_seq = oper_list[seq_idx]

            n = len(oper_seq)
            for i in range(n - 1):
                flag_swap = False
                flag_commutator = False
                for j in range(n - 1 - i):
                    if (oper_seq[j] % 2) > (oper_seq[j + 1] % 2):
                        continue
                    if oper_seq[j] < oper_seq[j + 1] or (oper_seq[j] % 2) < (oper_seq[j + 1] % 2):
                        is_normal_ordered=False
                        if (oper_seq[j] ^ 0b1) == oper_seq[j + 1]:
                            oper_list.append(oper_seq[:])  # Clone current sequence
                            weights.append(weights[seq_idx] * -1)

                            oper_list[seq_idx] = oper_seq[:j] + oper_seq[j + 2:]
                            oper_list[-1][j], oper_list[-1][j + 1] = oper_list[-1][j + 1], oper_list[-1][j]
                            flag_swap = False
                            if len(oper_list[seq_idx]) == 0:
                                weights.append(weights[seq_idx])
                                weights.pop(seq_idx)
                                oper_list.pop(seq_idx)
                                oper_list.append(1)
                            break
                        else:
                            flag_swap = True
                            oper_seq[j], oper_seq[j + 1] = oper_seq[j + 1], oper_seq[j]
                            weights[seq_idx] *= -1

                if not flag_swap:
                    break

            seq_idx += 1

    return weights, oper_list  # Return updated values

cpdef tuple remove_duplicates(list weights, list oper_list):
    """Remove duplicate fermion sequences from oper_list and adjust weights."""
    cdef int idx
    cdef list oper_seq

    # Iterate backwards to avoid indexing issues while deleting
    for idx in range(len(weights) - 1, -1, -1):
        if not isinstance(oper_list[idx], list):
            continue
        else: 
            oper_seq = oper_list[idx]
            if len(oper_seq) != len(set(oper_seq)):  # Check for duplicates
                oper_list.pop(idx)
                weights.pop(idx)

    return weights, oper_list  # Return updated lists

@boundscheck(False)
@wraparound(False)
cdef int fill_memory(double[:] input_vec, double[:] output_vec,
                long start_i, long start_j, long small_rep, long big_rep, long big_rep_spacing, double val) nogil:
    cdef int i, j
    cdef long i_val, j_val
    cdef double tmp

    for i in range(big_rep):
        for j in range(small_rep):
            i_val = start_i + i * big_rep_spacing + j
            j_val = start_j + i * big_rep_spacing + j
            output_vec[i_val] += input_vec[j_val] * val
            output_vec[j_val] += input_vec[i_val] * val
    return 0

# Parallel loop function in Cython using prange (OpenMP)
@boundscheck(False)
@wraparound(False)
cpdef matvec_fast(cnp.ndarray[double, ndim=1] input_vec, long long[:,:] H_base_data, cnp.ndarray[double, ndim=1] H_base_vals):
    cdef int counter=0
    cdef int n_tuples = H_base_data.shape[0]
    cdef cnp.ndarray[double,ndim=1] output_vec = np.zeros_like(input_vec,order='C')
    cdef long base_i, base_j, small_rep, big_rep, big_rep_spacing
    cdef double base_value = 0
    cdef int n_threads = 0

    for counter in range(n_tuples):
        base_i = H_base_data[counter][0]
        base_j = H_base_data[counter][1]
        small_rep = H_base_data[counter][2]
        big_rep = H_base_data[counter][3]
        big_rep_spacing = H_base_data[counter][4]
        base_value = H_base_vals[counter]
        if base_value != 0:
            fill_memory(input_vec,output_vec, base_i,base_j, small_rep,big_rep, big_rep_spacing,base_value)

    return output_vec

