import numpy as np
from  .c import fermion_operations as fo
from functools import partial

## Class implementing matrix-vector multiplication storing only minimal data
class FockOperVerySparse():
    def __init__(self, sparse_fock_basis, oper_list, weights):
        self.oper_list = self.padded_oper_list(oper_list)
        self.values = np.array(weights,dtype=complex)
        self.fock_basis = sparse_fock_basis 
        
    @staticmethod
    def padded_oper_list(oper_list):
        max_len = 0
        for op in oper_list:
            if isinstance(op,list):
                if len(op)>max_len:
                    max_len = len(op)
        new_list = []
        for op in oper_list:
            if isinstance(op,list):
                new_list.append(op + [-1]*(max_len-len(op)))
        return np.array(new_list,dtype=np.int32)

    def get_sparse_func(self):
        return partial(fo.very_sparse_matvec, Hamiltonian = self.oper_list, vals = self.values)

    def __matmul__(self, vector):
        return self.get_sparse_func()(vector)

class FockStatesVerySparse():
    def __init__(self, N):
        self.N = N