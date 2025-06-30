from  .c import fermion_operations as fo
from functools import partial

## Class implementing matrix-vector multiplication storing only minimal data
class OperSequenceDataSparse():
    def __init__(self, sparse_fock_basis, sparse_data):
        self.fock_basis = sparse_fock_basis 
        self.values = sparse_data[0]
        self.rc_indices = sparse_data[1]
        self.type_strings = sparse_data[2] 
        
    def get_sparse_func(self):
        return partial(fo.matvec_fast, H_base_data =  self.rc_indices, H_base_vals = self.values)

    def __matmul__(self, vector):
        if len(vector) != 4**self.fock_basis.N:
            raise ValueError("Vector length does not match the Fock basis size.")
        return self.get_sparse_func()(vector)
    
class FockStatesSparse():
    def __init__(self, N):
        self.N = N