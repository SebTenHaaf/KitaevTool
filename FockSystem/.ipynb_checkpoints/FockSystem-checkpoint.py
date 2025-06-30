
import numpy as np
import sys,os,json
from copy import deepcopy
from functools import partial
from scipy.sparse import coo_array
from collections import Counter

from IPython.core.display import Markdown
from IPython.display import display

from .FockSystemBase import FockSystemBase, hamming_weight, operator_verbose,operator_from_string
from .FockSystemSparse import FockOperVerySparse,FockStatesVerySparse
from .c import fermion_operations as fo


from typing import Callable, List, Any, Self


class FockStatesVerySparse(FockSystemBase):
    def __init__(self):
        return

class FockStates(FockSystemBase):
    def __init__(self, states, weights=None, N: int=None):
        self.states = self._parse_states(states)
        self.hashed =  {num: idx for idx, num in enumerate(self.states)}
        self.weights = (
            weights if weights is not None else [1 for i in range(len(self.states))]
        )

        ## For convenience, store the total number of Fermionic sites N
        if N is None:
            N = int(np.ceil(np.log2(np.max(self.states)+1)/2))
        self.N = N

    @staticmethod
    def _parse_states(states: [int,List,np.ndarray]):
        if isinstance(states,int):
            return np.arange(4**states)
        if isinstance(states,list):
            return np.array(states)
        if isinstance(states, np.ndarray):
            return states

    def _repr_markdown_(self) -> str:
        info_string = f'Fock basis with {len(self.states)} states<br>'
        info_string = f' '

        if len(self.states) > 32:
            return info_string + self.vis_state_list(self.states[:5], self.weights[:5], N=self.N)[1:] + "$\\cdots$ " + self.vis_state_list(self.states[-5:], self.weights[-5:], N=self.N) 
        return info_string + self.vis_state_list(self.states, self.weights, N=self.N)[0:]

    def __lt__(self, operator):
        old_states, new_states, new_parities = self.act_oper_list(
            operator.oper_list[0], self.states
        )
        result = FockState(new_states, N=self.N)
        return result

    def restrict(self, parity: str = None, Ez_inf: bool = False, U_inf: bool = False) -> "FockStates":
        """
        Sorts Fock states into 'even' and 'odd' parities (determined by count of 1's in binary)
        Optionally allows restricting the Fock space to infinite U or infinite Ez
        Args:
            Ez_inf (bool): If True, excludes states with a 'spin up' set
            U_inf (bool): If True, excludes states with both 'spin up' and 'spin down' set for a single site
            parity: 
        """
        all_states = self.states

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
        if parity=='even':
            return FockStates(even_states,N=self.N)
        elif parity=='odd':
            return FockStates(odd_states,N=self.N)
        else:
            return FockStates(all_states,N=self.N)

class OperSequenceData():
    """
    Stores data representing the action of an OperSequence instance on a specific FockStates instance.
    Maintains the minimal data in sparse format (rows, columns, values), with additional
    information that allows for quickly updating values when weights of the OperSequence are adjusted.

    Implements conversion to numpy arrays (.to_array()) or scipy coo_matrix format (.to_sparse_coo())

    Args:
        fock_basis (FockStates): the FockStates instance for which the data was generated.
        rows (np.ndarray[int]): row indices of non-zero elements
        cols (np.ndarray[int]): column indices of non-zero elements
        values (np.ndarray[complex]): values of non-zero elements
        parities (np.ndarray[int]): the sign under action of operators giving rise to a non-zero element
        type-string (np.ndarray[str]): string encoding of operators giving rise to a non-zero element
        data_array (Optional np.ndarray[complex]): full numpy array generated from rows,cols and values

    """
    def __init__(self, sparse_data: np.ndarray, fock_basis: FockStates):
        self.fock_basis = fock_basis 
        self.rows = sparse_data[0]
        self.cols = sparse_data[1]
        self.parities = sparse_data[2]
        self.type_strings = sparse_data[3]
        self.values = sparse_data[4]

    def _repr_markdown_(self) -> str:
        print(f'Rows: ', self.rows)
        print(f'Cols: ', self.cols)
        print(f'Vals: ', self.values)
        return ""

    def get_sparse_func(self) -> Callable:
        """
        Return a function that can be used by scipy's LinearOperator class
        for finding the lowest eigenvalues without storing a full matrix.

        Returns:
            sparse_func -> a function such that passing a vector x to sparse_func(x)
                            implements the matrix vector product to return a vector v
                            This is equivalent to getting the matrix form M with .to_array()
                            and applying M @ x = v 
        """

        filter_diag = np.where(self.rows!=self.cols)
        values = self.values*self.parities
        sparse_func = partial(fo.sparse_matvec, 
                            rows=np.append(self.rows, self.cols[filter_diag]),
                            cols=np.append(self.cols, self.rows[filter_diag]),
                            vals=np.append(values, np.conj(values[filter_diag])))
        return sparse_func

    def __matmul__(self, vector: np.ndarray) -> np.ndarray:
        return self.get_sparse_func()(vector)

    def update_values(self, type_keys: list[str], old_val: complex, new_val: complex) -> None:
        ## Set the new value for the relevant positions
        type_match = np.isin(self.type_strings, type_keys)
        signs = self.parities[type_match]
        new_values = signs*new_val
        self.values[type_match] = new_values

        ## If it has been created, update the matrix
        if hasattr(self, 'data_array'):
            rows = self.rows[type_match]
            cols = self.cols[type_match]
            old_values = signs*old_val
            filter_diagonal = np.where(rows!=cols) ## Prevent double counting of the diagonal terms
            np.add.at(
                self.data_array,
                (np.append(rows, cols[filter_diagonal]), np.append(cols, rows[filter_diagonal])),
                np.append(
                    -old_values + new_values,
                    -np.conj(old_values[filter_diagonal]) + np.conj(new_values[filter_diagonal]),
                ),
            )

    def to_array(self) -> np.ndarray:
        """
        Returns:
            
        """
        if hasattr(self, 'data_array'):
            return self.data_array
        array_size = len(self.fock_basis.states)
    
        arr = np.zeros((array_size, array_size), dtype=complex)
        idx = 0
        filter_diagonal = np.where(self.rows!=self.cols)
        rows = np.append(self.rows, self.cols[filter_diagonal])
        cols = np.append(self.cols, self.rows[filter_diagonal])
        new_values = self.values*self.parities
        vals = np.append(new_values, np.conj(new_values[filter_diagonal]))

        np.add.at(arr, (rows, cols), vals)
        self.data_array = arr
        return arr

    def to_sparse_coo(self) -> coo_array:
        N = len(self.fock_basis.states)
        filter_diagonal = np.where(self.rows!=self.cols)
        rows = np.append(self.rows, self.cols[filter_diagonal])
        cols = np.append(self.cols, self.rows[filter_diagonal])
        vals = np.append(self.values * self.parities, np.conj(self.values[filter_diagonal]) * self.parities[filter_diagonal])
        return coo_array((vals, (rows, cols)), shape=(N, N))

    def connected_components(self)-> List[List[int]]:
        idx =[[0]]
        components = [[self.rows[0],self.cols[0]]]
        for r,c in zip(self.rows[1:],self.cols[1:]):
            present_in = []
            for idx,comp in enumerate(components):
                r_present = (r in comp)
                c_present = (c in comp)
                if r_present and c_present:
                    present_in.append(idx)
                elif r_present:
                    components[idx].extend([c])
                    present_in.append(idx)
                elif c_present:
                    components[idx].extend([r])
                    present_in.append(idx)
            if len(present_in) == 0:
                components.append([r,c])
            elif len(present_in) >1:
                for p_idx in present_in[-1:0:-1]:
                    components[present_in[0]].extend(components[p_idx])
                    components.pop(p_idx)

        connected_components = [list(set(comp)) for comp in components]
        return connected_components


    def to_block_diagonal_basis(self) -> FockStates:
        components = self.connected_components()
        reverse_hash = []
        for key in self.fock_basis.hashed.keys():
            reverse_hash.append(key)

        new_states_order = []
        for j in range(len(components)):
            new_states_order.extend(list(components[j]))
        states_block_basis= [reverse_hash[i] for i in new_states_order]
        return FockStates(states_block_basis)


def get_file_path(filename:str)->str:
	""" 
		Retrieve a file path relative to package location, with correct os format
		Ensures parameters and configs are saved in correct location

		Args:
			filename (string): the name of the file whose path to obtain
		Returns:
			String containing the path to the file
	"""
	module_dir = os.path.dirname(os.path.abspath(__file__))
	return os.path.join(module_dir, filename)

filename_params = "operators_symbolic.json"

def load_dictionary(filename:str = filename_params)->dict:
    """
        Reads the data from an existing json file, 
        or creates a file with default settings

        Args:
            filename (string): name of the file to load, including extension
        Returns:
            dictionary: content of the file if it existed, default values otherwise
    """
    file_path = get_file_path(filename)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        with open(file_path, 'w') as f:
            default = {}
            json.dump(default, f)
            return default

_symbol_dict = load_dictionary(filename_params)

def save_dictionary(dictionary:dict, file_name:str = filename_params)->None:
    """
        Save a dictionary to a json file
        Args:
            dictionary: the dict with values to save
            file_path: str containing path to json file to store data
    """
    file_path = get_file_path(file_name)
    with open(file_path, 'w') as f:
        json.dump(dictionary, f)

class OperSequence(FockSystemBase):
    """
    Represents combinations of Fermionic operators and handles common arithmetic operations.
    Provides an interface to bind a Fermionic sequence to FockStates objects, to retrieve
    matrix or sparse representations of the sequence in a given basis. 
    Args:
        oper_list (list[list[int]]): a nested list storing the integer representations of fermionic operators in a sequence
        weights (list[complex]): a list of values storing the weight of each subsequence of operators
        basis_dict (dict) (optional): dictionary mapping a FockStates object to a generated OperSequenceData
        label (str) (optional): label to assign for plotting purposes, set by set_label()

    """
    ####################################################
    #####      INITIALIZATION AND FORMATTING       #####
    ####################################################
    def __init__(self, *operators: tuple[List[int],str,int,tuple], weights: List = None, bypass_parse: bool =False):
        """
        Initialise a sequence of operators with specified weights

        """
        ## For internally created OperSequence objects, directly load data
        if bypass_parse:
            self.oper_list = operators[0]
            self.weights = (
                weights if weights is not None else [1 for i in range(len(self.oper_list))]
            )
            return

        ## Parse the input operators and check shape of input weights if provided
        parsed_operators = self.parse_operator_input(*operators)
        if weights is not None:
            if len(weights) != len(parsed_operators):
                raise ValueError("Length of provided weights does not match number of subsequences")
        self.oper_list = parsed_operators
        self.weights = (
            weights if weights is not None else [1 for i in range(len(self.oper_list))]
        )

    @staticmethod
    def _format_weight_for_repr(w: complex) -> str :
        w_str = ""
        ## w purely real
        if w.imag == 0 and w.real != 0:
            if w.real > 0:
                w_str += " $+$ "
            else:
                w_str += " $-$ "
            if abs(w) == 1:
                return w_str
            else:
                w_str += f"{abs(np.round(w,3))}"
                return w_str

        ## w purely imaginary
        elif w.real == 0 and w.imag != 0:
            if w.imag > 0:
                w_str += " $+$ "
            else:
                w_str += " $-$ "
            if abs(w) == 1:
                w_str += "j"
                return w_str
            else:
                w_str += f"{abs(np.round(w,3))}j"
                return w_str

        ## w is a mix
        w_str += " $+$ " + f"{np.round(w,3)}"
        return w_str

    def _repr_markdown_(self):
        operstr = ""
        for w, oper_seq in zip(self.weights, self.oper_list):
            if isinstance(oper_seq, list):
                operstr += self._format_weight_for_repr(w) + self.vis_oper_list(oper_seq)
            else:
                operstr += " $+$ " + f"{w*oper_seq}"
        if operstr[0:4] == " $+$":
            return operstr[4:]
        return operstr
        
    def symbolic(self) -> None:
        operstr = ""
        for oper_seq in self.oper_list:
            if isinstance(oper_seq, list):
                symbol = _symbol_dict.get(self.oper_list_to_str(oper_seq),'')
                operstr += ' + ' + symbol + self.vis_oper_list(oper_seq)
        display(Markdown(operstr[3:]))
        
    
    def set_symbolic(self,symbol_strings:List[str]) -> None:
        if isinstance(symbol_strings,list):
            if len(symbol_strings) != len(self.oper_list):
                raise ValueError(f"Length of input list ({len(symbol_strings)}) does not match number of subsequences ({len(self.oper_list)})")
            for idx,symbol in enumerate(symbol_strings):
                operators = self.oper_list[idx]
                if not isinstance(operators,list):
                    raise ValueError(f"Cannot assign symbol to: {operators}")
                oper_string = self.oper_list_to_str(operators)
                _symbol_dict[oper_string] = symbol
            save_dictionary(_symbol_dict)
            
    def set_label(self,label:str) -> None:
        self.label = str(label)
    
    #######################################################
    #####      GETTING AND SETTING SUBSEQUENCES       #####
    #######################################################
    _parseinput_handlers = {
        int: '_parse_int',
        str: '_parse_str',
        list:'_parse_list',
        tuple:'_parse_tuple',
    }

    def parse_operator_input(self,*operators:tuple[List[int],str,int,tuple]) -> List[List[int]]:
        output_list = []
        for op in operators:
            handler_func_name = self._parseinput_handlers.get(type(op), '_handle_parse_default')
            handler_func = getattr(self,handler_func_name)
            output_list.append(handler_func(op))

        return output_list

    @staticmethod
    def _handle_parse_default(operators):
        raise TypeError("You done goofed boy")

    @staticmethod
    def _parse_int(operator_int: int) -> List[int]:
        return [operator_int]
    
    @staticmethod
    def _parse_str(operator_str: str) -> List[int]:
        split_str = operator_str.split(',')
        parsed_str = [operator_from_string(s) for s in split_str]
        parsed_str.reverse()
        return parsed_str

    @staticmethod
    def _parse_list(operator_list: List[int]) -> List[int]:
        for item in operator_list:
            if isinstance(item, list):
                raise TypeError('Nested lists can not be parsed as input type')
        output = list(operator_list)
        output.reverse()
        return output

    @staticmethod
    def _parse_tuple(operator_tuple: tuple[int,str]) -> List[int]:
        if isinstance(operator_tuple[0], tuple):
            output = []
            for subop_tuple in operator_tuple:
                for item in subop_tuple:
                    if isinstance(item,tuple):
                        raise TypeError('Not possible to parse a nested tuple with dimension > 2')
                output.append(operator_verbose(*subop_tuple))
            output.reverse()
            return output
        else:
            return [operator_verbose(*operator_tuple)]

    _getitem_handlers = {
        slice:'_get_slice',
        tuple:'_get_tuple',
        np.ndarray:'_get_ndarray',
        int: '_get_int',
        str: '_get_str',
    }

    def __contains__(self, subsequence):
        return all(item in self.oper_list for item in subsequence.oper_list)

    def __getitem__(self,indexing) -> "OperSequence":
        if isinstance(indexing, (sys.modules[__name__].FockStates,sys.modules[__name__].FockStatesVerySparse)):
            if not hasattr(self,'basis_dict'):
                return self.__and__(indexing)
            if indexing not in self.basis_dict:
                return self.__and__(indexing)
            return self.basis_dict[indexing] 

        handler_func_name = self._getitem_handlers.get(type(indexing), '_handle_get_default')
        handler_func = getattr(self,handler_func_name)
        return handler_func(indexing)

    def _handle_get_default(self,indexing: Any):
        raise TypeError(f"Getting subsequence by {type(indexing).__name__} not supported")
    
    def _get_slice(self,indexing: slice):
        subsequence = OperSequence(deepcopy(self.oper_list)[indexing],weights = deepcopy(self.weights)[indexing], bypass_parse=True)
        return subsequence
        
    def _get_tuple(self,indexing: tuple):
        new_list = deepcopy(self.oper_list)
        subsequence = OperSequence([new_list[i] for i in indexing], weights = [self.weights[i] for i in indexing], bypass_parse=True)
        return subsequence
    
    def _get_ndarray(self,indexing: np.ndarray):
        print("array slicing not yet supported")

    def _get_int(self,indexing: int):
        subsequence = OperSequence([deepcopy(self.oper_list[indexing])], weights = [self.weights[indexing]], bypass_parse=True)
        return subsequence

    def _get_str(self,indexing: str):
        print("string slicing not yet supported")

    def __call__(self, subsequence,value = None):
        raise TypeError(f"Calling {self.__class__.__name__} with {type(subsequence).__name__} is not yet supported")

    _setitem_handlers = {
        slice:'_set_slice',
        tuple:'_set_tuple',
        np.ndarray:'_set_ndarray',
        int:'_set_int',
        str:'_set_str',
    }

    def __setitem__(self, indexing, value: [int,float,complex]):
        if isinstance(indexing, self.__class__):
            old_val, str_list = self._set_opersequence(indexing,value)
        else:
            handler_func_name = self._setitem_handlers.get(type(indexing), '_handle_set_default')
            handler_func = getattr(self,handler_func_name)
            old_val, str_list = handler_func(indexing,value)
        if hasattr(self,'basis_dict'):
            for fock_basis, sparse_representation in self.basis_dict.items():
                sparse_representation.update_values(str_list, old_val, value)

    def _handle_set_default(self, indexing, value):
        raise TypeError(f"Setting value by {type(indexing).__name__} not supported")

    def _set_slice(self,indexing, value):
        print("Setting value by slice not yet supported")

    def _set_tuple(self,indexing, value):
        print("Setting value  by tuple not yet supported")

    def _set_ndarray(self,indexing, value):
        print("Setting value  by numpy array not yet supported")

    def _set_int(self,indexing, value):
        print("Setting value  by int not yet supported")

    def _set_str(self,indexing, value):
        print("Setting value by str not yet supported")

    def _set_opersequence(self, indexing,value):
        old_value = self.weights[self.oper_list.index(indexing.oper_list[0])]
        str_list = []
        for i in range(len(indexing.oper_list)):
            check_item = indexing.oper_list[i]
            if check_item in self.oper_list:
                self.weights[self.oper_list.index(check_item)] = value
                str_list.append(self.oper_list_to_str(check_item))
        return old_value, str_list

    def __gt__(self, values: [int,float,complex,List,np.ndarray]):
        if (
            isinstance(values,(int,float,complex))
        ):
            self.weights = [values for i in range(len(self.oper_list))]
        else:
            if len(self.oper_list) != len(values):
                raise ValueError(f"Length of input array ({len(values)}) does not match the number of subsequences ({len(self.oper_list)})")
            self.weights = values
        return self

    ######################################################
    #####             OPERATOR ARITHMETIC            #####
    ######################################################

    def __add__(self, oper_sequence):            
        if not isinstance(oper_sequence, (self.__class__,int,float,complex)):
            raise TypeError(f"Unsupported operand type(s) for +: {self.__class__.__name__} and '{type(oper_sequence).__name__}'")
        new_weights, new_oper_list = [],[]
        new_weights.extend(self.weights)
        new_oper_list.extend(self.oper_list)
        if isinstance(oper_sequence, (int,float,complex)):
            new_weights.append(oper_sequence)
            new_oper_list.append(1)
        else:
            new_weights.extend(oper_sequence.weights)
            new_oper_list.extend(oper_sequence.oper_list)
        result = OperSequence(new_oper_list, weights = new_weights, bypass_parse=True)
        result.merge_terms()
        return result

    def __iadd__(self, oper_sequence):
        if not isinstance(oper_sequence, (self.__class__,int,float,complex)):
            raise TypeError(f"Unsupported operand type(s) for +: {self.__class__.__name__} and '{type(oper_sequence).__name__}'")
        if isinstance(oper_sequence, (int,float,complex)):
            self.weights.append(oper_sequence)
            self.oper_list.append(1)
        else:
            self.weights.extend(oper_sequence.weights)
            self.oper_list.extend(oper_sequence.oper_list)
        self.merge_terms()
        return self

    def __sub__(self, oper_sequence):
        if not isinstance(oper_sequence, (self.__class__,int,float,complex)):
            raise TypeError(f"Unsupported operand type(s) for +: {self.__class__.__name__} and '{type(oper_sequence).__name__}'")
        new_weights = []
        new_oper_list = []
        new_weights.extend(self.weights)
        new_oper_list.extend(self.oper_list)
        if isinstance(oper_sequence, (int,float,complex)):
            new_weights.append(-oper_sequence)
            new_oper_list.append(1)
        else:
            new_weights.extend([-1 * w for w in oper_sequence.weights])
            new_oper_list.extend(oper_sequence.oper_list)

        result = OperSequence(new_oper_list, weights = new_weights, bypass_parse=True)
        result.merge_terms()
        return result

    def __isub__(self, oper_sequence):
        if not isinstance(oper_sequence, (self.__class__,int,float,complex)):
            return
        if isinstance(oper_sequence, (int,float,complex)):
            self.weights.append(-oper_sequence)
            self.oper_list.append(1)
        else:
            self.oper_list.extend(oper_sequence.oper_list)
            self.weights.extend([-1 * w for w in oper_sequence.weights])
        self.merge_terms()
        return self

    def __pos__(self) -> Self:
        return self

    def __neg__(self):
        new_weights = [-1 * w for w in self.weights]
        negative_oper = OperSequence(deepcopy(self.oper_list), weights= new_weights, bypass_parse=True)
        return negative_oper

    def __pow__(self, exponent: int):
        if not isinstance(exponent, int):
            return self
        product = deepcopy(self)
        for i in range(1, exponent):
            product *= self
        return product

    def __ipow__(self, exponent: int) -> Self:
        if not isinstance(exponent, int):
            return self
        for i in range(1, exponent):
            self *= self
        return self

    def __mul__(self, multiplier: ["OperSequence",int,float,complex]) -> "OperSequence":
        if (
            isinstance(multiplier, int)
            or isinstance(multiplier, complex)
            or isinstance(multiplier, float)
        ):
            new_weights = [w * multiplier for w in self.weights]
            product = OperSequence(self.oper_list, weights = new_weights,bypass_parse=True)
            return product
        product_outcome = self._multiplication_basis(
            self.weights, self.oper_list, multiplier.weights, multiplier.oper_list
        )
        product = OperSequence(product_outcome[1], weights = product_outcome[0], bypass_parse=True)
        product.normal_order()
        product.remove_duplicates()
        product.merge_terms()
        product.remove_zero_weight()
        return product

    def __rmul__(self, multiplier) ->  "OperSequence":
        if not isinstance(multiplier, (int,float,complex)):
            raise ValueError(f"Cannot multiply OperSequence with {type(multiplier)}")

        new_weights = [w * multiplier for w in self.weights]
        product = OperSequence(self.oper_list, weights = new_weights, bypass_parse=True)
        return product

    def __imul__(self, multiplier) -> Self:
        if (
            isinstance(multiplier, int)
            or isinstance(multiplier, complex)
            or isinstance(multiplier, float)
        ):
            self.weights = [w * multiplier for w in self.weights]
            return self

        product_outcome = self._multiplication_basis(
            self.weights, self.oper_list, multiplier.weights, multiplier.oper_list
        )
        self.oper_list = product_outcome[1]
        self.weights = product_outcome[0]
        self.normal_order()
        self.merge_terms()
        self.remove_duplicates()
        self.remove_zero_weight()
        return self

    def __truediv__(self,divisor: [int,float]) ->  "OperSequence":
        """
        / operator
        Divides the weights of the OperSequence by divisor, returns new OperSequence
        """
        if not isinstance(divisor, (int,float)):
            raise TypeError(f"Unsupported operand type(s) for /: {self.__class__.__name__} and '{type(divisor).__name__}'")
        oper_list = self.oper_list[:]
        new_weights = [w/divisor for w in self.weights]
        return  OperSequence(oper_list, weights = new_weights, bypass_parse=True)

    def __itruediv__(self,divisor:[int,float]) -> Self:
        """
        in-place / operator
        Divides the weights of the OperSequence by divisor
        """
        if not isinstance(divisor, (int,float)):
            raise TypeError(f"Unsupported operand type(s) for /: {self.__class__.__name__} and '{type(divisor).__name__}'")
        self.weights = [w/divisor for w in self.weights]
        return self
        
    @staticmethod
    def _multiplication_basis(weights_1, oper_list_1, weights_2, oper_list_2):
        #return fo.multiplication_basis(weights_1,oper_list_1, weights_2, oper_list_2)
        oper_products = []
        product_weights = []
        for idx_1, oper_seq_1 in enumerate(oper_list_1):
            for idx_2, oper_seq_2 in enumerate(oper_list_2):
                if isinstance(oper_seq_1, list) and isinstance(oper_seq_2, list):
                    temp = oper_seq_2.copy()
                    temp.extend(oper_seq_1)
                    oper_products.append(temp)
                    product_weights.append(weights_1[idx_1] * weights_2[idx_2])
                elif isinstance(oper_seq_1, list):
                    oper_products.append(oper_seq_1.copy())
                    product_weights.append(
                        weights_1[idx_1] * oper_seq_2 * weights_2[idx_2]
                    )
                elif isinstance(oper_seq_2, list):
                    oper_products.append(oper_seq_2.copy())
                    product_weights.append(
                        weights_2[idx_2] * oper_seq_1 * weights_1[idx_1]
                    )
                else:
                    oper_products.append(oper_seq_1 * oper_seq_2)
                    product_weights.append(weights_2[idx_2] * weights_1[idx_1])
        return product_weights, oper_products

    def __mod__(self, modulo: int):
        """
        % operator
        Returns True if the length of each subsequence satisfies length % modulo = 0,
        returns False otherwise.
        """
        if not isinstance(modulo, int):
            return False
        for oper_seq in self.oper_list:
            if isinstance(oper_seq, list):
                if len(oper_seq) % modulo != 0:
                    return False
        return True

    def __invert__(self) ->  "OperSequence":
        """
        ~ operator
        Returns a new OperSequence representing the complex conjugate of the original sequence
        Does not normal order the new sequence before returning.
        """
        invert_list = []
        flip_bit = 0b1
        for oper_seq in deepcopy(self.oper_list):
            new_opers = []
            if not isinstance(oper_seq, list):
                if isinstance(oper_seq, complex):
                    invert_list.append(oper_seq.conjugate())
                    continue
                invert_list.append(oper_seq)
                continue

            for oper in oper_seq:
                conjugate_oper = oper ^ flip_bit
                new_opers.append(conjugate_oper)
            new_opers.reverse()
            invert_list.append(new_opers)

        inverse = OperSequence(
            invert_list,
            weights = [w.conjugate() if isinstance(w, complex) else w for w in self.weights],
            bypass_parse=True
        )
        return inverse

    def __rshift__(self, shift_by: int) ->  "OperSequence":
        """
        >> operator.
        Increments the site that each operator in oper_list acts on.

        Args:
            shift_by (int): the amount of sites to decrease by
        """
        if not isinstance(shift_by, (int, float,np.ndarray,list)):
            raise TypeError(f"Unsupported operand type(s) for >>: {self.__class__.__name__} and '{type(shift_by).__name__}'")

        if isinstance(shift_by,(int,float)):
            new_opers = []
            for oper_seq in deepcopy(self.oper_list):
                if not isinstance(oper_seq, list):
                    new_opers.append(oper_seq)
                    continue
                for i in range(len(oper_seq)):
                    oper_seq[i] += (int(shift_by) * 2) << 1
                new_opers.append(oper_seq)
            shifted_sequence = OperSequence(new_opers, weights = self.weights[:], bypass_parse=True)
            return shifted_sequence
        else:
            base_sequence = OperSequence(deepcopy(self.oper_list), weights=self.weights[:], bypass_parse=True)
            shift_sequence = OperSequence(deepcopy(self.oper_list), weights=self.weights[:],bypass_parse=True)
            for shift in shift_by:
                shift_sequence += (base_sequence >> int(shift))
            return shift_sequence
    
    def __irshift__(self, shift_by) -> Self:
        """
        In-place >> operator.
        Increments the site that each operator in oper_list acts on.

        Args:
            shift_by (int): the amount of sites to decrease by
        """
        if not isinstance(shift_by, (int, float)):
            raise TypeError(f"Unsupported operand type(s) for >>=: {self.__class__.__name__} and '{type(shift_by).__name__}'")

        for oper_seq in self.oper_list:
            if not isinstance(oper_seq, list):
                continue
            for i in range(len(oper_seq)):
                oper_seq[i] += (int(shift_by) * 2) << 1
        return self

    def __lshift__(self, shift_by: int) ->  "OperSequence":
        """
        << operator.
        Decrements the site that each operator in oper_list acts on.

        Args:
            shift_by (int): the amount of sites to decrease by
        """
        if not isinstance(shift_by, (int, float)):
            raise TypeError(f"Unsupported operand type(s) for <<: {self.__class__.__name__} and '{type(shift_by).__name__}'")

        new_opers = []
        for oper_seq in deepcopy(self.oper_list):
            if not isinstance(oper_seq, list):
                new_opers.append(oper_seq)
                continue
            for i in range(len(oper_seq)):
                oper_seq[i] -= (int(shift_by) * 2) << 1
            new_opers.append(oper_seq)
        shifted_sequence = OperSequence(new_opers, weights = self.weights[:],bypass_parse=True)
        return shifted_sequence

    def __ilshift__(self, shift_by: int) -> Self:
        """
        In-place << operator.
        Decrements the site that each operator in oper_list acts on.

        Args:
            shift_by (int): the amount of sites to decrease by
        """
        if not isinstance(shift_by, (int, float)):
            raise TypeError(f"Unsupported operand type(s) for <<=: {self.__class__.__name__} and '{type(shift_by).__name__}'")

        for oper_seq in self.oper_list:
            if not isinstance(oper_seq, list):
                continue
            for i in range(len(oper_seq)):
                oper_seq[i] -= (int(shift_by) * 2) << 1
        return self

    ##############################################
    ######       COMPARISON STATEMENTS       #####
    ##############################################
    def _normalized(self) -> List[tuple]:
        """
        Creates a tuple format of the oper_list and weight attributes, that allows
        for checking equality/inequality in __eq__ and __ne___.
        """
        normalized_operator_list = [
            tuple(sub) if isinstance(sub, list) else sub for sub in self.oper_list
        ]
        normalized_sequence = [
            (w, l) for w, l in zip(self.weights, normalized_operator_list)
        ]
        return normalized_sequence

    def __eq__(self, oper_sequence:  "OperSequence") -> bool:
        """
        Check equality of two OperSequences. 
        Two sequences are determined equal
        if they have identical subsequences with the same weights, regardless of the
        order they appear in.
        """
        if not isinstance(oper_sequence, OperSequence):
            return False
        return Counter(self._normalized()) == Counter(oper_sequence._normalized())

    def __ne__(self, oper_sequence:  "OperSequence") -> bool:
        """
        Check inequality of two OperSequences. 
        Two sequences are determined equal
        if they have identical subsequences with the same weights, regardless of the
        order they appear in.
        """
        if not isinstance(oper_sequence,  "OperSequence"):
            return True
        return Counter(self._normalized()) != Counter(oper_sequence.normalized())
    
    ###########################################################
    #####        CONNECTING OPERATOR TO FOCK STATES       #####
    ###########################################################

    def __and__(self, fock_states: FockStates) -> OperSequenceData:
        """
        Shorthand code to generate an OperSequenceData object for a specified FockStates object.
        """

        if not isinstance(fock_states, (sys.modules[__name__].FockStates,sys.modules[__name__].FockStatesVerySparse)):
            raise ValueError(f'Cannot bind OperSequence to {type(fock_states)}')
        if not hasattr(self,'basis_dict'):
            self.basis_dict={}
        
        if fock_states in self.basis_dict.keys():
            return
        else:
            if isinstance(fock_states, sys.modules[__name__].FockStates):
                sparse_data = self._construct_sparse_repr(fock_states)
                self.basis_dict[fock_states] = OperSequenceData(sparse_data,fock_states)
                return self.basis_dict[fock_states]
            if isinstance(fock_states, sys.modules[__name__].FockStatesVerySparse):
                self.basis_dict[fock_states] = FockOperVerySparse(fock_states, self.oper_list,self.weights)
                return self.basis_dict[fock_states]
       
    def _construct_sparse_repr(self,fock_states):
        """
        Applies the stored hamiltonian to a set of Fock states
        Returns:
            rows (ndarray[int]): row indices of non-zero terms
            cols (ndarray[int]): column indices of non-zero terms
            pars (ndarray[int]): relative signs of operators for the non-zero terms
            type (ndarray[str]): the types of operators giving rise to non-zero terms
        """
        rows, cols, pars, types,vals = [], [], [], [], []
        for h, w in zip(self.oper_list, self.weights):
            if isinstance(h,list):
                type_str = self.oper_list_to_str(h)
                old_states, new_states, parities = self.act_oper_list(
                    h, fock_states.states, rel_sign=1
                )
                subspace_filt = [state in fock_states.hashed for state in new_states]
                old_states = old_states[subspace_filt]
                new_states = new_states[subspace_filt]
                parities = parities[subspace_filt]

                types.extend([type_str] * len(parities))
                rows.extend([fock_states.hashed.get(state) for state in old_states])
                cols.extend([fock_states.hashed.get(state) for state in new_states])
                pars.extend(parities.tolist())
                vals.extend([w for _ in range(len(parities))])

        list_1, list_2 = map(
            list,
            zip(
                *[
                    (b, a) if a > b else (a, b)
                    for a, b in zip(rows, cols)
                ]
            ),
        )
        rows = np.array(list_1)
        cols = np.array(list_2)

        return (
            np.array(rows, dtype=np.int32),
            np.array(cols, dtype=np.int32),
            np.array(pars, dtype=np.int32),
            np.array(types,dtype=str),
            np.array(vals, dtype = complex)
        )
    
    ###########################################################
    #####     ORDERING AND CLEANING OPERATOR SEQUENCE     #####
    ###########################################################

    def sort(self) -> None:
        store_ints = []
        for idx in range(len(self.weights)-1,-1,-1):
            if not isinstance(self.oper_list[idx], list):
                store_ints.append((self.weights[idx], self.oper_list[idx]))
                self.oper_list.pop(idx)
                self.weights.pop(idx)


        nested = deepcopy(self.oper_list)
        weights = deepcopy(self.weights)

        max_len = max(len(lst) for lst in nested)
        padded_nested = [([-1] * (max_len - len(lst)) + lst) for lst in nested]
        
        # Zip and sort by reversed sublist (i.e., last digit first)
        combined = list(zip(padded_nested, nested, weights))
        sorted_combined = sorted(combined, key=lambda x: x[0][::-1])
        
        # Extract the results
        sorted_nested = [x[1] for x in sorted_combined]
        sorted_weights = [x[2] for x in sorted_combined]

        final_sorted = sorted(zip(sorted_nested, sorted_weights), key=lambda x: len(x[0]))
        final_nested = [x[0] for x in final_sorted]
        final_weights = [x[1] for x in final_sorted]
        
        if len(store_ints)>0:
            self.weights = [i[0] for i in store_ints]
            self.oper_list = [i[1] for i in store_ints]
            self.weights.extend(final_weights)
            self.oper_list.extend(final_nested)
        else:
            self.weights = final_weights
            self.oper_list = final_nested
        

    def remove_zero_weight(self) ->  "OperSequence":
        for idx in range(len(self.weights)-1, -1,-1):
            if abs(np.round(self.weights[idx],13)) == 0.0:
                self.weights.pop(idx)
                self.oper_list.pop(idx)
        return self

    ## Remove duplicate operators
    def remove_duplicates(self) -> None:
        self.weights,self.oper_list = fo.remove_duplicates(self.weights,self.oper_list)
        return

        for idx in range(len(self.weights) - 1, -1, -1):
            if not isinstance(self.oper_list[idx], list):
                continue
            elif len(self.oper_list[idx]) != len(set(self.oper_list[idx])):
                self.oper_list.pop(idx)
                self.weights.pop(idx)

    def merge_terms(self) -> None:
        self.weights,self.oper_list = fo.merge_terms_cython(self.weights,self.oper_list)
        return
        merged_weight,merged_list = [],[]
    
        for seq,weight in zip(self.oper_list, self.weights):
            if seq not in merged_list:
                merged_list.append(seq)
                merged_weight.append(weight)
            else:
                get_idx = merged_list.index(seq)
                merged_weight[get_idx]+=weight
        self.weights,self.oper_list = merged_weight,merged_list

    def is_normal_ordered(self):
        for oper_seq in self.oper_list:
            if isinstance(oper_seq, list):
                for i in range(len(oper_seq) - 1):
                    if (oper_seq[i] % 2) < (oper_seq[i + 1] % 2):
                        return False
                    elif (oper_seq[i] < oper_seq[i + 1]) and (
                        (oper_seq[i] % 2) < (oper_seq[i + 1] % 2)
                    ):
                        return False
        return True

    def normal_order(self):
        self.weights,self.oper_list = fo.normal_order(self.weights,self.oper_list)
        return 

        is_normal_ordered=False
        while not is_normal_ordered:
            is_normal_ordered=True
            for seq_idx, oper_seq in enumerate(self.oper_list):
                if not isinstance(oper_seq, list):
                    continue

                n = len(oper_seq)
                for i in range(n - 1):
                    flag_swap = False
                    for j in range(n - 1 - i):
                        if (oper_seq[j] % 2) > (oper_seq[j + 1] % 2):
                            continue
                        if oper_seq[j] < oper_seq[j + 1] or (oper_seq[j] % 2) < (
                            oper_seq[j + 1] % 2
                        ):
                            is_normal_ordered = False
                            if (oper_seq[j] ^ 0b1) == oper_seq[j + 1]:
                                self.oper_list.append(oper_seq[:])
                                self.weights.append(self.weights[seq_idx] * -1)

                                self.oper_list[seq_idx] = oper_seq[:j] + oper_seq[j + 2 :]
                                self.oper_list[-1][j], self.oper_list[-1][j + 1] = (
                                    self.oper_list[-1][j + 1],
                                    self.oper_list[-1][j],
                                )
                                flag_swap=False
                                if len(self.oper_list[seq_idx]) == 0:
                                    self.weights.append(self.weights[seq_idx])
                                    self.weights.pop(seq_idx)
                                    self.oper_list.pop(seq_idx)
                                    self.oper_list.append(1)
                                break
                            else:
                                flag_swap = True
                                oper_seq[j], oper_seq[j + 1] = (
                                    oper_seq[j + 1],
                                    oper_seq[j],
                                )
                                self.weights[seq_idx] *= -1

                    ## If no swap took place, list is sorted
                    if not flag_swap:
                        break
        return self

