import numpy as np
from IPython.display import display, Markdown

from typing import List

def printmd(item):
    display(Markdown(item._repr_markdown))

def hamming_weight(states: np.ndarray):
    """
    Constant-time, vectorized method for counting number of 1's in a number in binary.
    Works up to 32-bit integers.

    Shoutout to stackoverflow.

    Args:
        states (np.ndarray): Array of integers
    Returns:
        np.ndarray: Number of 1's in each inputs' binary representation
    """
    count = states - ((states >> 1) & 0o33333333333) - ((states >> 2) & 0o11111111111)
    return ((count + (count >> 3)) & 0o30707070707) % 63

def operator_verbose(type_str: str, position: int, spin: str):
    """
    Verbose method for constructing an operator, passes the call on to
    self.construct_operator
    Args:
        type (str): specify whether to make an annihilation or creation operator
        position (int): the fermionic site to act on
        spin (str): 'up' or 'down'
    Returns:
        oper (int): integer representing the operator
    """
    annihilation = type_str.startswith("a")
    spin = 0 if (spin.startswith("d")) else 1
    oper = construct_operator(position, spin, annihilation)
    return oper

def operator_from_string(type_str):
    """
    Method for constructing an operator from a string sequence, implementing
    the convention utilized by OpenFermion. Passes a
    correctly formatted call on to construct_operator()
    """
    position = int(type_str[0])//2
    spin = int(type_str[0]) &0B1
    creation = 0 if type_str[-1] == '^' else 1
    return construct_operator(position,spin, creation)

def construct_operator(position: int, spin: int, annihilation: bool = True):
    """
    Creates the integer representing an operator. The first bits store the bit position of
    the operator to act on, which depends on position and spin
    The largest bit is set or unset depending on whether it stores a creation or annihilation operator
    Args:
        position (int): fermionic site of operator to act on
        spin (int): 0 = spin down, 1 = spin up
        creation(bool): True = creation, False = annihiliation

    Returns:
        oper (int): integer representing an operator
    """
    shift = ((position << 1) + spin) << 1
    oper = shift + annihilation
    return oper

class FockSystemBase:
    def __init__(self, N=5, store_fock_states=True):
        """
        Base Class for handling fermionic operator and states logic through binary operations
    
        """

    def act_oper(self, oper: int, states: np.ndarray):
        """
        Applies an operator to a list of states, to generate the new states
        A destructed state is shown as -1, to distuinguish it from the empty state (0)

        Args:
            oper (int): integer representing an operator
            states (ndarray): list of integers representing Fock states
        Returns:
            new_states: list of the states after applying the operator
            signs: list of signs resulting from applying the operator
        """
        ## Check the creation/annihilation bit
        check_bit = oper & 0b1

        ## Get the position of the bit to flip
        flip_bit_pos = oper >> 1
        flip_bit = 1 << flip_bit_pos

        ## Calculate relative signs of the operator
        parity_bits = states & (flip_bit-1) 
        signs = hamming_weight(states & parity_bits) & 0b1
        signs = signs * -2 + 1

        ## Flip the bit that the operator acts on
        new_states = states ^ flip_bit

        ## Check the original state of the flipped bit with the creation/annihilation bit
        destroyed = ((flip_bit & states) == 0) == check_bit
        new_states[
            destroyed
        ] = -1  ## -1 = destroyed state (0 is already taken by the empty state)
        ## Benefit: any state *-1 will also be <0
        return new_states, signs

    def act_oper_list(self, oper_list, states, rel_sign=1):
        """
        Apply a sequence of operators to a list of states,
        removes any 'destroyed' terms
        Args:
            oper_list: list of ints representing a sequence of states
            states: list of ints representing Fock states
            rel_sign: the relative sign of the operator
        Returns:
            states: list of the original states which were not destroyed
            new_states: resulting new states
            signs: relative sign after applying the sequence of operators
        """
        signs = np.full(len(states), 1)
        new_states = states
        for oper in oper_list:
            new_states, new_parity = self.act_oper(oper, new_states)
            filter_states = np.where(new_states > -1)
            new_states = new_states[filter_states]
            signs = signs[filter_states]
            new_parity = new_parity[filter_states]
            states = states[filter_states]
            signs *= new_parity
        return states, new_states, signs * rel_sign

    @staticmethod
    def normal_order_naive(oper_list: list):
        """
        Place sequence of operators in normal order, tracking the sign
        Normal order = sorting the list of integers from smallest to largest
        Sorting done with a bubble sort.
        To do: handle the presence of same-site operators (for now ignored)
        Args:
            oper_list (list)
        """
        ferm_sign = 1
        for i in range(len(oper_list) - 1):
            flag_swap = False
            for j in range(len(oper_list) - 1 - i):
                if (oper_list[j] % 2) > (oper_list[j + 1] % 2):
                    continue
                if oper_list[j] < oper_list[j + 1] or (oper_list[j] % 2) < (
                    oper_list[j + 1] % 2
                ):
                    flag_swap = True
                    oper_list[j], oper_list[j + 1] = oper_list[j + 1], oper_list[j]
                    ferm_sign *= -1
            ## if no swap took place, list is sorted
            if not flag_swap:
                return oper_list, ferm_sign

        return oper_list, ferm_sign

    def state_to_oper_list(self, state: int):
        """
        Convert from a state to the list of creation operators
        to create the state from vaccuum.
        """
        oper_list = []
        pos = 0
        while state:
            if state & 0b1:
                oper = construct_operator(
                    int(np.floor(pos / 2)), pos % 2, annihilation=False
                )
                oper_list.append(oper)
            state = state >> 1
            pos += 1
        return self.normal_order_naive(oper_list)[0]

    def bra_oper_ket(self, states: list, phi: np.ndarray, operators: list):
        """
        Calculate <phi|operator|phi> matrix for an operator sequence
        If multiple operator sequences are included, the contributions will be summed
        Args:
            states: list of states to operate on
            phi: array of weights of the states
            operators: list of operator lists to apply
        """
        size = len(phi[:, 0])
        T_ij = np.zeros((size, size), dtype=complex)
        zero_col = np.zeros(size, dtype=complex)
        trans_array = np.zeros((len(phi[0]), len(phi)), dtype="complex")

        for oper_list in operators:
            old_states, new_states, signs = self.act_oper_list(oper_list, states)

            col_indices = [
                np.where(states == old_states[i])[0][0]
                for i in range(len(old_states))
                if new_states[i] in states
            ]
            row_indices = [
                np.where(states == new_states[i])[0][0]
                for i in range(len(new_states))
                if new_states[i] in states
            ]
            relevant_signs = [
                [signs[i]] for i in range(len(signs)) if new_states[i] in states
            ]

            if len(row_indices) > 0:
                trans_array[row_indices] = (
                    np.transpose(phi[:, col_indices]) * relevant_signs
                )
                T_ij += np.conj(phi) @ trans_array
        return T_ij

    @staticmethod
    def vis_oper(oper: int):
        """
        Convert an integer representing an operator to a readable string
        Args:
            oper (int): integer representing an operator
        Returns:
            (str): string visualizing the operator

        """
        spins = ["\u2193", "\u2191"]

        creation = (oper & 0b1) == 0
        act_pos = oper >> 1
        act_pos_proper = np.floor(act_pos / 2)
        spin_vis = spins[act_pos % 2]

        if creation:
            return "$c^{\u2020}_{" + f"{int(act_pos_proper)},{spin_vis}" + "}$"
        else:
            return "$c_{" + f"{int(act_pos_proper)},{spin_vis}" + "}$"

    def vis_oper_list(self, oper_list:List[List[int]], displ: bool=False):
        """
        Convert a sequence of operators to a readable string
        """
        full_str = ""

        for oper in list(reversed(oper_list)):
            full_str += self.vis_oper(oper)
        if displ:
            display(Markdown(full_str))
        else:
            return full_str

    def vis_oper_str(self, oper_str:str, displ:bool = False):
        oper_list = self.oper_str_to_list(oper_str)
        output = self.vis_oper_list(oper_list, displ=displ)
        return output

    def vis_state(self, state, displ=False,N=None):
        """
        Show a 'state' in a readable fashion
        args:
            state (int): the Fock state to visualize
            displ (bool): if True, prints the state
                          if False, returns a string
        """
        n = state
        if isinstance(n, np.ndarray):
            if len(n) == 0:
                return "0"
            elif n[0] == -1:
                return "0"
        elif n < 0:
            return "0"

        str = "\u007c"
        sgn = 1
        count = 0
        while n:
            check_up = n & 1
            n >>= 1
            check_down = n & 1
            n >>= 1
            if check_up and check_down:
                str += "\u2193\u2191,"
            elif check_up:
                str += "\u2193,"
            elif check_down:
                str += "\u2191,"
            else:
                str += "0,"
            count += 1
        if N is not None:
            while count < N:
                str += "0,"
                count += 1
        str = str[:-1]
        str += "\u3009"
        if displ:
            display(Markdown(str))
        else:
            return str

    @staticmethod
    def format_w(w):
        w_str = ''
        if w>0:
            w_str += '+ '
        if w == 1:
            return w_str
        if w.imag == 0:
            w_str+=f'{w.real}'
            return w.str
        w_str+= f'{w}'
        return w_str

    def vis_state_list(self, states, weights=None,N=None, displ=False, ):
        """
        Visualise a superposition of Fock states
        Args:
            states (list): list of integers representign states
            weights (floats): the weight of each state
            displ (bool): if True prints the visualisation
                          if False returns the string
        """
        str = ""
        if weights is not None:
            for w, s in zip(weights, states):
                w = np.round(w, 3)
                if w != 0:
                    str += (
                        self.format_w(w)
                        + self.vis_state(s, displ=False,N=N)
                    )
        else:
            for state in states:
                str += self.vis_state(state, displ=False,N=N) + " "
        if displ:
            display(Markdown(str))
        else:
            return str

    @staticmethod
    def oper_list_to_str(oper_list):
        """
        Convert a list of integers representing operators
        to a string representation
        """
        return ".".join(map(str, oper_list))

    @staticmethod
    def oper_str_to_list(oper_str):
        """
        Convert a string representing operators
        to a list of operators
        """
        return list(map(int, oper_str.split(".")))

    @staticmethod
    def conjugate_list(oper_list):
        flip_bit = 0b1
        new_opers = []
        for oper in oper_list:
            conjugate_oper = oper ^ 0b1
            new_opers.append(conjugate_oper)
        new_opers.reverse()
        return new_opers

