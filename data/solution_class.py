#
# solution_class.py
#
"""module for the class solution"""

import os
import sys
import json
import numpy as np
from . import reagent
from . import reaction_mechanism_class


class solution(reaction_mechanism_class.reaction_mechanism):
    """The solution class"""

    # default file properties
    file_prefix = os.path.join(os.getcwd(), "Puzzles")
    file_suffix = ".solution"

    @staticmethod
    def test():
        number_of_tests = 2
        test = [False] * number_of_tests
        # run this test
        test[0] = True

        if test[0]:
            #                    Br2     Br   HBr    H2    H
            coef = np.array(
                [
                    [-1.0, 2.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 1.0, -1.0, 1.0],
                    [-1.0, 1.0, 1.0, 0.0, -1.0],
                    [0.0, -1.0, 1.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, -1.0, 2.0],
                ],
                dtype=float,
            )
            species = ["Br2", "Br", "HBr", "H2", "H"]
            num_rxn = 5
            num_mol = 5
            solution.file_prefix = os.path.join(solution.file_prefix, "Bromine Puzzle")
            solution_file_name = "Br2"

        elif test[1]:
            #                       O*   O2    O3    OH*   OOH   HOOH  H2O    H2   H*
            coef = np.array(
                [
                    [2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0],
                    [1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0],
                    [1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0],
                    [0.0, -1.0, 0.0, 2.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0],
                ],
                dtype=float,
            )
            species = ["O", "O2", "O3", "OH", "OOH", "HOOH", "H2O", "H2", "H"]
            num_rxn = 11
            num_mol = 9
            solution.file_prefix = os.path.join(solution.file_prefix, "Oxygen Puzzle")
            solution_file_name = "O2"
        else:
            pass

        test_solution = solution(num_rxn, num_mol, species, coef)
        test_solution.store(solution_file_name)
        new_solution = solution.load_object(solution_file_name)
        if test_solution == new_solution:
            print("The solution was successfully stored and loaded, no data was lost")
            return True
        else:
            print(
                "There was an error storing and loading the solution, the objects are not identical"
            )
            print(test_solution.__dict__, new_solution.__dict__, sep="\n\n")
            return False

    @classmethod
    def default_object(cls):
        return cls(1, 1, ["H"], np.array([[0.0]]))

    def __init__(self, *args, **kw_args):
        # parent class
        super().__init__(*args, **kw_args)

    # this method  is invoked when str() is called on an solution object
    def __str__(self):
        return "This is an instance of a solution object"

    def remove_reaction(self, array_of_rxn_to_remove):
        if self.number_of_reactions > len(array_of_rxn_to_remove):
            self.number_of_reactions -= len(array_of_rxn_to_remove)
            # array_of_rxn_indicies = [ (x-1) for x in array_of_rxn_to_remove ]
            # print(array_of_rxn_to_remove, file=sys.stderr)
            # subtract one so that the rxn# = array indices
            # print(self.coefficient_array, self.number_of_reactions, file=sys.stderr)
            self.coefficient_array = np.delete(
                self.coefficient_array, array_of_rxn_to_remove, 0
            )
            # print(self.coefficient_array, self.number_of_reactions, file=sys.stderr)
            self.update()
            print("Removed rxn" + str(array_of_rxn_to_remove), file=sys.stderr)
        else:
            print("We cannot remove the only rxn we have", file=sys.stderr)
            sys.exit(0)

    def add_reaction(self, array_of_rxn_to_add=None):
        self.number_of_reactions += 1
        input_array = (
            np.zeros(self.coefficient_array.shape[1])
            if array_of_rxn_to_add is None
            else array_of_rxn_to_add
        )
        # we insert at a new reaction at the begining of the array
        # axis of 0 means we insert along the 'reaction axis'
        # obj of 0 means we insert at the begining of that axis
        self.coefficient_array = np.insert(
            self.coefficient_array, obj=0, values=input_array, axis=0
        )
        self.update()
        print("Added rxn" + str(input_array), file=sys.stderr)


if __name__ == "__main__":
    solution.test()
