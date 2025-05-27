#
# reaction_mechanism_class.py
#
"""module for the class reagent"""

import os
from typing import Optional, List

import numpy as np

# imports from the ckwatson package
from . import json_base_class
from . import molecular_species


class reaction_mechanism(json_base_class.Json_base):
    """The parent class for reagent/puzzle/solution"""

    # default file properties
    # file_prefix = os.path.join(os.getcwd(), 'Folder1', 'Folder2', etc....)
    file_prefix = os.path.join(os.getcwd(), "Data")
    file_suffix = ".rxn"

    # TODO: Not initialized?
    transition_state_energies: Optional[List[str]] = None

    @staticmethod
    def test():
        number_of_tests = 3
        test = [False] * number_of_tests
        # run this test
        test[0] = True

        if test[0]:
            name = "O2"
            n_rxn = 2
            n_species = 3
            mol_list = ["O", "O2", "O3"]
            rxn_array = np.array(
                [
                    [2.0, -1.0, 0.0],
                    [-1.0, -1.0, 1.0],
                ],
                dtype=float,
            )
        elif test[1]:
            name = "Br2"
            n_rxn = 1
            n_species = 2
            mol_list = ["Br2", "Br"]
            rxn_array = np.array(
                [
                    [-1.0, 2.0],
                ],
                dtype=float,
            )
        elif test[2]:
            name = "H2"
            n_rxn = 1
            n_species = 2
            mol_list = ["H2", "H"]
            rxn_array = np.array(
                [
                    [-1.0, 2.0],
                ],
                dtype=float,
            )

        test_rxn = reaction_mechanism(n_rxn, n_species, mol_list, rxn_array)
        test_rxn.store(name)
        new_rxn = reaction_mechanism.load_object(name)
        if test_rxn == new_rxn:
            print(
                "The reaction_mechanism was successfully stored and loaded, no data was lost"
            )
            return True
        else:
            print(
                "There was an error storing and loading the reaction_mechanism, the objects are not identical"
            )
            print(test_rxn.__dict__, new_rxn.__dict__, sep="\n\n\n\n\n\n\n\n")
            return False

    @classmethod
    def default_object(cls):
        return cls(1, 1, ["H"], np.array([[0.0]]))

    def __init__(
        self, num_rxn, num_species, mol_list, rxn_array, reagEnrg, *args, **kw_args
    ):
        # handle molecule JSON serialization properly
        self.JSON_Encoder = molecular_species.mol_JSON_Encoder

        self.number_of_reactions = int(
            num_rxn
        )  # number of columns in self.coefficient_array
        self.number_of_species = int(
            num_species
        )  # number of rows in self.coefficient_array ALSO the length of self.molecular_species_dict
        # it is very important to align the order of the species list here
        # the dictionary retains this order and successful integration depends on this ordering matching with the ordering of the coeff array
        self.molecular_species_dict = dict(
            [
                (
                    species_name,
                    molecular_species.molecular_species(
                        species_name, reagEnrg[species_name]
                    ),
                )
                for species_name in mol_list
            ]
        )
        # the r, p, v matrices
        self.coefficient_dict = dict(zip(mol_list, range(len(mol_list))))
        self.coefficient_array = np.array(rxn_array, dtype=float)
        self.reactant_coefficient_array = None
        self.product_coefficient_array = None
        super().__init__(*args, **kw_args)

    def __eq__(self, other):
        return (
            isinstance(other, reaction_mechanism)
            and self.number_of_reactions == other.number_of_reactions
            and self.number_of_species == other.number_of_species
            and np.all(self.molecular_species_dict == other.molecular_species_dict)
            and np.all(self.coefficient_dict == other.coefficient_dict)
            and np.all(self.coefficient_array == other.coefficient_array)
            and np.all(
                self.reactant_coefficient_array == other.reactant_coefficient_array
            )
            and np.all(
                self.product_coefficient_array == other.product_coefficient_array
            )
            and super().__eq__(other)
        )

    # this method  is invoked when str() is called on an reaction_mechanism object
    def __str__(self):
        return "This is an instance of a reaction_mechanism object"

    def __repr__(self):
        return str(self.prepare_store())

    def get_name_set(self):
        temp_array = [None] * self.number_of_species
        for name, item in self.coefficient_dict.items():
            temp_array[item] = name
        return temp_array

    def get_energy_set(self):
        temp_array = [None] * self.number_of_species
        for name, item in self.coefficient_dict.items():
            temp_array[item] = self.molecular_species_dict[name].energy
        return temp_array

    # this function is overloaded by subclasses
    def prepare_store(self):
        prepared_dict = {
            "number_of_reactions": self.number_of_reactions,
            "number_of_species": self.number_of_species,
            "coefficient_dict": self.coefficient_dict,
            "coefficient_array": self.coefficient_array.tolist(),
        }
        return prepared_dict

    # this function is overloaded by subclasses
    def prepare_load(self, loaded_dict={}):

        # print("The dict:", dict([(x, 5) for x in loaded_dict['coefficient_dict'].keys()]), file=sys.stderr)
        loaded_dict["molecular_species_dict"] = dict(
            [
                (
                    species_name,
                    molecular_species.molecular_species.load_object(species_name),
                )
                for species_name in loaded_dict["coefficient_dict"].keys()
            ]
        )
        loaded_dict["coefficient_array"] = np.array(
            loaded_dict["coefficient_array"], dtype=float
        )
        return loaded_dict

    def update(self):
        # check to make sure we didn't end in some weird case where violated the design requirements
        """
        print("Num species", self.number_of_species, '\n',
            "mol dict:", len(self.molecular_species_dict), self.molecular_species_dict,  '\n',
            "coef dict:", len(self.coefficient_dict), self.coefficient_dict,  '\n',
            '\n')
        """
        assert (
            self.number_of_species
            == len(self.molecular_species_dict)
            == len(self.coefficient_dict)
        )
        assert self.number_of_reactions == self.coefficient_array.shape[0]
        assert self.number_of_species == self.coefficient_array.shape[1]
        # do the work
        self.reactant_coefficient_array = np.select(
            [self.coefficient_array < 0.0, True], [np.fabs(self.coefficient_array), 0.0]
        )
        self.product_coefficient_array = np.select(
            [self.coefficient_array > 0.0, True], [self.coefficient_array, 0.0]
        )


if __name__ == "__main__":
    reaction_mechanism.test()


# spacer
