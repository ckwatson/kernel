#
# condition_class.py
#
"""module for the class condition"""

import os
import sys
import numpy as np
from . import json_base_class


class Condition(json_base_class.Json_base):
    """The condition class"""

    # default file properties
    file_prefix = os.path.join(os.getcwd(), "Puzzles")
    file_suffix = ".cond"

    @staticmethod
    def test():
        number_of_tests = 2
        test = [False] * number_of_tests
        # run this test
        test[0] = True

        if test[0]:
            condition_file_name = "Br2"
            rxn_temp = float(200.0)
            r_list = ["Br2", "H2"]
            r_temps = [25.0, 35.0]
            r_concs = [1.0, 4.0]
            species = ["Br2", "Br", "HBr", "H2", "H"]
            m_concs = [0.0, 0.0, 0.0, 0.0, 0.0]
            #                = ['Br2',  'Br'  , 'HBr' , 'H2'  , 'H']
            # conc            = [ 1.0 ,   0.0  ,  0.0  ,  4.0  , 0.0]

        elif test[1]:
            condition_file_name = "O2"
            rxn_temp = float(200.0)
            r_list = ["O2", "H2"]
            r_temps = [25.0, 35.0]
            r_concs = [10.0, 20.0]
            species = ["O", "O2", "O3", "OH", "OOH", "HOOH", "H2O", "H2", "H"]
            m_concs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            #                = [  'O',  'O2',  'O3',  'OH',   'OOH',  'HOOH',  'H2O',  'H2',    'H']
            # conc            = [  0.0,  10.0,   0.0,   0.0,    0.0,     0.0,    0.0,   20.0,    0.0]

        test_condition = Condition(rxn_temp, species, r_list, r_temps, r_concs, m_concs)
        test_condition.store(condition_file_name)
        new_condition = Condition.load_object(condition_file_name)
        if test_condition == new_condition:
            print("The condition was successfully stored and loaded, no data was lost")
            return True
        else:
            print(
                "There was an error storing and loading the condition, the objects are not identical"
            )
            print(
                test_condition.__dict__, new_condition.__dict__, sep="\n\n\n\n\n\n\n\n"
            )
            return False

    @classmethod
    def default_object(cls):
        return cls(0.0, ["H"], ["H"], [0.0], [0.0], [0.0])

    def __init__(
        self,
        rxn_temp,
        species,
        reagents,
        reagent_temp_list,
        reagent_conc_list,
        molecule_conc_list,
        *args,
        **kw_args
    ):
        self.reaction_temperature = rxn_temp
        self.reagent_temperatures = dict(zip(reagents, reagent_temp_list))
        self.reagent_concentrations = dict(zip(reagents, reagent_conc_list))
        self.molecule_concentrations = dict(zip(species, molecule_conc_list))
        super().__init__(*args, **kw_args)

    def __eq__(self, other):
        # this isn't robust enough
        return (
            isinstance(other, Condition)
            and self.reaction_temperature == other.reaction_temperature
            and self.reagent_temperatures == other.reagent_temperatures
            and self.reagent_concentrations == other.reagent_concentrations
            and self.molecule_concentrations == other.molecule_concentrations
            and super().__eq__(other)
        )

    # this method is invoked when str() is called on a condition object
    def __str__(self):
        return "This is an instance of a condition object"

    # this function is overloaded by subclasses
    def prepare_store(self):
        prepared_dict = {
            "reaction_temperature": self.reaction_temperature,
            "reagent_temperatures": self.reagent_temperatures,
            "reagent_concentrations": self.reagent_concentrations,
            "molecule_concentrations": self.molecule_concentrations,
        }
        return prepared_dict

    # this function is overloaded by subclasses
    def prepare_load(self, loaded_dict={}):
        # this should not be needed
        # loaded_dict['reaction_temperature'] = float(loaded_dict['reaction_temperature'])
        # loaded_dict['reagent_temperatures'].update((key, float(value)) for key, value in loaded_dict['reagent_temperatures'].items())
        # loaded_dict['reagent_concentrations'].update((key, float(value)) for key, value in loaded_dict['reagent_concentrations'].items())
        # loaded_dict['molecule_concentrations'].update((key, float(value)) for key, value in loaded_dict['molecule_concentrations'].items())
        return loaded_dict

    def update(self):
        pass


if __name__ == "__main__":
    Condition.test()
