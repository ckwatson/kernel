#
# reagent.py
#
"""module for the class reagent"""

import os
import numpy as np
from . import reaction_mechanism_class

class reagent(reaction_mechanism_class.reaction_mechanism):
    """The reagent class"""
    # default file properties
    file_prefix = os.path.join(os.getcwd(), 'Data', 'Reagents')
    file_suffix = '.reag'

    @staticmethod
    def test():
        number_of_tests = 3
        test = [ False ] * number_of_tests
        # run this test
        test[2] = True

        if(test[0]):
            name        = "O2"
            conc        = 0.0
            n_rxn       = 2
            n_species   = 3
            mol_list    =              [ 'O',  'O2', 'O3' ]
            rxn_array   = np.array([   [  2.0, -1.0,  0.0 ], 
                                       [ -1.0, -1.0,  1.0 ],
                                   ], dtype = float)
        elif(test[1]):
            name        = "Br2"
            conc        = 0.0
            n_rxn       = 1
            n_species   = 2
            mol_list    =              [ 'Br2', 'Br']
            rxn_array   = np.array([   [ -1.0,  2.0 ],
                                   ], dtype = float)
        elif(test[2]):
            name        = "H2"
            conc        = 0.0
            n_rxn       = 1
            n_species   = 2
            mol_list    =              [ 'H2',  'H' ]
            rxn_array   = np.array([   [ -1.0,  2.0 ],
                                   ], dtype = float)
        else:
            pass    

        test_reagent = reagent(name, conc, n_rxn, n_species, mol_list, rxn_array)
        test_reagent.store(name)
        new_reagent = reagent.load_object(name)
        if(test_reagent == new_reagent):
            print("The reagent was successfully stored and loaded, no data was lost")
            #print(test_reagent.__dict__, new_reagent.__dict__, sep='\n\n')
            return True
        else:
            print("There was an error storing and loading the reagent, the objects are not identical")
            #print(test_reagent.__dict__, new_reagent.__dict__, sep='\n\n')
            return False

    @classmethod
    def default_object(cls):
        return cls("default_reagent", 0.0, 1, 1, ['H'], np.array([[0.0]]))

    def __init__(self, name, concentration, *args, **kw_args):
        # identifying informaiton
        self.name                           = str(name)
        self.concentration                  = float(concentration)

        # parent class
        super().__init__(*args, **kw_args)


    def __eq__(self, other):
        return (isinstance(other, reagent) 
            and self.name                                  == other.name                        
            and self.concentration                         == other.concentration               
            and super().__eq__(other)
            )

    # this method  is invoked when str() is called on an reagent object 
    def __str__(self):
        return "This is an instance of a reagent object"

    # this function is overrided by any subclasses
    def prepare_store(self):
        prepared_dict = {   'name'              : self.name, 
                            'concentration'     : self.concentration,
                        }
        prepared_dict.update(super().prepare_store())
        return prepared_dict

    def prepare_load(self,  loaded_dict = {}):
        # if we need to do something special
        loaded_dict.update(super().prepare_load(loaded_dict))
        return loaded_dict


if(__name__ == "__main__"):
    self.test()
