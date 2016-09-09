#
# puzzle_class.py
#
"""module for the class puzzle"""

import os
import json
import numpy as np
import warnings
from collections import OrderedDict  # ordered dictionary requires python 3.3
# imports from the ckwatson package
from . import reaction_mechanism_class


class puzzle(reaction_mechanism_class.reaction_mechanism):
    """The puzzle class"""
    # default file properties
    file_prefix = os.path.join(os.getcwd(), 'Puzzles',)
    file_suffix = '.puz'

    @staticmethod
    def test():
        number_of_tests = 2
        test = [ False ] * number_of_tests

        # run this test
        # only one test should be true at once, this is a quick hack
        test[0] = True

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------
        if(test[0]):
            #                    Br2     Br   HBr    H2    H
            coef = np.array([   [-1.0,  2.0,  0.0,  0.0,  0.0], 
                                [ 0.0, -1.0,  1.0, -1.0,  1.0], 
                                [-1.0,  1.0,  1.0,  0.0, -1.0], 
                                [ 0.0, -1.0,  1.0,  0.0, -1.0], 
                                [ 0.0,  0.0,  0.0, -1.0,  2.0],
                            ], dtype = float)

            Ea       = None
            species  =          ['Br2' ,  'Br'  , 'HBr' , 'H2'  , 'H'] 
            reagents =          ['Br2', 'H2']
            num_rxn  = 5
            num_mol  = 5  
            puzzle.file_prefix = os.path.join(puzzle.file_prefix, 'Bromine Puzzle')
            puzzle_file_name = 'Br2'

            # the reagents for this puzzle
            reagent_name            = "Br2"
            reagent_num_rxn         = 1
            reagent_num_species     = 2
            reagent_mol_list        =              [ 'Br2', 'Br']
            reagent_rxn_array       = np.array([   [ -1.0,  2.0 ],  
                                                ], dtype = float)
            reagent_dictionary =  [ (reagent_name, reaction_mechanism_class.reaction_mechanism(reagent_num_rxn, reagent_num_species, reagent_mol_list, reagent_rxn_array)) ]
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------

        elif(test[1]): 
            #                       O*   O2    O3    OH*   OOH   HOOH  H2O    H2    H
            coef = np.array([   [  2.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], 
                                [ -1.0, -1.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], 
                                [ -1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0, -1.0,  1.0], 
                                [  1.0, -1.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0, -1.0], 
                                [  0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  1.0, -1.0,  1.0], 
                                [  1.0, -1.0,  0.0, -1.0,  1.0,  0.0,  0.0,  0.0,  0.0],
                                [  0.0,  0.0,  0.0, -1.0,  0.0,  0.0,  1.0,  0.0, -1.0],
                                [  0.0,  0.0,  0.0, -2.0,  0.0,  1.0,  0.0,  0.0,  0.0],
                                [  0.0,  0.0,  0.0,  0.0, -1.0,  1.0,  0.0,  0.0, -1.0],
                                [  0.0, -1.0,  0.0,  2.0,  0.0,  0.0,  0.0, -1.0,  0.0],
                                [  0.0,  0.0,  0.0,  0.0, -1.0,  1.0,  0.0, -1.0,  1.0]
                            ], dtype = float)

            Ea       = np.array(["high", "high", "high", "low", "high", "low", "high", "low", "medium", "medium", "high"])
            species  =          [  'O',  'O2',  'O3',  'OH',   'OOH',  'HOOH',  'H2O',   'H2',    'H']
            reagents =          ['O2', 'H2']
            num_rxn  = 11
            num_mol  = 9
            puzzle.file_prefix = os.path.join(puzzle.file_prefix, 'Oxygen Puzzle')
            puzzle_file_name = 'O2'

            # the reagent for this puzzle
            reagent_name            = "O2"
            reagent_num_rxn         = 2
            reagent_num_species     = 3
            reagent_mol_list        =               [ 'O',  'O2', 'O3' ]
            reagent_rxn_array       = np.array([    [  2.0, -1.0,  0.0 ], 
                                                    [ -1.0, -1.0,  1.0 ],
                                                ], dtype = float)
            reagent_dictionary =  [ (reagent_name, reaction_mechanism_class.reaction_mechanism(reagent_num_rxn, reagent_num_species, reagent_mol_list, reagent_rxn_array)) ]
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------
        else:
            pass

        test_puzzle = puzzle(reagent_dictionary, num_rxn, num_mol, species, coef, Ea = Ea)
        test_puzzle.store(puzzle_file_name)
        new_puzzle = puzzle.load_object(puzzle_file_name)
        if(test_puzzle == new_puzzle):
            print("The puzzle was successfully stored and loaded, no data was lost")
            return True
        else:
            print('\n\n\n\n\n\n\n')
            print("There was an error storing and loading the puzzle, the objects are not identical")
            print(test_puzzle.__dict__, new_puzzle.__dict__, sep='\n\n\n\n\n\n\n\n')
            return False

    @classmethod
    def default_object(cls):
        return cls([], 1, 1, ['H'], np.array([[0.0]]), Ea = None)

    def __init__(self, *args, reagent_dictionary, Ea = None, **kw_args):
        # required puzzle variables
        self.reagent_dict              = OrderedDict(reagent_dictionary)
        # a note here, I believe the reason for using Ordered Dictionaries was that dictionaries are not hashable and so simple comparisons do not work
        # by forcing them to be ordered we can use __eq__ to compare them 
        # this does seem to be the case as the == fails when stored and then loaded from JSON file
        # there may be a more elegent way to handle this, possibly making another JSON handler for each object, but this works for now
        self.have_transition_energies   = None
        self.transition_state_energies  = Ea
        #this calls the reaction_mechanism to initialze, using the arguments left unused by this puzzle class.
        super().__init__(*args, **kw_args) 
    def __eq__(self, other):
        # useful for comparing all parameter values
        #for (key_a, value_a), (key_b, value_b) in zip(type(self).__dict__.items(), type(other).__dict__.items()):
        #    print(key_a, value_a, value_b, '\n', value_a == value_b, sep = '   ')
        return (isinstance(other, puzzle) 
            and self.reagent_dict                       == other.reagent_dict
            and self.have_transition_energies           == other.have_transition_energies
            and np.all(self.transition_state_energies   == other.transition_state_energies)
            and super().__eq__(other)
            )

    # this method  is invoked when str() is called on an experiment_class object 
    def __str__(self):
        return "This is an instance of a puzzle object"

    def get_name_set(self):
        # the parent class gets our own molecular specices name list
        set_of_names = super().get_name_set()
        #for reagent_object in self.reagent_dict:
        #    set_of_names += reagent_object.get_name_set()

        # make our list unique
        #unique_list = []
        #[unique_list.append(item) for item in set_of_names if item not in unique_list]
        #print(unique_list)
        #return unique_list
        return set_of_names

    def remove_reaction(self, array_of_rxn_to_remove):
        if self.number_of_reactions > (len(array_of_rxn_to_remove) + 1): 
            self.number_of_reactions -= len(array_of_rxn_to_remove)
            array_of_rxn_indicies = [ x-1 for x in array_of_rxn_to_remove ]
            # subtract one so that the rxn# = array indices
            self.coefficient_array = np.delete(self.coefficient_array, array_of_rxn_indicies, 0)
            self.update()
            print("removed rxn" + str(array_of_rxn_to_remove))
        else:
            print("We cannot remove the only rxn we have")
            sys.exit(0)

    def add_reaction(self):
        self.update()
        print("Added rxn" + str(array_of_rxn_to_add))

    def prepare_store(self):
        prepared_dict = {   'reagent_dict'                  : { reagent_name : reagent_object.prepare_store() for reagent_name, reagent_object in self.reagent_dict.items() },
                            'transition_state_energies'     : self.transition_state_energies.tolist() if self.have_transition_energies else None,
                        }
        prepared_dict.update(super().prepare_store())
        return prepared_dict

    def prepare_load(self,  loaded_dict = {}):
        loaded_dict['reagent_dict']              = OrderedDict([ (reagent_name, reaction_mechanism_class.reaction_mechanism.load_object("placeholder", json_dictionary=reagent_json_string)) for reagent_name, reagent_json_string in loaded_dict['reagent_dict'].items() ])
        loaded_dict['transition_state_energies'] = np.array(loaded_dict['transition_state_energies']) if loaded_dict['transition_state_energies'] is not None else None
        loaded_dict.update(super().prepare_load(loaded_dict))
        return loaded_dict

    def update(self):
        super().update() # let the parent class do the regular work
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.have_transition_energies = True if (self.transition_state_energies != None) else False

if(__name__ == "__main__"):
    puzzle.test()


#end 
