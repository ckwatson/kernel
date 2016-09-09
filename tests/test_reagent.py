from ..data.molecular_species import molecular_species
from ..data.reagent           import reagent
import numpy as np
import pytest

def test_load_store():
    test_coef = np.array([  [1.0, 1.0, -1.0],[-1.0, 2.0, -1.0]], dtype=float)
    test_molecule_one   = molecular_species('H',  10.0, {'H' : 1, 'O' : 0})
    test_molecule_two   = molecular_species('H2', 0.0,  {'H' : 2, 'O' : 0})
    test_molecule_three = molecular_species('H3', 30.0, {'H' : 3, 'O' : 0})
    test_molecule_list = [test_molecule_one, test_molecule_two, test_molecule_three]
    test_reagent = reagent('H3', 15.0, 2, 3, test_molecule_list, test_coef)
    print(test_reagent.name + '.reagent')
    test_reagent.store(test_reagent.name + '.reagent')
    new_reagent = reagent.load_reagent(test_reagent.name + '.reagent')
    assert(test_reagent == new_reagent)