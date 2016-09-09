from ..data.molecular_species import molecular_species
import pytest

def test_load_store():
    test_molecule = molecular_species('H2', 0.0, {'H' : 2, 'O' : 0})
    test_molecule.store(test_molecule.name + 'molecule')
    new_molecule = molecular_species.load_molecular_species(test_molecule.name + 'molecule')
    assert(test_molecule == new_molecule)