import numpy as np
from deepdiff import DeepDiff

from ..data.molecular_species import molecular_species
from ..data.reagent import reagent


def test_init():
    test_reagent = reagent(
        "H3",
        15.0,
        # The parameters below are for the class `reaction_mechanism`.
        num_rxn=2,
        num_species=3,
        mol_list=["H", "H2", "H3"],
        rxn_array=np.array([[1.0, 1.0, -1.0], [-1.0, 2.0, -1.0]], dtype=float),
        reagEnrg={"H3": 30.0, "H2": 0.0, "H": 10.0},
    )
    got = test_reagent.__dict__.copy()
    del got["JSON_Encoder"]

    want = {
        "name": "H3",
        "concentration": 15.0,
        "number_of_reactions": 2,
        "number_of_species": 3,
        "molecular_species_dict": {
            # TODO: Are we sure we don't need atom_list?
            "H": molecular_species("H", 10.0),  # , {'H': 1, 'O': 0}),
            "H2": molecular_species("H2", 0.0),  # , {'H': 2, 'O': 0}),
            "H3": molecular_species("H3", 30.0),  # , {'H': 3, 'O': 0})
        },
        "coefficient_dict": {"H": 0, "H2": 1, "H3": 2},
        "coefficient_array": np.array([[1.0, 1.0, -1.0], [-1.0, 2.0, -1.0]]),
        "reactant_coefficient_array": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        "product_coefficient_array": np.array([[1.0, 1.0, 0.0], [0.0, 2.0, 0.0]]),
    }
    assert not DeepDiff(got, want, ignore_order=True)
