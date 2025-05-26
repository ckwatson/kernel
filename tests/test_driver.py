from kernel.engine.driver import preequilibrate_reagent
from kernel.data.condition_class import Condition
from kernel.data.reaction_mechanism_class import reaction_mechanism
import numpy as np


def test_preequilibrate():
    mechanism = reaction_mechanism(
        num_rxn=1,
        num_species=2,
        mol_list=["O2", "O"],
        rxn_array=np.array([[-1, 2]]),
        reagEnrg={"O2": 1, "O": 0},
    )
    condition = Condition(
        rxn_temp=300,
        species=["O2", "O"],
        reagents=["O2"],
        reagent_temp_list=[300],
        reagent_conc_list=[1.0],
        molecule_conc_list=[0, 0],
    )
    preequilibrate_reagent("test_job", mechanism, condition, "O2", diag=False)
    expected_concentrations = {"O2": 0.32, "O": 1.36}
    for species, expected in expected_concentrations.items():
        assert np.isclose(
            condition.molecule_concentrations[species], expected, atol=1e-2
        )
