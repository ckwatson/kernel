from kernel.engine.driver import preequilibrate_reagent, run_proposed_experiment
from kernel.data.condition_class import Condition
from kernel.data.reaction_mechanism_class import reaction_mechanism
from kernel.data.solution_class import solution
import numpy as np


def make_simple_solution():
    # Provide required species energies for the reaction mechanism
    species = ["A", "B"]
    coef = np.array([[-1.0, 1.0]])
    # Use non-equal energies to avoid zero delta_energy_array
    reag_enrg = {"A": 0.0, "B": 1000.0}  # ensure nonzero energy difference
    num_rxn = 1
    num_species = 2
    return solution(num_rxn, num_species, species, coef, reag_enrg)


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


def test_run_proposed_experiment_normal_case():
    sol = make_simple_solution()
    rxn_temp = 300.0
    reagents = ["A"]
    reagent_temps = [300.0]
    reagent_concs = [1.0]
    molecule_concs = [1.0, 0.0]
    cond = Condition(
        rxn_temp,
        sol.get_name_set(),
        reagents,
        reagent_temps,
        reagent_concs,
        molecule_concs,
    )
    # Use a time range and enough points, and ensure a flat region at the end
    times = np.linspace(0, 12, 20)
    concentrations = np.zeros((20, 2))
    # First 15 points: linear change, last 5 points: flat (simulate equilibrium)
    concentrations[:15, 0] = np.linspace(1.0, 0.3, 15)  # [A] decreases
    concentrations[:15, 1] = np.linspace(0.0, 0.7, 15)  # [B] increases
    concentrations[15:, 0] = 0.3  # [A] flat
    concentrations[15:, 1] = 0.7  # [B] flat
    # Add a small noise to avoid flat region detection issues
    rng = np.random.default_rng(42)
    concentrations += rng.normal(0, 1e-4, concentrations.shape)
    data = np.vstack([times, concentrations.T])
    result = run_proposed_experiment("test_job", cond, sol, data, diag=False)
    assert result is not None
    # Check output shape: (n_species+1, n_timepoints)
    assert result.shape[0] == 3 and result.shape[1] >= 10
    # Check that concentrations are non-negative and monotonic (A decreases, B increases)
    a_conc = result[1]
    b_conc = result[2]
    assert np.all(a_conc >= 0) and np.all(b_conc >= 0)
    assert a_conc[0] > a_conc[-1]
    assert b_conc[0] < b_conc[-1]
