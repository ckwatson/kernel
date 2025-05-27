#
# driver.py
#
"""Solve the equilibrium ode"""

# These modules are used to parse arguments
import os
import logging
from typing import Optional
from tabulate import tabulate
from . import experiment_class
from . import handy_functions as HANDY
import numpy as np

from ..data import (
    solution_class,
    condition_class,
    reaction_mechanism_class,
    puzzle_class,
)

diag = os.path.join("Diagnose" + os.sep)
graph = os.path.join("Graphs" + os.sep)


# for now we use 'rxn_mechanism' since puzzle and solution can both be used here
# input_model is an instance of class experiment
# this function executes the necessary mathematical operations to find the Keq array - i.e. equilibrates the reaction


def equilibrate(job_id: str, input_model: experiment_class.Experiment, diag=False):
    # the meat and potatoes
    input_model.find_rate_constant()
    input_model.find_reaction_rate_function()
    input_model.find_reaction_profile(job_id, diagnostic_output=diag)
    input_model.find_experimental_Keq_array()
    input_model.find_flat_region(job_id=job_id, remove=True)


def run_true_experiment(
    job_id: str,
    puzzle: puzzle_class.puzzle,
    condition: condition_class.Condition,
    diag=False,
) -> np.ndarray:
    logger = logging.getLogger(job_id).getChild("run_true_experiment")
    logger.info("            First, pre-equilibrate every reagent:")
    for reagent_name, mechanism in puzzle.reagent_dict.items():
        preequilibrate_reagent(job_id, mechanism, condition, reagent_name, diag=diag)

    # now that we have left the loop over REAGENTS we have pre-equilibrated all necessary REAGENTS
    logger.info(
        "            Pre-equilibration finished -- now all reagents are under pre-equilibrium."
    )
    # this temp array seems to be due to a change in the parameter specs for the experiment objects instantiation method
    # basically a quick hack that needs to be factorized/cleaned up

    sorted_species_names = sorted(
        puzzle.coefficient_dict, key=puzzle.coefficient_dict.get
    )
    starting_concentrations = [
        condition.molecule_concentrations[name] for name in sorted_species_names
    ]

    # this is the experiment object that represents the "actual reaction"
    true_model = experiment_class.Experiment(
        puzzle,
        condition.reaction_temperature,
        input_time=[0.0],
        rxn_profile=np.array([starting_concentrations]),
    )
    # now we perform the same mathematical operations as before, but this time we have all the molecules present instead of isolated reactions
    logger.info(
        "            Now we can finally let the actual reaction happen -- let's pour everything into the beaker:"
    )
    equilibrate(job_id, true_model, diag=diag)  # the magical math happens

    data = [
        ["Starting Concentrations (mol)"] + starting_concentrations,
        ["Reactant Rate Constants"] + true_model.reactant_rate_constants.tolist(),
        ["Product Rate Constants"] + true_model.product_rate_constants.tolist(),
    ]
    if true_model.theoretical_Keq_array is not None:
        data.append(["Theoretical K_eq"] + true_model.theoretical_Keq_array.tolist())
    if true_model.experimental_Keq_array is not None:
        data.append(["Experimental K_eq"] + true_model.experimental_Keq_array.tolist())
    table = tabulate(
        data,
        headers=sorted_species_names,
        floatfmt=".4g",
        tablefmt="github",
    )
    logger.info(
        '            True model successfully constructed: \n%s\n            For `K_eq`, use the "theoretical" value, since this is a source-of-truth experiment. (The "theoretical" and the "empirical" values should match, though.)',
        table,
    )
    # Return a 2D array with the first column being timestamps and the rest being the concentrations of each species.
    return np.transpose(
        np.column_stack([true_model.time_array, true_model.reaction_profile])
    )


def preequilibrate_reagent(
    job_id: str,
    mechanism: reaction_mechanism_class.reaction_mechanism,
    condition: condition_class.Condition,
    reagent_name: str,
    diag: bool = False,
) -> None:
    """
    Some reagents may dissociate into multiple species even when sitting ideally in a canister/beaker, so we need to
    pre-equilibrate them.

    For example, a canister/beaker that was filled with pure O2 would actually be comprised of some small amount
    of O and O2, as the O2 can dissociate on its own.

    This does not happen for all reagents, though. For example, H2 does not dissociate into H and H2.

    How do we know which reagents need to be pre-equilibrated? It's defined in the puzzle definition file.
    The field `reagentPERs` tells us, for each reagent in the puzzle, and for each elementary reaction in the puzzle,
    whether the reaction is would naturally happen in a canister/beaker of that reagent.

    For example, if the puzzle has 5 elementary reactions, and the reagent is C2H6, then this snippet:

    ```json
        "reagentPERs": {
        "C2H6": [
            true,
            false,
            false,
            false,
            false
        ]
    }
    ```

    says that the first elementary reaction would happen in a canister/beaker of C2H6, but the other 4 would not.

    This function updates the `molecule_concentrations` field of the `condition` object.
    """
    logger = logging.getLogger(job_id).getChild("preequilibrate_reagent")
    logger.info("                Pre-equilibrate " + reagent_name + ":")

    # Zero out all the condition objects molecule concentrations as a safety measure.
    # TODO: This should probably be made a function in the condition class at some later point.
    for name in mechanism.coefficient_dict.keys():
        condition.molecule_concentrations[name] = 0

    reagent_id = mechanism.coefficient_dict[reagent_name]
    user_specified_concentration = condition.reagent_concentrations[reagent_name]

    # If there is only one species no pre-equilibration is needed as no dissociation can occur.
    if mechanism.number_of_species == 1:
        logger.info(
            "                        This reagent only has one species so no pre-equilibration happened. "
            "This reagent will join the experiment with the user-specified concentration of %s mol.",
            user_specified_concentration,
        )
        # then directly add the reagents concentrations to their associated molecule concentrations
        condition.molecule_concentrations[reagent_name] += user_specified_concentration
    else:
        # Make the array that will hold the concentration values for each REAGENT, the "beakers" that the user can change
        # the concentration of.
        starting_concentrations = [0] * mechanism.number_of_species
        # we start with the concentrations defined by the user in the condition object they passed to us
        starting_concentrations[reagent_id] = user_specified_concentration

        # create the experiment object, this object handles all the necessary mathematical calculations to determine
        # how much of each species is created, consumed during dissociation/pre-equilibration
        pre_equil_model = experiment_class.Experiment(
            mechanism,
            condition.reagent_temperatures[reagent_name],
            rxn_profile=np.array([starting_concentrations]),
        )

        # actually preform the mathematical calculations
        # (diagnostics) diag is an optional argument that if true prints all the output from the integrator inside the experiment object
        # this is a lotttt of extra output
        equilibrate(job_id, pre_equil_model, diag=diag)
        final_concentrations = pre_equil_model.reaction_profile[-1]

        table = tabulate(
            [starting_concentrations, final_concentrations],
            headers=sorted(
                mechanism.coefficient_dict.keys(), key=mechanism.coefficient_dict.get
            ),
            floatfmt=".4g",
            tablefmt="github",
        )
        logger.info(
            "                        Concentrations before and after pre-equilibration: \n%s",
            table,
        )

        # Now that we have the pre-equilibrated concentrations, place them into the condition object.
        for name, index in mechanism.coefficient_dict.items():
            concentration_of_this_species = final_concentrations[index]
            if np.isnan(concentration_of_this_species):
                concentration_of_this_species = starting_concentrations[index]
                logger.warning(
                    '                 Concentration of "%s" is NaN. Falling back to un-pre-equilibrated concentration of %s. This is a bug.',
                    name,
                    concentration_of_this_species,
                )
            condition.molecule_concentrations[name] += concentration_of_this_species

    # collect the equilibrated concentrations
    table = tabulate(
        [condition.molecule_concentrations.values()],
        headers=condition.molecule_concentrations.keys(),
        floatfmt=".4g",
        tablefmt="github",
    )
    logger.info(
        "                    Updated concentrations of all reagents before the main experiment take place:\n%s",
        table,
    )


def run_proposed_experiment(
    job_id: str,
    condition: condition_class.Condition,
    solution: solution_class.solution,
    data: np.ndarray,
    diag=False,
) -> Optional[np.ndarray]:
    logger = logging.getLogger(job_id).getChild("run_proposed_experiment")

    proposed_model = experiment_class.Experiment(
        solution,
        condition.reaction_temperature,
        input_time=data[0],
        rxn_profile=data[1:].T,
    )
    try:
        # try to find the rate constants
        rate_constants = proposed_model.get_matrix_rate_solution(job_id=job_id)
        logger.debug("Rate Constants: %s", rate_constants)
    # try to handle bad rate constants
    except HANDY.NegativeCoefficientException as u_error:
        bad_rxn = np.flatnonzero(u_error.value)

        # if more than one reaction has forward and backward rate constants of negative value then crash
        if bad_rxn.size > 1:

            logger.info(
                "An issue has been detected. Reactions "
                + str(bad_rxn + 1)
                + " are unstable.\n Cannot proceed with user reaction, input new reaction."
            )
            return None
        # if only one reaction is 'bad' then try to remove it and solve again
        elif bad_rxn.size == 1:

            logger.info(
                "An issue has been detected. Reaction "
                + str(bad_rxn[0] + 1)
                + " is unstable. \n Trying to correct simulation by removing reaction."
            )
            try:
                proposed_model.remove_rxn(bad_rxn[0])

                # try to find the rate constants again
                rate_constants = proposed_model.get_matrix_rate_solution(job_id=job_id)

            # if we fail again then crash
            except HANDY.NegativeCoefficientException:
                logger.error(
                    "Another issue has been detected. Reaction "
                    + str(bad_rxn[0] + 1)
                    + " is unstable. \n Cannot proceed, crashing."
                )
                return None

        # completed try successfully

    # find the reaction rate
    proposed_model.find_reaction_rate_function()

    # calculate the new reaction profile
    sorted_species_names = sorted(
        solution.coefficient_dict.keys(), key=solution.coefficient_dict.get
    )
    input_concentrations = [
        condition.molecule_concentrations[name] for name in sorted_species_names
    ]
    proposed_model.find_reaction_profile(
        job_id=job_id,
        input_concentration=np.array(input_concentrations),
        diagnostic_output=diag,
    )
    proposed_model.find_flat_region(job_id=job_id, remove=True)
    data = [
        ["Starting Concentrations (mol)"] + input_concentrations,
        ["Reactant Rate Constants"]
        + proposed_model.reactant_rate_constants.tolist(),
        ["Product Rate Constants"] + proposed_model.product_rate_constants.tolist(),
    ]
    if proposed_model.theoretical_Keq_array is not None:
        data.append(
            ["Theoretical K_eq"] + proposed_model.theoretical_Keq_array.tolist()
        )
    table = tabulate(
        data,
        headers=sorted_species_names,
        floatfmt=".4g",
        tablefmt="github",
    )
    logger.info(
        '            User-proposed model successfully constructed: \n%s\n            For `K_eq`, the "theoretical" value is not available, since we are pretending to be conducting this experiment in a laboratory and measuring things live.',
        table,
    )
    return np.transpose(
        np.column_stack([proposed_model.time_array, proposed_model.reaction_profile])
    )
