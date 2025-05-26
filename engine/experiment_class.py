#
# experiment_class.py
#
"""module for the class experiment"""

import sys, logging
from typing import Optional

import numpy as np
import numpy.ma as ma
from tabulate import tabulate

from kernel.data.reaction_mechanism_class import reaction_mechanism
from . import handy_functions as HANDY

# data file classes

# constants we import to help with integration
from scipy.constants import R  # 8.3144621 J / ( mol * K )
from scipy.integrate import (
    odeint,
)  # solve differential rate equation, we want flat concentrations at long [http://docs.scipy.org/doc/scipy/reference/integrate.html] (the doc for integrate)

# suppress tiny numbers, numbers appear as zero if < 1E-12
np.set_printoptions(suppress=True)


class Experiment:
    """
    Represents a chemical reaction or experiment, simulating the mixing of beakers/canisters or the spontaneous
    reactions in one beaker/canister without mixing.
    """

    stream = sys.stdout
    # ODE solver parameters
    abserr = 1.0e-8  # see documentation http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
    relerr = 1.0e-6  # see documentation ^ same as above
    step_size = 300  # see documentation ^ same as above
    num_steps = 100000  # see documentation ^ same as above
    # this last parameters is class specific
    max_Keq_steps = 8  # max number of steps we take to try and find a better Keq

    # constant values
    SCALING_FACTOR = 1000.0 * R  # units are
    ENERGY_BARRIER_DICTIONARY = {"low": 1000.0, "medium": 3000.0, "high": 5000.0}
    TEMPERATURE_CONVERSION_FACTOR = (
        273.15  # units are Kelvin (equivalent to 0 degrees Celsius)
    )
    STANDARD_TEMPERATURE = 298.15  # units are Kelvin (equivalent to 25 degrees Celsius)
    EIGENVALUE_TOLERANCE = 1e-12  # mask values lower than the tolerance when calculating the rate constants

    # choose a range of concentration values from which the (k, k-1) are calculated
    # the average slope from the START_POINT to the END_POINT is taken to be the rate constant
    RATE_CONSTANT_EXTRACTION_START_POINT = 0.0
    RATE_CONSTANT_EXTRACTION_END_POINT = 12.0

    def __init__(
        self,
        input_reaction_mechanism: reaction_mechanism,
        input_temp: float,
        input_time: Optional[np.ndarray] = None,
        rxn_profile: Optional[np.ndarray] = None,
        Keq_threshold: Optional[float] = None,
        mass_balance_threshold: Optional[float] = None,
    ):
        # see the following hyperlink for documentation on this string formatting:
        # https://docs.python.org/release/3.0.1/library/string.html#formatstrings
        formatting_string = "I'm born, wahoo!   My temperature is: {0:>8.2F}\N{DEGREE SIGN}C  {1:>8.2F}\N{DEGREE SIGN}K"
        # logger.info(formatting_string.format(input_temp, input_temp + experiment.TEMPERATURE_CONVERSION_FACTOR))

        # required puzzle variables
        self.species_array = input_reaction_mechanism.get_name_set()
        self.number_of_reactions = input_reaction_mechanism.number_of_reactions
        self.number_of_species = input_reaction_mechanism.number_of_species

        # the r, p, v matricess
        self.coefficient_array = input_reaction_mechanism.coefficient_array
        self.reactant_coefficient_array = (
            input_reaction_mechanism.reactant_coefficient_array
        )
        self.product_coefficient_array = (
            input_reaction_mechanism.product_coefficient_array
        )

        # as sorted single values
        self.temp = float(input_temp + Experiment.TEMPERATURE_CONVERSION_FACTOR)
        self.RT = R * np.asfarray(input_temp + Experiment.TEMPERATURE_CONVERSION_FACTOR)
        self.scaling_factor = Experiment.SCALING_FACTOR
        self.Keq_threshold = (
            Keq_threshold if (Keq_threshold is not None) else pow(10, -7)
        )
        self.mass_balance_threshold = (
            mass_balance_threshold
            if (mass_balance_threshold is not None)
            else pow(10, -7)
        )

        # the Es, Ea, Er, Ep matrices
        # Size of (self.number_of_species)
        self.species_energy_array = input_reaction_mechanism.get_energy_set()

        # logger.info("energy array", self.species_energy_array)
        self.activated_energy_array = (
            (input_reaction_mechanism.transition_state_energies)
            if ("transition_state_energies" in input_reaction_mechanism.__dict__)
            else None
        )
        self.reactant_energy_array = None
        self.product_energy_array = None

        rate_constant_size_source = input_reaction_mechanism.number_of_reactions
        # the (k, k-), Rate, Keq matrices
        self.reactant_rate_constants = np.zeros(
            (rate_constant_size_source), dtype=float
        )
        self.product_rate_constants = np.zeros((rate_constant_size_source), dtype=float)
        self.reaction_rate_function = None  # np.zeros(self.number_of_reactions)
        self.theoretical_Keq_array = None  # np.zeros(self.number_of_reactions)
        self.experimental_Keq_array = None  # np.zeros(self.number_of_reactions)

        # the array of time points,
        # np.zeros(determined_at_runtime)
        self.time_array = input_time
        # the species concentration values at each time point, and the initial conc values
        # np.zeros(self.time_array.size[0], number_of_species)
        self.reaction_profile = (
            np.zeros((1, len(self.species_array)))
            if rxn_profile is None
            else rxn_profile
        )

        # logger.info("conc array ", self.reaction_profile)

        # all experiments require an array of transition state energies which is stored in self.activated_energy_array; these values are either given, in the case of a 'true model', or they are calculated, in the case of a 'proposed model'
        # from this the energies of the reactants and the products can be calculated
        # towards the goal of smoothing out plots and making calculations easier, as well as improving the user's experience
        # these calculated values are scaled, this scaling occurs below

        # first we calculate the 'unscaled' Er and Ep
        self.find_Er()
        self.find_Ep()

        # find the largest energy difference between products and reactants for all elementary reactions
        delta_energy_array = np.fabs(
            np.subtract(self.product_energy_array, self.reactant_energy_array)
        )
        # the constant SCALING_FACTOR, defined above, is normalized by the largest energy difference
        scaling_factor = np.divide(self.scaling_factor, np.amax(delta_energy_array))
        # species energies are then scaled by this factor
        self.species_energy_array = np.multiply(
            scaling_factor, self.species_energy_array
        )

        # we recalculate the Er and Ep, which depend on the self.species_energy_array, which has been scaled
        self.find_Er()
        self.find_Ep()

        #### possible diagnostic checks ####
        # make sure that none of the energies are NaN or +/- infinity
        assert np.all(np.isfinite(self.species_energy_array))

    # an experiment object can represent a "true/real world" experiment, or a "solution/proposed/conceptual" experiment
    # a conceptual experiment has the need to add or remove elementary reactions from the experiment object
    #
    # I'm not 100% sure why i dont have an add_rxn function at the moment... I will find out in the future
    def remove_rxn(self, reaction_number):

        # logger.info("Before delete", self.coefficient_array.shape, self.reactant_coefficient_array.shape, self.product_coefficient_array.shape)

        # remove the associated data
        self.coefficient_array = np.delete(
            self.coefficient_array, reaction_number, axis=0
        )
        self.reactant_coefficient_array = np.delete(
            self.reactant_coefficient_array, reaction_number, axis=0
        )
        self.product_coefficient_array = np.delete(
            self.product_coefficient_array, reaction_number, axis=0
        )

        # logger.info("After delete", self.coefficient_array.shape, self.reactant_coefficient_array.shape, self.product_coefficient_array.shape)

        # recalculate the energies of the reactants and the products
        # this is necessary to "refill/rebuild" the self.reactant_energy_array and self.product_energy_array so that they are the appropriate size
        self.number_of_reactions -= 1
        self.find_Er()
        self.find_Ep()

        # logger.info("Had to delete reaciton " + str(reaction_number+1))

    # this method  is invoked when str() is called on an experiment_class object
    def __str__(self):
        user_ouput = "This is an instance of an experiment class object"
        return user_ouput

    # returns an array of the difference between products and reactant for each elementary reaction
    def get_delta_E(self):
        return np.subtract(self.product_energy_array, self.reactant_energy_array)

    # calculates the total energy of all species that are reactants for each reaction
    def find_Er(self):
        self.reactant_energy_array = np.sum(
            np.multiply(self.species_energy_array, self.reactant_coefficient_array),
            axis=1,
        )

    # calculates the total energy of all species that are products for each reaction
    def find_Ep(self):
        self.product_energy_array = np.sum(
            np.multiply(self.species_energy_array, self.product_coefficient_array),
            axis=1,
        )

    # creates the Ea array = 1/3/5kJ + max{Er, Ep}
    # calculates the activation energies, where the values are ENERGY_BARRIER plus the max of (Er, Ep) for each reaction
    # the ENERGY_BARRIER is a dictionary so that each elementary reaction in a puzzle can have its own tailored barrier
    def find_Ea(self):
        Ea_scaling_factor = (
            Experiment.ENERGY_BARRIER_DICTIONARY["medium"]
            if (self.activated_energy_array == None)
            else [
                Experiment.ENERGY_BARRIER_DICTIONARY[i]
                for i in self.activated_energy_array
            ]
        )
        self.activated_energy_array = np.add(
            np.maximum(self.reactant_energy_array, self.product_energy_array),
            Ea_scaling_factor,
        )

    def find_rate_constant(self):
        """
        Calculates the rate constants for each reaction; using the species energy, the activated energy.
        """
        # create the Er and Ep from the provided Es
        self.find_Er()
        self.find_Ep()
        # Ea depends on the Er/Ep, so it needs to be calculated after them:
        self.find_Ea()

        # Fill the array with the Rate constant values. The formulas are:
        # e^{(Er - Ea) / RT}
        # e^{(Ep - Ea) / RT}
        self.reactant_rate_constants = np.exp(
            (self.reactant_energy_array - self.activated_energy_array) / self.RT
        )
        self.product_rate_constants = np.exp(
            (self.product_energy_array - self.activated_energy_array) / self.RT
        )

        # Fill the theoretical_Keq_array, which depends on the rate constant.
        # Calculates an array of Keq's for each elementary reaction, based on the theoretical definition.
        # This only makes sense for 'true' models, where the elementary reaction rates can be exactly calculated.
        self.theoretical_Keq_array = (
            self.reactant_rate_constants / self.product_rate_constants
        )

    def find_reaction_rate_function(self):
        """
        Creates a reaction rate function based on the reactant and product rate constants.
        This function calculates the reaction rate for each elementary reaction based on the concentrations
        of the reactants and products.
        """

        def reaction_rate(conc):
            reactant_rates = self.reactant_rate_constants * np.prod(
                conc**self.reactant_coefficient_array, axis=1
            )
            product_rates = self.product_rate_constants * np.prod(
                conc**self.product_coefficient_array, axis=1
            )
            return reactant_rates - product_rates

        self.reaction_rate_function = reaction_rate

    def find_experimental_Keq_array(self, job_id: str = "unknown job") -> np.ndarray:
        """
        Calculate an array of equilibrium constants (Keq) for each elementary reaction using numerical approximation.

        This method is intended for 'solution/proposed' models, where Keq is determined based on the concentrations
        of reacting species. The equilibrium constants are calculated as the ratio of concentrations of species
        when the reaction profile has reached a steady state or 'plateau'.
        """
        logger = logging.getLogger(job_id).getChild("find_experimental_Keq_array")
        logger.debug("               Finding Keq array empirically...")
        # Select a point in the reaction profile which is approximately the beginning of the 'plateau'.
        initial_timestep = self.find_flat_region(job_id, remove=False)
        # Calculate the average of each chemical species over this 'plateau'.
        # This is the concentration at equilibrium. ['S1'mean, 'S2'mean, ...., 'Sa'mean]
        concentrations = np.nanmean(self.reaction_profile[initial_timestep:], axis=0)
        # take product of ([A]^a * [B]^b * ... etc) for each elementary reaction
        logger.debug(
            "                 concentrations at equilibrium: "
            + HANDY.np_repr(concentrations)
        )
        logger.debug(
            "                 coefficient_array: "
            + HANDY.np_repr(self.coefficient_array)
        )
        Keq = np.prod(np.power(concentrations, self.coefficient_array), axis=1)
        Keq = np.nan_to_num(Keq)
        logger.debug(f"                 Experimental Keq: {Keq}")
        self.experimental_Keq_array = Keq
        return Keq

    def find_reaction_profile(
        self,
        job_id: str = "unknown job",
        input_concentration=None,
        diagnostic_output=False,
    ):
        """
        Solves the coupled ode's, effectively 'runs' the experiment
        Remember that the goal here is to find the 'plateau' - i.e. when the change in the Keq since the last 'step' is below some threshold we consider the reaction as 'completed'
        input_concentration allows us to optionally replace the reaction_profile currently stored in the experiment object
        diagnostic_output is a boolean parameter that, by default, supresses the large amount of possible output from the ode solver
        """
        logger = logging.getLogger(job_id).getChild("find_reaction_profile")

        # the 'calculation' step
        def condition_elementary(conc, t=None):
            # [a,b,c,d,e](powered 'down') = they're pushed down [x,0,0,0,0] the x locations of a 2D array
            # but then multiplied across the other dimension, so [a1,b1,c1,d1,e1]
            # transpose the first array(rotate southwest at leftmost point)
            # multiply across and sum downwards
            reaction_array = self.reaction_rate_function(conc)
            # This is the dY/dt, the change in concentration over time.
            return np.dot(reaction_array, self.coefficient_array)

        self.slice_counter = 0
        self.time_slicee = None

        # for diagnostic purposes
        def condition_elementary_diagnostic(conc):
            dYdt = condition_elementary(conc)
            # not 100% sure on what's happening here
            self.slice_counter = (
                (self.slice_counter + 1)
                if (self.slice_counter + 1 < len(self.time_slicee))
                else 0
            )
            return dYdt

        # the two floats that we use to compare each step
        previous_ln_Keq = None

        # if new concentrations are provided then 'reset' any previously held values
        if input_concentration is not None:
            ode_conc = input_concentration
            self.time_array = None
            self.reaction_profile = None
        # otherwise use the stored values
        else:
            ode_conc = self.reaction_profile[0, :]
            self.reaction_profile = None

        # the 'calculation' loop
        logger.info("				Calculating... ")
        for current_Keq_step in range(1, Experiment.max_Keq_steps + 1):

            # if the time_array is not empty then we want to continue from the end of it, otherwise start at 0
            start_time = self.time_array[-1] if (self.time_array is not None) else 0
            # generate time points over which the odeint solves the coupled ode's, i need refactor this part to be a lot cleaner
            self.time_slicee = np.linspace(
                start_time, start_time + Experiment.step_size, num=Experiment.num_steps
            )

            # solve the couple ode's, with or with extra output
            if diagnostic_output:
                ode_solution, infodict = odeint(
                    condition_elementary_diagnostic,
                    ode_conc,
                    self.time_slicee,
                    full_output=diagnostic_output,
                    atol=Experiment.abserr,
                    rtol=Experiment.relerr,
                    ixpr=True,
                )
            else:
                ode_solution = odeint(
                    condition_elementary,
                    ode_conc,
                    self.time_slicee,
                    atol=Experiment.abserr,
                    rtol=Experiment.relerr,
                    ixpr=True,
                )
            # only compare the previous Keq with the current Keq if we have a previous Keq
            if previous_ln_Keq is not None:
                # logger.info("\n\n=============================== Not the last Keq ===============================")
                # store the 'step' that we just took in the experiment object's private members
                self.time_array = np.concatenate(
                    (self.time_array, self.time_slicee), axis=0
                )
                self.reaction_profile = np.concatenate(
                    (self.reaction_profile, ode_solution), axis=0
                )

                # logger.info(str(self.reaction_profile.shape)
                # 	+ "\n" + HANDY.np_repr(self.time_slicee)
                # 	+ "\n" + HANDY.np_repr(ode_solution)
                # 	)

                # calculate the Keq array and take the log
                self.find_experimental_Keq_array(job_id)
                current_ln_Keq = np.log(self.experimental_Keq_array)

                # compare the last and the current Keq
                if np.all(
                    np.fabs(current_ln_Keq - previous_ln_Keq)
                    < np.asarray(
                        [self.Keq_threshold for x in range(self.number_of_reactions)]
                    )
                ):
                    logger.info("" + str(int(start_time)) + " -- close enough!")
                    break
                else:
                    previous_ln_Keq = current_ln_Keq
                    logger.info(
                        "" + str(int(start_time)) + "..."
                    )  # start_time -> start_time/experiment.step_size

            # we need to run at least one more time before we can check conditions
            else:
                # logger.info("\n\n=============================== First Keq ===============================")
                # store the 'step' that we just took in the experiment object's private members
                self.time_array = self.time_slicee
                self.reaction_profile = ode_solution
                # logger.info(str("") + HANDY.np_repr(self.time_slicee) + "\n" + HANDY.np_repr(ode_solution))

                # calculate the Keq array and take the log
                self.find_experimental_Keq_array(job_id)
                previous_ln_Keq = np.log(self.experimental_Keq_array)

            # the initial concentrations for the next 'step'
            ode_conc = self.reaction_profile[-1, :]
        # if we don't break from the loop then we didn't reach a 'plateau' as defined by our
        else:
            HANDY.warning(
                "				We reached our max_Keq_steps without finding an 'appropriate' Keq"
            )

    def slice_array_by_time(
        self,
        job_id: str,
        array_to_slice: Optional[np.ndarray] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ):
        """
        Returns a slice of `array_to_slice` based on the provided start and end times.
        If `array_to_slice` is None, it defaults to the experiment's `reaction_profile`.
        If `start` or `end` is None, the full array is returned.
        Boundary checks are performed to ensure that the start and end times are within the range of the `time_array`.
        """
        logger = logging.getLogger(job_id).getChild("get_reaction_profile")
        if array_to_slice is None:
            # if no array_to_slice is provided, we use the reaction_profile
            array_to_slice = self.reaction_profile
            logger.info(
                "               No array to slice provided, using the reaction profile."
            )
        if start is None or end is None:
            logger.info(
                "               No slice input provided, returning the full reaction profile."
            )
            return array_to_slice
        start_time = float(start)
        end_time = float(end)
        # Boundary checks for the start and end times.
        if end_time >= self.time_array[-1]:
            logger.warning(
                f"               End time selected, {end_time}, is larger than the time-array. "
                f"Slicing up to the last available time, {self.time_array[-1]}."
            )
            end_time = self.time_array[-1]
        if start_time >= self.time_array[-1]:
            logger.warning(
                f"               Start time selected, {start_time}, is larger than the time-array. "
                f"Slicing up to the last available time, {self.time_array[-1]}."
            )
            start_time = self.time_array[-1]
        if start_time < self.time_array[0]:
            logger.warning(
                f"               Start time selected, {start_time}, is smaller than the first time value. "
                f"Slicing from the first available time, {self.time_array[0]}."
            )
            start_time = self.time_array[0]
        if end_time < self.time_array[0]:
            logger.warning(
                f"               End time selected, {end_time}, is smaller than the first time value. "
                f"Slicing from the first available time, {self.time_array[0]}."
            )
            end_time = self.time_array[0]
        if start_time > end_time:
            logger.warning(
                f"               Start time {start_time} should be earlier than end time {end_time}. "
                f"Reversing the order."
            )
            start_time, end_time = end_time, start_time
        logger.info(
            f"               Slicing the {start_time}~{end_time} part from the time-array with a length of "
            f"{self.time_array[-1]:.2f}."
        )
        # creating the masking array to select the sliced versions of the reaction_profile.
        condition = (self.time_array >= start_time) & (self.time_array <= end_time)
        return array_to_slice[condition]

    def get_matrix_rate_solution(self, job_id: str = "unknown job"):
        """
        Computes the rate constants for a chemical reaction system using a matrix-based approach.

        This method calculates the forward and backward rate constants for a set of chemical reactions
        based on experimental data, including concentration profiles and reaction coefficients. It uses
        matrix operations to solve for the rate constants and ensures that the results are consistent
        with the experimental equilibrium constants.

        Args:
            job_id (str): A unique identifier for the job, used for logging purposes. Defaults to "unknown job".

        Returns:
            np.ndarray: A 2D array containing the forward and backward rate constants for each reaction.
                        The shape of the array is (number_of_reactions, 2), where the first column contains
                        the forward rate constants and the second column contains the backward rate constants.

        Raises:
            HANDY.NegativeCoefficientException: If any of the calculated rate constants are negative.

        Notes:
            - The method slices the concentration and time arrays to focus on a specific time range for
              rate constant extraction.
            - It calculates the change in concentration over time (dS/dt) and reshapes the arrays to
              facilitate matrix operations.
            - The equilibrium constant (Keq) is determined experimentally and used to compute the Q value,
              which represents the reaction quotient.
            - The method constructs matrices (A, M, X) to solve for the rate constants using eigenvalue
              decomposition and matrix inversion.
            - Negative rate constants are checked and an exception is raised if any are found.
        """
        logger = logging.getLogger(job_id).getChild("get_matrix_rate_solution")
        # get the concentration values and trim the dSdt to match the sample size
        slice_of_concentrations = self.slice_array_by_time(
            job_id=job_id,
            array_to_slice=self.reaction_profile,
            start=Experiment.RATE_CONSTANT_EXTRACTION_START_POINT,
            end=Experiment.RATE_CONSTANT_EXTRACTION_END_POINT,
        )
        logger.debug(
            f"               Concentrations over the selected period of time is in an array of shape {slice_of_concentrations.shape}"
            # + ":\n                 " + HANDY.np_repr(slice_of_concentrations).replace("\n", "\n                 ")
        )

        slice_of_time = self.slice_array_by_time(
            job_id=job_id,
            array_to_slice=self.time_array,
            start=Experiment.RATE_CONSTANT_EXTRACTION_START_POINT,
            end=Experiment.RATE_CONSTANT_EXTRACTION_END_POINT,
        )

        # Calculate the change in concentration over time (dSdt)
        # dS is the change in concentration, it is a 2D tensor of shape (n-1, number_of_species).
        dS = np.diff(slice_of_concentrations, axis=0)
        # dt is the change in time, it is a 1D tensor of shape (n-1,).
        dt = np.diff(slice_of_time)
        # `.reshape(-1, 1)` converts the row vector into a column vector. That is, from shape (n-1,) to (n-1, 1).
        # Equivalently, you can use `[:, np.newaxis]`.
        dSdt = dS / dt.reshape(-1, 1)

        # Trim the first elements off the concentration matrix to match the length of the dS/dt matrix.
        slice_of_concentrations = slice_of_concentrations[1:]
        # Reshape the concentration array to be a 3D tensor, where the first dimension is the number of time points, the second dimension is 1 (to allow for broadcasting), and the third dimension is the number of species.
        slice_of_concentrations = slice_of_concentrations.reshape(
            -1, 1, self.number_of_species
        )
        logger.debug(
            f"               We reshaped `slice_of_concentrations` to {slice_of_concentrations.shape}"
            # + ":\n                 " + HANDY.np_repr(slice_of_concentrations).replace("\n", "\n                 ")
        )

        # Do the same to dSdt, so that it is a 3D tensor with the same shape as `slice_of_concentrations`.
        logger.debug(f"               dS/dt is of shape {dSdt.shape}.")
        dSdt = dSdt.reshape(-1, 1, self.number_of_species)
        logger.debug(
            f"               We reshaped `dS/dt` to {dSdt.shape}"
            # + ":\n                 " + HANDY.np_repr(dSdt).replace("\n", "\n                 ")
        )

        table = tabulate(
            self.reactant_coefficient_array,
            headers=self.species_array,
            floatfmt=".4g",
            tablefmt="github",
        )
        logger.info(
            f"               Reactant coefficients, Coef(f), is a {self.reactant_coefficient_array.shape}"
            f" array:\n                 "
            + str(table).replace("\n", "\n                 ")
        )
        table = tabulate(
            self.product_coefficient_array,
            headers=self.species_array,
            floatfmt=".4g",
            tablefmt="github",
        )
        logger.info(
            f"               Product coefficients, Coef(b), is a {self.product_coefficient_array.shape}"
            f" array:\n                 "
            + str(table).replace("\n", "\n                 ")
        )

        # we calculate the Keq based on experimental definition, concentration ratios on the 'plateau'
        emKeq = self.find_experimental_Keq_array(job_id)
        Q = self.get_mass_action_imbalance(emKeq, slice_of_concentrations)
        logger.debug(f"               `Q` is a {Q.shape} array.")

        # logger.info(exKeq.shape, Q.shape, self.number_of_reactions ,self.reactant_coefficient_array.shape, self.product_coefficient_array.shape)
        # reshape, and handling any NaN's
        Q = Q.reshape(-1, self.number_of_reactions, 1)
        Q = np.nan_to_num(Q)
        logger.debug(f"               We reshaped `Q` to {Q.shape}.")

        # the next part is our matrix-based solution to determining rate constants.
        # The problem is that we don't have a handle on the unique solutions yet.
        # TODO: I still need to implement the zero vector isolation removal function.
        # read the documentation here to explain what A,M,inverse M and so forth are doing.

        # Create the A matrix by broadcasting the coefficient array and multiplying with Q.
        A = self.coefficient_array[None, :, :] * Q
        logger.info(
            f"               A is a {A.shape} array, "
            f"corresponding to (num_time_points, num_reactions, num_species)."
        )

        # construct the M matrix
        M = np.einsum("iaj,ibj->ab", A, A)
        logger.info(f"               M is a {M.shape} array.")

        # construct the X matrix
        X = np.matrix(np.zeros((self.number_of_reactions, 1)), dtype=float)
        for a in range(self.number_of_reactions):
            X[a] = np.sum(dSdt[:, 0, :] * A[:, a, :])
            # X[a] = np.sum(np.sum((dSdt[:,0,:] * A[:,a,:]), axis = 1), axis = 0)

        # logger.info("M: \n" + str(M))
        # calculate the eigenvalues and eigenvectors
        if np.all(np.transpose(M) == M):
            logger.info("			We just confirmed that M is symmetric.")
            e_values, e_vectors = np.linalg.eigh(M)
        else:
            logger.info(
                "			Uh Oh, M doesn't seem to be symmetric, do we have a problem?"
            )
            e_values, e_vectors = np.linalg.eig(M)

        # logger.info("\nEigenvectors\n" + str(e_vectors)
        # 	+ "\nEigenvalues\n"  + str(e_values))

        # check if eigenvectors are singular
        # if not then we should be able to check if they are correct
        if np.linalg.det(e_vectors) == 0:

            logger.info("			P is singular, (determinant = 0)")
        else:
            temp = e_vectors * np.transpose(e_vectors)

            logger.info(
                "			We check that P-1AP = Eigenvalues"
                + "\n"
                + str(np.diag(np.linalg.inv(e_vectors) * M * e_vectors))
                + "\n"
                + "Check e_vectors*transpose(e_vectors) = I (the identity matrix)"
                + "\n"
                + str(temp)
            )

        # calculate the inverse of the M matrix
        M_inverse = np.matrix(
            np.zeros((self.number_of_reactions, self.number_of_reactions)), dtype=float
        )
        masked_e_values = ma.masked_less(
            e_values, Experiment.EIGENVALUE_TOLERANCE
        )  # mask any values less than our tolerance parameter

        # logger.info("Eval shape: " + str(e_values.shape)
        # 	+ "\n Evals: " + HANDY.np_repr(e_values)
        # 	+ "\n Masked eval shape: " + str(masked_e_values.shape)
        # 	+ "\n Masked evals: " + HANDY.np_repr(masked_e_values))
        for a in range(self.number_of_reactions):
            for b in range(self.number_of_reactions):
                M_inverse[a, b] = np.sum(
                    e_vectors[a, :] * e_vectors[b, :] / masked_e_values
                )

        M_inverse = np.nan_to_num(M_inverse)

        # print out the forward rate constants
        f_guess = np.asfarray(M_inverse * X)
        b_guess = f_guess / self.experimental_Keq_array[:, np.newaxis]
        self.reactant_rate_constants = np.nan_to_num(f_guess.flatten())
        self.product_rate_constants = np.nan_to_num(b_guess.flatten())

        logger.info(
            f"			reactant rate constants: {self.reactant_rate_constants}\n"
            f"           product rate constants: {self.product_rate_constants}"
        )
        self.rate_constant_array = np.concatenate((f_guess, b_guess), axis=1)
        # now we check for negative rate constants
        check = np.all(np.less(self.rate_constant_array, 0.0), axis=1)
        if np.any(check):
            sys.stdout.flush()
            raise HANDY.NegativeCoefficientException(check)

        return self.rate_constant_array

    def get_mass_action_imbalance(
        self, emKeq: np.ndarray, slice_of_concentrations: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the mass action imbalance (Q) for a set of chemical reactions.
        This method computes the reaction quotient (Q) for each elementary reaction based on the concentrations of the
        reactants and products at a given time point, scaling the latter with the empirically-determined equilibrium
        constant (Keq).

        The formula used is $Q = (∏[R_j]^{μ_j}) − (∏[P_i]^{ν_i} / Keq)$.

        See [Law of mass action](https://en.wikipedia.org/wiki/Law_of_mass_action) for more details.

        Distinguish this with the reaction quotient, another quantity often denoted with Q, which is the ratio of the
        concentrations of products to reactants at a given time point, without scaling by the equilibrium constant.

        @param emKeq: The experimental equilibrium constant for each reaction.
        @param slice_of_concentrations: A 3D array of concentrations of species at different time points.
        @return: A 1D array of the mass action imbalance (Q) for each reaction.
        """
        # reshape, and handling any NaN's
        exKeq = emKeq.reshape(1, -1)
        exKeq = np.nan_to_num(exKeq)
        # calculate the Q value (Rate = f * Q)
        # this is why we reshaped our arrays
        forward_rates = np.prod(
            np.power(slice_of_concentrations, self.reactant_coefficient_array),
            axis=2,
        )
        reverse_rates = np.prod(
            np.power(slice_of_concentrations, self.product_coefficient_array),
            axis=2,
        )
        return forward_rates - reverse_rates / exKeq

    def find_flat_region(
        self, job_id: str, threshold: float = 1e-15, remove: bool = True
    ) -> int:
        """
        The simulation may have run way longer than necessary, resulting in a flat region in the reaction profile.
        This makes the graph hard to read, so we want to remove the flat region at the end of the reaction profile.

        @return cutoff: The number of time-steps to keep in the reaction profile after removing the flat region.
        """
        logger = logging.getLogger(job_id).getChild("remove_flat_region")
        # Sample 1% moments of the reaction profile (but no more than 100 and at least 10, unless we don't have
        # that many time steps, in each case we use all time steps) to find the flat region.
        N = self.time_array.shape[0]
        # Ideally, we want to sample 1% of the reaction profile. How many time-steps is that?
        n = N // 100
        # We want to sample at least 10 time-steps.
        n = max(n, 10)
        # But we don't want to sample more than 100 time-steps.
        n = min(n, 100)
        # If we have less than 100 time-steps, we sample all of them.
        num_samples = min(N, n)
        stride = max(1, N // num_samples)
        logger.info(
            f"               Sampling the reaction profile every {stride} time-steps to find the flat region."
            f" That's {num_samples} points sampled from {N} time-steps available."
        )
        sampled_profile = self.reaction_profile[::stride]
        has_flat_region = False
        for i in range(1, len(sampled_profile)):
            if np.allclose(sampled_profile[i], sampled_profile[i - 1], atol=threshold):
                has_flat_region = True
                break
        if not has_flat_region:
            logger.info("               No flat region found in the reaction profile.")
            return
        cutoff = stride * i
        logger.info(
            f"               Concentrations approximately reach equilibrium at {i / len(sampled_profile) * 100:.2f}% "
            f"of the calculated length (time-step: {i}). Cutting off the reaction profile at {cutoff} time-steps."
        )
        if remove:
            self.time_array = self.time_array[:cutoff]
            self.reaction_profile = self.reaction_profile[:cutoff]
        return cutoff
