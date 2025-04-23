# -*-coding:utf-8 -*-
"""
@File      :   simulated_annealing.py
@Time      :   2025/04/23 11:08:47
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Simulated annealing algorithm for global optimization.
"""

import numpy as np
from tqdm import tqdm


class SimulatedAnnealing:
    def __init__(
        self,
        initial_solution,
        objective_function,
        jump_function,
        initial_temperature,
        cooling_rate,
        max_iterations,
        progress_bar: bool = False,
    ):
        """
        Initializes the SimulatedAnnealing class with an initial solution and parameters.

        Parameters
        ----------
        initial_solution : np.ndarray
            The initial solution to start the optimization.
        objective_function : callable (np.ndarray -> float)
            The objective function to minimize.
        jump_function : callable (np.ndarray, float -> np.ndarray)
            The function to generate a new solution from the current solution and temperature.
        initial_temperature : float
            The initial temperature for the annealing process.
        cooling_rate : float
            The rate at which the temperature decreases.
        max_iterations : int
            The maximum number of iterations to perform.
        """
        self.current_solution = initial_solution
        self.current_energy = objective_function(initial_solution)
        self.objective_function = objective_function
        self.jump_function = jump_function
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.best_solution = initial_solution
        self.best_energy = self.current_energy

        self.progress_bar = progress_bar

        self.history = []
        self.history.append(
            (self.current_solution, self.current_energy, self.current_energy)
        )

    def optimize(self):
        """
        Performs the simulated annealing optimization.

        Returns
        -------
        best_solution : np.ndarray
            The best solution found during the optimization.
        best_energy : float
            The energy of the best solution found.
        """
        for iteration in tqdm(
            range(self.max_iterations),
            total=self.max_iterations,
            smoothing=0,
            disable=not self.progress_bar,
        ):
            # Generate a new candidate solution and calculate its energy
            new_solution = self.jump_function(self.current_solution, self.temperature)
            new_energy = self.objective_function(new_solution)

            # Update the best solution if the new solution is better
            if new_energy < self.best_energy:
                self.best_solution = new_solution
                self.best_energy = new_energy

            # If the new solution is better, accept it unconditionally
            # If the new solution is worse, accept it with a certain probability
            # based on the temperature and the energy difference
            if np.random.rand() < SimulatedAnnealing.jump_probability(
                self.current_energy, new_energy, self.temperature
            ):
                self.current_energy = new_energy
                self.current_solution = new_solution

            # Reduce temperature and update history
            self.temperature *= self.cooling_rate
            self.history.append(
                (self.current_solution, self.current_energy, self.best_energy)
            )

        return self.best_solution, self.best_energy

    @staticmethod
    def jump_probability(current_energy, new_energy, temperature):
        """
        Calculates the probability of accepting a new solution.

        Parameters
        ----------
        current_energy : float
            The energy of the current solution.
        new_energy : float
            The energy of the new solution.
        temperature : float
            The current temperature.

        Returns
        -------
        float
            The probability of accepting the new solution.
        """
        # If the new solution is better, accept it unconditionally
        if new_energy < current_energy:
            return 1.0
        # If the new solution is worse, calculate the acceptance probability
        # using the Boltzmann factor
        else:
            return np.exp((current_energy - new_energy) / temperature)
