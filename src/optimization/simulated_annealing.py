# -*-coding:utf-8 -*-
"""
@File      :   basinhopping.py
@Time      :   2025/04/23 10:56:59
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Simulated annealing for global optimization.
"""

import numpy as np
import pandas as pd

from fight.fleets import ARMADA, SHIP_NAMES, SHIP_COSTS, RELATIVE_VALUES
from fight.scorer import FightScore, create_fleet_target_value
from optimizer.simulated_annealing import SimulatedAnnealing


class FleetSimulatedAnnealing(SimulatedAnnealing):
    def __init__(
        self,
        authorized_ships: np.ndarray,
        fleet_value_max: float,
        fleet_other: np.ndarray,
        n_sim: int = 10,
        n_iter: int = 100,
        progress_bar: bool = False,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.99,
    ):
        self.authorized_ships = authorized_ships
        self.fleet_value_max = fleet_value_max
        self.fleet_other = fleet_other
        self.n_sim = n_sim
        self.n_iter = n_iter
        super().__init__(
            initial_solution=np.ones(len(authorized_ships)) / len(authorized_ships),
            objective_function=self.objective_function,
            jump_function=self.jump_function,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            max_iterations=n_iter,
            progress_bar=progress_bar,
        )

    def objective_function(self, fleet_proportions: np.ndarray) -> float:
        """
        Objective function to minimize.

        Parameters
        ----------
        fleet_proportions : np.ndarray of shape (n_ship_authorized,)
            The proportions of the authorized ships in the fleet.

        Returns
        -------
        float
            The negative score of the fleet.
        """
        if fleet_proportions.sum() == 0:
            return np.inf

        # Transform the fleet proportions to the full fleet size by adding zeros for non-authorized ships
        fleet_proportions_full = np.zeros(SHIP_COSTS.shape[0])
        fleet_proportions_full[self.authorized_ships] = fleet_proportions

        # Create the fleet target value
        fleet_own = create_fleet_target_value(
            target_value=self.fleet_value_max,
            ship_proportions=fleet_proportions_full,
        )

        # Simulate the fights
        scorer = FightScore(
            fleet_own=fleet_own,
            fleet_other=self.fleet_other,
            n_sim=self.n_sim,
            ship_costs=SHIP_COSTS,
            relative_values=RELATIVE_VALUES,
        )
        scorer.simulate_fights()

        # Return the negative score of the fleet
        return -scorer.fight_values_normalized_mean

    def jump_function(
        self, fleet_proportions: np.ndarray, temperature: float
    ) -> np.ndarray:
        """
        Generates a new solution from the current solution and temperature.

        Parameters
        ----------
        fleet_proportions : np.ndarray of shape (n_ship_authorized,)
            The current fleet proportions.
        temperature : float
            The current temperature.

        Returns
        -------
        np.ndarray of shape (n_ship_authorized,)
            The new fleet proportions.
        """
        # Create a random step in the direction of the fleet proportions
        step_size = 1 / np.sqrt(1 - min(np.log(temperature), 0))
        step = np.random.normal(0, step_size, size=fleet_proportions.shape)
        step -= np.mean(step)  # Center the step around zero

        new_fleet_proportions = fleet_proportions + step
        new_fleet_proportions = np.clip(
            new_fleet_proportions, 0, 1
        )  # Ensure proportions are between 0 and 1
        new_fleet_proportions /= np.sum(
            new_fleet_proportions
        )  # Normalize the proportions

        return new_fleet_proportions
