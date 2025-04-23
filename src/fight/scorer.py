# -*-coding:utf-8 -*-
"""
@File      :   scorer.py
@Time      :   2025/04/22 11:03:45
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Scoring class for evaluating the performance of a model.
"""


import numpy as np
import pandas as pd

from fight import fighter
from fight.fleets import ARMADA, SHIP_NAMES, SHIP_COSTS, RELATIVE_VALUES

from matplotlib import pyplot as plt
from tqdm import tqdm


class FightScore:
    def __init__(
        self,
        fleet_own: np.ndarray,
        fleet_other: np.ndarray,
        n_sim: int = 10,
        ship_costs: np.array = SHIP_COSTS,
        relative_values: np.ndarray = RELATIVE_VALUES,
    ):
        """
        Initializes the FightScore class with a compact fleet and ship costs.

        Parameters
        ----------
        fleet_own : np.ndarray of shape (n_ship_types,)
            Our fleet.
        fleet_other : np.ndarray of shape (n_ship_types,)
            The opponent's fleet.
        n_sim : int, optional
            The number of simulations to run. (default is 10)
        ship_costs : np.array of shape (n_ship_types, n_elements), optional
            The costs of the ships. (optional, default are the values of the ships from the game)
        relative_values : np.ndarray of shape (n_elements,), optional
            The relative values of the ships. (optional, default is [1, 2, 3])
        """
        self.fleet_own = fleet_own
        self.fleet_other = fleet_other
        self.n_sim = n_sim
        self.ship_costs = ship_costs
        self.relative_values = relative_values

        # Compute the value of the fleets
        self.fleet_own_value = value_fleet(
            fleet_own, ship_costs=ship_costs, relative_values=relative_values
        )
        self.fleet_other_value = value_fleet(
            fleet_other, ship_costs=ship_costs, relative_values=relative_values
        )

        # Initialize the results
        n_ship_types = ship_costs.shape[0]
        n_elements = ship_costs.shape[1]

        self.fleet_own_results = np.zeros((n_sim, n_ship_types), dtype=int)
        self.fleet_other_results = np.zeros((n_sim, n_ship_types), dtype=int)
        self.fight_values = np.full((n_sim,), np.nan)
        self.fight_values_mean = np.nan

        self.fight_values_normalized = np.full((n_sim,), np.nan)
        self.fight_values_normalized_mean = np.nan

        self.fight_costs_own = np.zeros((n_sim, n_elements), dtype=int)
        self.fight_costs_own_mean = np.zeros((n_elements,), dtype=int)
        self.fight_costs_own_std = np.zeros((n_elements,), dtype=int)

        self.fight_costs_other = np.zeros((n_sim, n_elements), dtype=int)
        self.fight_costs_other_mean = np.zeros((n_elements,), dtype=int)
        self.fight_costs_other_std = np.zeros((n_elements,), dtype=int)

    def simulate_single_fight(self, round: int = 0):
        """
        Simulates a fight between the two fleets and computes the value of the fight.
        After the fight, it populates the results of the fleets and the value of the fight.
        The results are stored in the class attributes `fleet_own_results` and `fleet_other_results`.

        Parameters
        ----------
        round : int, optional
            The round number of the simulation. (default is 0)
        """
        # Create a fight field with the two fleets
        fleet_own_results, fleet_other_results = fighter.simulate_fight(
            self.fleet_own, self.fleet_other
        )

        # Store the results in the class attributes
        self.fleet_own_results[round, : len(fleet_own_results)] = fleet_own_results
        self.fleet_other_results[round, : len(fleet_other_results)] = (
            fleet_other_results
        )

        # Compute the value of the fight
        fleet_own_value_result = value_fleet(
            fleet_own_results,
            ship_costs=self.ship_costs,
            relative_values=self.relative_values,
        )
        fleet_other_value_result = value_fleet(
            fleet_other_results,
            ship_costs=self.ship_costs,
            relative_values=self.relative_values,
        )

        # Compute the value of the fight
        own_losses = self.fleet_own_value - fleet_own_value_result
        other_losses = self.fleet_other_value - fleet_other_value_result
        self.fight_values[round] = other_losses - own_losses
        self.fight_values_normalized[round] = (
            self.fight_values[round] / self.fleet_other_value
        )

        # Compute the costs of the fleets in resources
        self.fight_costs_own[round, :] = fleet_cost(
            fleet_own_results, ship_costs=self.ship_costs
        ) - fleet_cost(self.fleet_own, ship_costs=self.ship_costs)
        self.fight_costs_own_mean = np.nanmean(self.fight_costs_own, axis=0)
        self.fight_costs_own_std = np.nanstd(self.fight_costs_own, axis=0)

        self.fight_costs_other[round, :] = fleet_cost(
            fleet_other_results, ship_costs=self.ship_costs
        ) - fleet_cost(self.fleet_other, ship_costs=self.ship_costs)
        self.fight_costs_other_mean = np.nanmean(self.fight_costs_other, axis=0)
        self.fight_costs_other_std = np.nanstd(self.fight_costs_other, axis=0)

    def simulate_fights(self):
        """
        Simulates multiple fights between the two fleets and computes the value of the fights.
        The results are stored in the class attributes `fleet_own_results`, `fleet_other_results`,
        `fight_values`, and `fight_values_normalized`.

        Parameters
        ----------
        """
        for round in range(self.n_sim):
            self.simulate_single_fight(round)

        # Compute the mean values of the fights
        self.fight_values_mean = np.nanmean(self.fight_values)
        self.fight_values_normalized_mean = np.nanmean(self.fight_values_normalized)


def value_fleet(
    compact_fleet: np.array,
    ship_costs: np.array = SHIP_COSTS,
    relative_values: np.ndarray = RELATIVE_VALUES,
) -> float:
    """
    Computes the value of a fleet based on the number of ships and their costs.

    Parameters
    ----------
    compact_fleet : np.array of shape (n_ship_types,)
        The compact fleet to compute the value for.
    ship_costs : np.array of shape (n_ship_types, n_elements)
        The costs of the ships. (optional, default are the values of the ships from the game)
    relative_values : np.ndarray of shape (n_elements,)
        The relative values of the ships. (optional, default is [1, 2, 3])

    Returns
    -------
    float
        The value of the fleet, in terms of material coefficient 1.
    """

    return (
        fleet_cost(compact_fleet=compact_fleet, ship_costs=ship_costs) @ relative_values
    )


def fleet_cost(
    compact_fleet: np.array,
    ship_costs: np.array = SHIP_COSTS,
) -> float:
    """
    Computes the cost of a fleet based on the number of ships and their costs.

    Parameters
    ----------
    compact_fleet : np.array of shape (n_ship_types,)
        The compact fleet to compute the cost for.
    ship_costs : np.array of shape (n_ship_types, n_elements)
        The costs of the ships. (optional, default are the values of the ships from the game)

    Returns
    -------
    float
        The cost of the fleet.
    """
    compact_fleet_ = np.zeros((ship_costs.shape[0],), dtype=int)
    compact_fleet_[: compact_fleet.shape[0]] = compact_fleet
    return compact_fleet_ @ ship_costs


def create_fleet_target_value(
    target_value: float,
    ship_proportions: np.ndarray,
    ship_costs: np.array = SHIP_COSTS,
    relative_values: np.ndarray = RELATIVE_VALUES,
) -> np.array:
    """
    Creates a fleet with a target value based on the number of ships and their costs.

    Parameters
    ----------
    target_value : float
        The target value of the fleet.
    ship_proportions : np.ndarray of shape (n_ship_types,)
        The proportions of the ships in the fleet.
    ship_costs : np.array of shape (n_ship_types, n_elements)
        The costs of the ships. (optional, default are the values of the ships from the game)
    relative_values : np.ndarray of shape (n_elements,)
        The relative values of the ships. (optional, default is [1, 2, 3])

    Returns
    -------
    np.array of shape (n_ship_types,)
        The fleet with the target value.
    """
    # Create a fleet with a target value based on the number of ships and their costs
    ship_proportions_ = np.zeros((ship_costs.shape[0],), dtype=float)
    ship_proportions_[: ship_proportions.shape[0]] = ship_proportions

    # Check if the proportions are valid
    if np.sum(ship_proportions_) == 0:
        raise ValueError("The sum of the ships proportions must be greater than 0.")
    if any(ship_proportions_ < 0):
        raise ValueError("The ships proportions must be greater than or equal to 0.")

    # Normalize the proportions to sum to 1
    ship_proportions_ /= np.sum(ship_proportions_)
    ship_values = ship_proportions_ * target_value

    # Compute the value of each ship standardized by the relative values
    ship_cost_standardized = ship_costs @ relative_values

    # Compute the number of ships needed for each type
    fleet = ship_values / ship_cost_standardized
    fleet = np.round(fleet).astype(int)

    return fleet


def fight_value(
    compact_fleet_own: np.ndarray,
    compact_fleet_other: np.ndarray,
    ship_costs: np.array = SHIP_COSTS,
    relative_values: np.ndarray = RELATIVE_VALUES,
) -> float:
    """
    Computes the value of a fight between two fleets. The value of a fight is the
    value of the other fleet destroyed minus the value of the own fleet destroyed.

    Parameters
    ----------
    compact_fleet_own : np.ndarray of shape (n_ship_types,)
        The compact fleet of the player.
    compact_fleet_other : np.ndarray of shape (n_ship_types,)
        The compact fleet of the opponent.
    ship_costs : np.array of shape (n_ship_types, n_elements)
        The costs of the ships. (optional, default are the values of the ships from the game)
    relative_values : np.ndarray of shape (n_elements,)
        The relative values of the ships. (optional, default is [1, 2, 3])

    Returns
    -------
    float
        The value of the fight.
    """
    # Create a fight field with the two fleets
    compact_fleet_own_result, compact_fleet_other_result = fighter.simulate_fight(
        compact_fleet_own, compact_fleet_other
    )

    # Compute the value of the fleets before and after the fight
    value_own_start = value_fleet(compact_fleet_own, ship_costs, relative_values)
    value_own_result = value_fleet(
        compact_fleet_own_result, ship_costs, relative_values
    )
    value_other_start = value_fleet(compact_fleet_other, ship_costs, relative_values)
    value_other_result = value_fleet(
        compact_fleet_other_result, ship_costs, relative_values
    )

    # Compute the losses of the fight
    losses_own = value_own_start - value_own_result
    losses_other = value_other_start - value_other_result

    # Compute the value of the fight
    value_fight = losses_other - losses_own

    return value_fight
