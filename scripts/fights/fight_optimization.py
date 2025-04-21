# -*-coding:utf-8 -*-
"""
@File      :   fight_optimization.py
@Time      :   2025/04/20 14:25:37
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Optimization of the fleet with a ship cost constraint.
"""


import numpy as np
import pandas as pd

from fight import fighter
from fight.fleets import ARMADA, SHIP_NAMES, SHIP_COSTS

from matplotlib import pyplot as plt
from tqdm import tqdm

RELATIVE_VALUES = np.array([1, 2, 3])


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
    compact_fleet_ = np.zeros((ship_costs.shape[0],), dtype=int)
    compact_fleet_[: compact_fleet.shape[0]] = compact_fleet

    return compact_fleet_ @ ship_costs @ relative_values


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


def create_random_fleet(
    target_value: float, ships_authorized: np.ndarray
) -> np.ndarray:
    """
    Creates a random fleet with a target value based on the number of ships and their costs.

    Parameters
    ----------
    target_value : float
        The target value of the fleet.
    ships_authorized : np.ndarray of shape (n_ship_types,)
        The authorized ships in the fleet.

    Returns
    -------
    np.array of shape (n_ship_types,)
        The fleet with the target value.
    """
    # Create a random fleet with a target value based on the number of ships and their costs
    authorized_ship_proportions = np.random.dirichlet(
        np.ones(ships_authorized.sum()), size=1
    )[0]
    ship_proportions = np.zeros((ships_authorized.shape[0],), dtype=float)
    ship_proportions[ships_authorized] = authorized_ship_proportions

    return ship_proportions


def optimize_fleet(
    target_value: float,
    ship_others: np.ndarray,
    ships_authorized: np.ndarray = None,
    ship_costs: np.array = SHIP_COSTS,
    relative_values: np.ndarray = RELATIVE_VALUES,
    n_sim: int = 100,
) -> np.array:
    """
    Optimizes the fleet to maximize the value of the fight.

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
        The optimized fleet.
    """
    if ships_authorized is None:
        ships_authorized = np.ones((ship_costs.shape[0],), dtype=bool)

    optimal_fleet_value = -np.inf
    optimal_fleet = np.zeros((ship_costs.shape[0],), dtype=int)

    fleet_values = np.zeros((n_sim,))

    for i in tqdm(
        range(n_sim), total=n_sim, desc="Optimizing fleet", unit="sim", smoothing=0
    ):
        # Create a random fleet with the target value
        ship_proportions = create_random_fleet(target_value, ships_authorized)

        # Create a fleet with the target value
        fleet = create_fleet_target_value(
            target_value, ship_proportions, ship_costs, relative_values
        )

        # Compute the value of the fight with the fleet
        fleet_value = np.mean(
            [
                fight_value(fleet, ship_others, ship_costs, relative_values)
                for _ in range(10)
            ]
        )

        # Check if the fleet is better than the previous one
        if fleet_value > optimal_fleet_value:
            optimal_fleet_value = fleet_value
            optimal_fleet = fleet

        fleet_values[i] = fleet_value

    return optimal_fleet, fleet_values


if __name__ == "__main__":
    # Example usage
    compact_fleet = np.array([1, 2, 3])
    ship_costs = np.array([[1, 2], [4, 3], [5, 5]])
    relative_values = np.array([1, 2])

    value = value_fleet(compact_fleet, ship_costs, relative_values)
    print(f"Value of the fleet: {value} (70)")

    value = value_fleet(compact_fleet)
    print(f"Value of the fleet: {value}")

    target_value = 150
    ship_proportions = np.array([1, 1, 1])
    fleet = create_fleet_target_value(
        target_value, ship_proportions, ship_costs, relative_values
    )
    print(f"Fleet with target value: {fleet}")
    print(
        f"Fleet value: {value_fleet(fleet, ship_costs, relative_values)} ({target_value})"
    )

    compact_fleet_own = np.zeros((ARMADA.nb_ships_,), dtype=int)
    compact_fleet_other = np.zeros((ARMADA.nb_ships_,), dtype=int)

    compact_fleet_own[2] = 2000
    compact_fleet_other[2] = 1000

    values = np.zeros(1000)
    for i in range(len(values)):
        values[i] = fight_value(compact_fleet_own, compact_fleet_other)

    fig, ax = plt.subplots()
    ax.hist(values, label="Fight value")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Fight value distribution")
    ax.legend()
    plt.show()

    # Test the optimization function
    target_value = 50_000_000
    ship_others = np.zeros((ARMADA.nb_ships_,), dtype=int)
    ship_proportions_others = np.zeros((ARMADA.nb_ships_,), dtype=float)
    ship_proportions_others[np.array([7])] = 1

    ship_others = create_fleet_target_value(
        target_value, ship_proportions_others, SHIP_COSTS, RELATIVE_VALUES
    )

    authorized_ships = np.array([False] * ARMADA.nb_ships_)
    authorized_ships[17:23] = True

    optimal_fleet, fleet_values = optimize_fleet(
        target_value, ship_others, ships_authorized=authorized_ships, n_sim=1000
    )
    optimal_fleet_series = pd.Series(
        data=optimal_fleet.values, index=SHIP_NAMES.values[: len(optimal_fleet)]
    )
    print(f"Optimal fleet: {optimal_fleet_series}")

    plt.plot(np.sort(fleet_values), label="Fleet values")
    plt.xlabel("Simulation")
    plt.ylabel("Value")
    plt.title("Fleet values distribution")
