# -*-coding:utf-8 -*-
"""
@File      :   fight_confidence.py
@Time      :   2025/04/20 10:26:55
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Confidence intervals for the fight simulation.
"""


import numpy as np

from fight import fight
from fight.fleets import ARMADA, SHIP_NAMES

from matplotlib import pyplot as plt


def simulate_fight(
    compact_fleet_1: fight.CompactFleet, compact_fleet_2: fight.CompactFleet
) -> fight.FightField:
    """
    Simulates a fight between two fleets of ships.

    Parameters
    ----------
    n1 : int
        Number of ships in fleet 1.
    n2 : int
        Number of ships in fleet 2.

    Returns
    -------
    FightField
        The fight field containing the two fleets and the result of the fight.
    """
    fleet_1 = fight.fleet_from_compact_fleet(compact_fleet_1, ARMADA)
    fleet_2 = fight.fleet_from_compact_fleet(compact_fleet_2, ARMADA)

    # Create a fight field and simulate the fight
    fight_field = fight.FightField(fleet_1, fleet_2, armada=ARMADA)

    fight.fight_fleets(fight_field)

    return fight_field


def simulate_multiple_fights(
    compact_fleets_array: np.ndarray, n_sim: int
) -> np.ndarray:
    """
    Simulates multiple fights between two fleets of ships.

    Parameters
    ----------
    compact_fleets_array : np.ndarray of shape (2, ARMADA.nb_ships_)
        The compact fleets to simulate as an array.
    n_sim : int
        The number of simulations to run.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The results of the simulations.
    """
    compact_fleet_1 = fight.CompactFleet.from_array(compact_fleets_array[0])
    compact_fleet_2 = fight.CompactFleet.from_array(compact_fleets_array[1])

    results = np.zeros((n_sim, 2, ARMADA.nb_ships_), dtype=int)

    for i in range(n_sim):
        fight_field = simulate_fight(compact_fleet_1, compact_fleet_2)
        compact_fleet_result_1 = fight_field.compact_fleet_1().to_array()
        compact_fleet_result_2 = fight_field.compact_fleet_2().to_array()

        results[i, 0, : len(compact_fleet_result_1)] = compact_fleet_result_1
        results[i, 1, : len(compact_fleet_result_2)] = compact_fleet_result_2

    return results


def plot_fight_results(
    results: np.ndarray,
    compact_fleet_start: np.ndarray,
    ship_names: np.ndarray,
    normalize=False,
) -> None:
    """
    Plots the results of the fight simulations.

    Parameters
    ----------
    results : np.ndarray
        The results of the simulations.
    compact_fleet_start : np.ndarray
        The initial fleet counts.
    ship_names : np.ndarray
        The names of the ships.
    normalize : bool, optional
            Whether to normalize the results (default is False).

    Returns
    -------
    None
    """
    mean_results = np.mean(results, axis=0)
    std_results = np.std(results, axis=0)
    mean_start = compact_fleet_start

    if normalize:
        normalize_factor = np.maximum(compact_fleet_start, 1)
        mean_results = mean_results / normalize_factor
        std_results = std_results / normalize_factor
        mean_start = mean_start / normalize_factor

    fig, axes = plt.subplots(figsize=(10, 6), ncols=2, sharey=True)
    for i, ax in enumerate(axes):
        ax.set_title(f"Fleet {i + 1}")
        ax.set_xlabel("Ship Names")
        ax.set_ylabel("Number of Ships")
        ax.bar(
            ship_names,
            mean_results[i],
            yerr=std_results[i],
            fc="limegreen",
            alpha=0.5,
            label="Mean",
        )
        ax.bar(
            x=ship_names,
            height=mean_start[i] - mean_results[i],
            bottom=mean_results[i],
            fc="red",
            alpha=0.5,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_ylim(0, np.max(mean_start) * 1.2)

    fig.tight_layout()

    return fig, axes


if __name__ == "__main__":
    compact_fleets_array = np.zeros((2, ARMADA.nb_ships_), dtype=np.int32)

    compact_fleets_array[0, 2] = 10
    compact_fleets_array[1, 17] = 10

    results = simulate_multiple_fights(
        compact_fleets_array,
        n_sim=1,
    )

    print(results.shape)
    fig, ax = plot_fight_results(
        results,
        compact_fleet_start=compact_fleets_array,
        ship_names=SHIP_NAMES.values,
        normalize=True,
    )
    plt.show()
