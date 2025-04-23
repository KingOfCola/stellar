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
            initial_temperature=1.0,
            cooling_rate=0.99,
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Defense fleet only
    authorized_ships = np.array([17, 18, 19, 20, 21, 22])
    fleet_value_max = 10_000_000

    # 8M of Cruisers
    fleet_other_proportions = np.zeros(len(SHIP_NAMES), dtype=int)
    fleet_other_proportions[4] = 1
    fleet_other_proportions[2] = 0
    fleet_other = create_fleet_target_value(
        target_value=fleet_value_max * 1.0,
        ship_proportions=fleet_other_proportions,
    )

    sa_optimizer = FleetSimulatedAnnealing(
        authorized_ships=authorized_ships,
        fleet_value_max=fleet_value_max,
        fleet_other=fleet_other,
        n_sim=20,
        n_iter=1000,
        progress_bar=True,
    )
    (fleet_proportions, best_score) = sa_optimizer.optimize()

    best_scores = np.array([history_item[2] for history_item in sa_optimizer.history])
    scores = np.array([history_item[1] for history_item in sa_optimizer.history])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=range(len(best_scores)), y=best_scores, label="Best Score", ax=ax)
    sns.lineplot(x=range(len(scores)), y=scores, label="Current Score", ax=ax)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score")
    ax.set_title("Simulated Annealing Optimization")
    ax.legend()
    plt.show()

    fleet_proportions_full = np.zeros(SHIP_COSTS.shape[0])
    fleet_proportions_full[authorized_ships] = fleet_proportions
    fleet_own = create_fleet_target_value(
        target_value=fleet_value_max,
        ship_proportions=fleet_proportions_full,
    )
    fight_field_df = pd.DataFrame(
        data={"Defender": fleet_own.values, "Attacker": fleet_other.values},
        index=SHIP_NAMES.values,
    )
    fleet_own_series = pd.Series(data=fleet_own.values, index=SHIP_NAMES.values)
    fleet_proportions_series = pd.Series(
        data=fleet_proportions_full, index=SHIP_NAMES.values
    )

    print(fight_field_df)
    print(fleet_proportions_series)

    # Estimate the fight score
    scorer = FightScore(
        fleet_own=fleet_own,
        fleet_other=fleet_other,
        n_sim=10,
        ship_costs=SHIP_COSTS,
        relative_values=RELATIVE_VALUES,
    )
    scorer.simulate_fights()

    print(scorer.fight_values_normalized_mean)

    # Try with handpicked values
    fleet_proportions = np.array([1, 0, 0, 0, 0, 0])
    fleet_proportions_full = np.zeros(SHIP_COSTS.shape[0])
    fleet_proportions_full[authorized_ships] = fleet_proportions
    fleet_own = create_fleet_target_value(
        target_value=fleet_value_max,
        ship_proportions=fleet_proportions_full,
    )
    fleet_own_series = pd.Series(data=fleet_own.values, index=SHIP_NAMES.values)
    fleet_proportions_series = pd.Series(
        data=fleet_proportions_full, index=SHIP_NAMES.values
    )

    print(fleet_own_series)
    # print(fleet_proportions_series)

    # Estimate the fight score
    scorer = FightScore(
        fleet_own=fleet_own,
        fleet_other=fleet_other,
        n_sim=100,
        ship_costs=SHIP_COSTS,
        relative_values=RELATIVE_VALUES,
    )
    scorer.simulate_fights()

    print(scorer.fight_values_normalized_mean)
    print("\n" * 3)
