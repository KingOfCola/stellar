# -*-coding:utf-8 -*-
"""
@File      :   basinhopping.py
@Time      :   2025/04/22 10:56:59
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Basinhopping algorithm for global optimization.
"""

import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, OptimizeResult

from fight.fleets import ARMADA, SHIP_NAMES, SHIP_COSTS, RELATIVE_VALUES
from fight.scorer import FightScore, create_fleet_target_value


class FleetBasinHopper:
    def __init__(
        self,
        authorized_ships: np.ndarray,
        fleet_value_max: float,
        fleet_other: np.ndarray,
        n_sim: int = 10,
        n_iter: int = 100,
    ):
        """
        Initializes the FleetBasinHopper class with a compact fleet and ship costs.

        Parameters
        ----------
        authorized_ships : np.ndarray of shape (n_ship_types,)
            The authorized ships for the fleet.
        fleet_value_max : float
            The maximum value of the fleet.
        fleet_other : np.ndarray of shape (n_ship_types,)
            The opponent's fleet.
        n_sim : int, optional
            The number of simulations to run. (default is 10)
        """
        self.authorized_ships = authorized_ships
        self.fleet_value_max = fleet_value_max
        self.fleet_other = fleet_other
        self.n_sim = n_sim

        self.n_iter = n_iter

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

        fleet_proportions_full = np.zeros(SHIP_COSTS.shape[0])
        fleet_proportions_full[self.authorized_ships] = fleet_proportions

        fleet_own = create_fleet_target_value(
            target_value=self.fleet_value_max,
            ship_proportions=fleet_proportions_full,
        )
        scorer = FightScore(
            fleet_own=fleet_own,
            fleet_other=self.fleet_other,
            n_sim=self.n_sim,
            ship_costs=SHIP_COSTS,
            relative_values=RELATIVE_VALUES,
        )
        scorer.simulate_fights()

        return (
            -scorer.fight_values_normalized_mean
        )  # We want to maximize the score, so we minimize the negative score

    def optimize(self) -> OptimizeResult:
        """
        Optimizes the fleet using the basinhopping algorithm.

        Returns
        -------
        OptimizeResult
            The result of the optimization.
        """
        initial_guess = np.random.dirichlet(np.ones(len(self.authorized_ships)))

        bounds = [(0, 1) for _ in range(len(self.authorized_ships))]

        result = basinhopping(
            func=self.objective_function,
            x0=initial_guess,
            take_step=FleetStepper(),
            niter=self.n_iter,
            T=1.0,
            stepsize=1.0,
            disp=True,
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "bounds": bounds,
                "options": {"disp": True},
            },
        )

        return result


class FleetStepper:
    def __init__(self, step_size: float = 0.1):
        """
        Initializes the FleetStepper class with a fleet and step size.

        Parameters
        ----------
        step_size : float, optional
            The step size for the optimization. (default is 0.1)
        """
        self.step_size = step_size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Steps through the fleet.

        Returns
        -------
        np.ndarray of shape (n_ship_types,)
            The stepped fleet.
        """
        x1 = np.copy(x) / np.sum(x)  # Normalize the fleet proportions
        x2 = np.random.dirichlet(np.ones_like(x1))
        center = np.ones_like(x1) / len(x1)  # Center of the simplex
        direction = x2 - center  # Direction to the new point
        direction /= np.linalg.norm(direction)  # Normalize the direction
        direction *= self.step_size

        # Checks that no coordinate crosses zero.
        d_neg = direction < 0
        maximal_jump = np.min(
            x1[d_neg] / (-direction[d_neg])
        )  # Maximum jump in the direction before crossing zero

        if maximal_jump < 1:
            # If the jump is too big, we need to scale it down
            direction *= maximal_jump

        return x1 + direction  # Return the new fleet proportions


if __name__ == "__main__":
    # Defense fleet only
    authorized_ships = np.array([17, 18, 19, 20, 21, 22])
    fleet_value_max = 10_000_000

    # 8M of Cruisers
    fleet_other_proportions = np.zeros(len(SHIP_NAMES), dtype=int)
    fleet_other_proportions[10] = 1
    fleet_other_proportions[2] = 1
    fleet_other = create_fleet_target_value(
        target_value=fleet_value_max * 1.0,
        ship_proportions=fleet_other_proportions,
    )

    optimizer = FleetBasinHopper(
        authorized_ships=authorized_ships,
        fleet_value_max=fleet_value_max,
        fleet_other=fleet_other,
        n_sim=100,
        n_iter=20,
    )
    result = optimizer.optimize()
    print(result)

    fleet_proportions = result.x
    fleet_proportions_full = np.zeros(SHIP_COSTS.shape[0])
    fleet_proportions_full[authorized_ships] = fleet_proportions
    fleet_own = create_fleet_target_value(
        target_value=fleet_value_max,
        ship_proportions=fleet_proportions_full,
    )
    fight_field_df = pd.DataFrame(
        data={"Defender": fleet_own.values, "Attacker": fleet_other.values}, index=SHIP_NAMES.values
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
    fleet_proportions = np.array([0, 0, 0, 0, 0, 1])
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
