import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from utils.paths import output
from fight.fleets import ARMADA, SHIP_NAMES, SHIP_COSTS, RELATIVE_VALUES
from fight.scorer import FightScore, create_fleet_target_value
from optimization.simulated_annealing import FleetSimulatedAnnealing
from reporting.fleet_optimizer import make_optimization_report, FleetOptimizationData

SHIP_PERMUTATION = np.array(
    [
        2,
        3,
        4,
        5,
        8,
        10,
        6,
        7,
        11,
        9,
        0,
        1,
        14,
        12,
        13,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
    ]
)

if __name__ == "__main__":
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
        n_iter=100,
        progress_bar=True,
    )
    (fleet_proportions, best_score) = sa_optimizer.optimize()

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

    # Estimate the fight score
    scorer = FightScore(
        fleet_own=fleet_own,
        fleet_other=fleet_other,
        n_sim=100,
        ship_costs=SHIP_COSTS,
        relative_values=RELATIVE_VALUES,
    )
    scorer.simulate_fights()

    fleet_optimization_data = FleetOptimizationData(
        attacker_ship_counts=fleet_other[SHIP_PERMUTATION].values,
        defender_ship_counts=fleet_own[SHIP_PERMUTATION].values,
        attacker_costs_mean=scorer.fight_costs_other_mean,
        defender_costs_mean=scorer.fight_costs_own_mean,
        attacker_costs_std=scorer.fight_costs_other_std,
        defender_costs_std=scorer.fight_costs_own_std,
    )
    make_optimization_report(
        fleet_optimization_data=fleet_optimization_data,
        output_filename=output("reports/fleet_report.pdf").absolute().as_posix(),
    )

if False:

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
