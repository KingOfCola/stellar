import numpy as np

from fight import fight
from fight.fleets import ARMADA, SHIP_NAMES


def simulate_fight(
    compact_fleet_1_start: np.ndarray, compact_fleet_2_start: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates a fight between two fleets of ships.

    Parameters
    ----------
    compact_fleet_1_start : np.ndarray of shape (n_ship_types,)
        The starting fleet 1 to simulate.
    compact_fleet_2_start : np.ndarray of shape (n_ship_types,)
        The starting fleet 2 to simulate.

    Returns
    -------
    FightField
        The fight field containing the two fleets and the result of the fight.
    """
    # Wrap the compact fleets into CompactFleet objects
    compact_fleet_1_start_ = fight.CompactFleet.from_array(compact_fleet_1_start)
    compact_fleet_2_start_ = fight.CompactFleet.from_array(compact_fleet_2_start)

    # Convert the compact fleets to the appropriate format
    fleet_1_ = fight.fleet_from_compact_fleet(compact_fleet_1_start_, ARMADA)
    fleet_2_ = fight.fleet_from_compact_fleet(compact_fleet_2_start_, ARMADA)

    # Create a fight field and simulate the fight
    fight_field = fight.FightField(fleet_1_, fleet_2_, armada=ARMADA)

    fight.fight_fleets(fight_field)

    # Convert the results back to compact fleets
    compact_fleet_1_result_ = fight_field.compact_fleet_1()
    compact_fleet_2_result_ = fight_field.compact_fleet_2()

    return compact_fleet_1_result_.to_array(), compact_fleet_2_result_.to_array()
