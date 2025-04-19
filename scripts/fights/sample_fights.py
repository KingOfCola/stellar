# -*-coding:utf-8 -*-
"""
@File      :   sample_fights.py
@Time      :   2025/04/19 15:19:45
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Sample fights for the game.
"""

import numpy as np

from fight import fight
from fight.fleets import ARMADA, SHIP_NAMES


def simulate_fight(n1: int, n2: int) -> None:
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
    counts_1 = np.int_(np.random.dirichlet(np.ones(ARMADA.nb_ships_), size=1)[0] * n1)
    counts_2 = np.int_(np.random.dirichlet(np.ones(ARMADA.nb_ships_), size=1)[0] * n2)

    # Create two fleets with random ships
    compact_fleet_1 = fight.CompactFleet.from_array(counts_1)
    compact_fleet_2 = fight.CompactFleet.from_array(counts_2)

    fleet_1 = fight.fleet_from_compact_fleet(compact_fleet_1, ARMADA)
    fleet_2 = fight.fleet_from_compact_fleet(compact_fleet_2, ARMADA)

    # Create a fight field and simulate the fight
    fight_field = fight.FightField(fleet_1, fleet_2)

    fight.fight_fleets(fight_field)

    return fight_field

%timeit simulate_fight(100000, 200000)


v1 = np.array([1, 5, 3, 4, 5])
v2 = np.array([2, 200, 30, 40, 50])


compact_fleet_1_array = np.array([5, 2])
compact_fleet_2_array = np.array([3, 7])
armada_array = np.array([v1, v2]).T[1:, :]

armada = fight.Armada.from_array(armada_array)
compact_fleet_1 = fight.CompactFleet.from_array(compact_fleet_1_array)
compact_fleet_2 = fight.CompactFleet.from_array(compact_fleet_2_array)

fleet_1 = fight.fleet_from_compact_fleet(compact_fleet_1, armada)
fleet_2 = fight.fleet_from_compact_fleet(compact_fleet_2, armada)

fight_field = fight.FightField(fleet_1, fleet_2)

print("Battle started")
fight_field.describe()
fight.fight_fleets(fight_field)

print("Battle ended")
fight_field.describe()

compact_fleet_1_end = fight_field.compact_fleet_1()
compact_fleet_2_end = fight_field.compact_fleet_2()

print("Fleet 1 end:", compact_fleet_1_end.to_array())
print("Fleet 2 end:", compact_fleet_2_end.to_array())
