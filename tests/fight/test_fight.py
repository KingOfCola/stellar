# -*-coding:utf-8 -*-
"""
@File      :   test_fight.py
@Time      :   2025/04/16 17:56:40
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Tests of the fight module.
"""

import numpy as np

from fight import fight

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

fleet_1.describe()
fleet_2.describe()
