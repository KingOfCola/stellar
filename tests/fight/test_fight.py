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

v1 = np.array([1, 2, 3, 4, 5])
v2 = np.array([2, 200, 30, 40, 50])


compact_fleet_array = np.array([2, 5])
armada_array = np.array([v1, v2]).T[1:, :]

armada = fight.Armada.from_array(armada_array)
compact_fleet = fight.CompactFleet.from_array(compact_fleet_array)

fleet = fight.fleet_from_compact_fleet(compact_fleet, armada)

fleet_1_summary = [(5, v1), (2, v2)]
fleet_2_summary = [(3, v1), (7, v2)]

fleet_1 = np.concatenate(
    [np.repeat(ship[:, None], count, axis=1) for count, ship in fleet_1_summary], axis=1
)
fleet_2 = np.concatenate(
    [np.repeat(ship[:, None], count, axis=1) for count, ship in fleet_2_summary], axis=1
)

result_1, result_2 = fight.fight_single_round(fleet_1, fleet_2)
print("Fleet 1 summary:")
print(result_1)

print("Fleet 2 summary:")
print(result_2)
