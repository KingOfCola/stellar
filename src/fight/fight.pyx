# -*-coding:utf-8 -*-
'''
@File      :   fight.pyx
@Time      :   2025/04/16 17:56:40
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Fight module.
    This module contains utilities to simulate a fight between two fleets.
    Fleets are represented as 2D numpy arrays `fleet`, where each column `fleet[:, i]` represents a ship:
    - `fleet[0, i]` is the type of ship.
    - `fleet[1, i]` is the attack value of the ship.
    - `fleet[2, i]` is the remaining shield value of the ship.
    - `fleet[3, i]` is the shield generator value of the ship.
    - `fleet[4, i]` is the remaining hull value of the ship.
'''

import numpy as np

TYPE_IDX = 0
ATTACK_IDX = 1
SHIELD_IDX = 2
SHIELD_GENERATOR_IDX = 3
HULL_IDX = 4

cpdef fight_single_round(fleet_1: int[:, ::1], fleet_2: int[:, ::1]) -> tuple[int[:, ::1], int[:, ::1]]:
    """
    Simulate a single round of fight between two fleets.
    The function returns the remaining fleet after the fight.
    """
    # Each fleet attacks the other fleet
    fight_one_way(fleet_1, fleet_2)
    fight_one_way(fleet_2, fleet_1)

    # Remove destroyed ships from both fleets
    fleet_1 = fleet_1[:, np.where(fleet_1[HULL_IDX, :] > 0)[0]]
    fleet_2 = fleet_2[:, np.where(fleet_2[HULL_IDX, :] > 0)[0]]


    # Return the remaining fleet
    return (fleet_1, fleet_2)

cpdef fight_one_way(fleet_attacker: int[:, ::1], fleet_target: int[:, ::1]):
    """
    Simulate inplace a one-way fight between two fleets.
    
    Parameters
    ----------
    fleet_attacker : int[:, ::1]
        The attacking fleet.

    fleet_target : int[:, ::1]
        The defending fleet.
    """
    cdef int attacker_idx, target_idx
    cdef int damage, shielded_damage, unshielded_damage
    cdef int nb_ships_attacker, nb_ships_target

    nb_ships_attacker = fleet_attacker.shape[1]
    nb_ships_target = fleet_2.shape[1]

    # Each ship fom the first fleet attacks a random ship of the second fleet
    for attacker_idx in range(nb_ships_attacker):
        # Select a random target from the second fleet
        target_idx = np.random.randint(0, nb_ships_target)

        # Calculate damage based on the attack value of the attacker and the shield value of the target
        damage = fleet_attacker[ATTACK_IDX, attacker_idx]
        shielded_damage = max(0, damage - fleet_2[SHIELD_IDX, target_idx])
        unshielded_damage = min(damage - shielded_damage, fleet_2[HULL_IDX, target_idx])

        # Apply damage to the target ship
        fleet_2[SHIELD_IDX, target_idx] -= shielded_damage
        fleet_2[HULL_IDX, target_idx] -= unshielded_damage

    