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
cimport numpy as cnp

from libc.stdlib cimport malloc, free, realloc

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int64

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef cnp.int64_t DTYPE_t

cdef int TYPE_IDX = 0
cdef int ATTACK_IDX = 1
cdef int SHIELD_IDX = 2
cdef int SHIELD_GENERATOR_IDX = 3
cdef int HULL_IDX = 4

cdef int SHIP_CHAR_SIZE = 5

cdef int MAX_ROUNDS = 6

cdef DTYPE_t DEFLECTION_RATIO = 100

cdef class FightField:
    """
    Class to simulate a fight between two fleets.
    The class contains the following attributes:
    - fleet_1: The first fleet.
    - fleet_2: The second fleet.
    - max_rounds: The maximum number of rounds to simulate.
    """

    cdef Fleet fleet_1, fleet_2
    cdef int max_rounds

    def __cinit__(self, Fleet fleet_1, Fleet fleet_2, int max_rounds=MAX_ROUNDS):
        """
        Initialize the FightField object.
        The function takes two fleets and the maximum number of rounds to simulate.
        The fleets are stored as 2D numpy arrays, where each column represents a ship.
        """
        # Copy the fleets to the allocated memory
        self.fleet_1 = fleet_1
        self.fleet_2 = fleet_2

        # Store the additional parameters
        self.max_rounds = max_rounds

    def describe(self):
        """
        Prints the number of ships in each fleet and a sample of the fleets.
        """
        print("Fleet 1:")
        self.fleet_1.describe()
        print()
        print("Fleet 2:")
        self.fleet_2.describe()

    def compact_fleet_1(self):
        """
        Convert the first fleet to a compact fleet.

        Returns
        -------
        CompactFleet
            The compact fleet of the first fleet.
        """
        return compact_fleet_from_fleet(self.fleet_1)

    def compact_fleet_2(self):
        """
        Convert the second fleet to a compact fleet.

        Returns
        -------
        CompactFleet
            The compact fleet of the second fleet.
        """
        return compact_fleet_from_fleet(self.fleet_2)

    def __dealloc__(self):
        pass

cdef class Fleet:
    """
    Class to represent a fleet of ships.
    The class contains the following attributes:
    - ships: The fleet as a 2D numpy array.
    - nb_ships: The number of ships in the fleet.
    """

    cdef DTYPE_t* ships
    cdef int nb_ships

    def __cinit__(self, int nb_ships):
        """
        Initialize the Fleet object.
        The function takes a fleet as a 2D numpy array and stores it as a 1D array.
        """
        self.nb_ships = nb_ships

    cdef void initialize(self):
        """
        Initialize the fleet with zeros.
        """
        self.ships = <DTYPE_t *> malloc(self.nb_ships * SHIP_CHAR_SIZE * sizeof(DTYPE_t))

    cdef void copy_2d_array(self, cnp.ndarray[DTYPE_t, ndim=2] ships):
        """
        Copy the fleet to the allocated memory.
        """
        self.initialize()

        # Copy the fleet to the allocated memory
        cdef int ship_idx, ship_char
        for ship_idx in range(self.nb_ships):
            for ship_char in range(SHIP_CHAR_SIZE):
                self.ships[ship_idx * SHIP_CHAR_SIZE + ship_char] = ships[ship_char, ship_idx]

    cdef bint is_empty(self):
        """
        Checks whether the fleet is empty
        """
        return self.nb_ships == 0

    def describe(self):
        """
        Prints the number of ships and a sample of the ships array
        """
        print("Number of ships: ", self.nb_ships)
        print("      Type    Attack    Shield Shield g.      Hull")
        for ship_idx in range(min(5, self.nb_ships)):
            for ship_char in range(SHIP_CHAR_SIZE):
                print(f"{self.ships[ship_idx * SHIP_CHAR_SIZE + ship_char]: >10}", end="")
            print()

        if self.nb_ships > 10:
            print("...")

        for ship_idx in range(max(5, self.nb_ships - 5), self.nb_ships):
            for ship_char in range(SHIP_CHAR_SIZE):
                print(f"{self.ships[ship_idx * SHIP_CHAR_SIZE + ship_char]: >10}", end="")
            print()

    def __dealloc__(self):
        # Deallocate the fleet
        if self.ships is not NULL:
            free(self.ships)

cdef class Armada:
    cdef DTYPE_t* ships
    cdef int nb_ships

    def __cinit__(self, int nb_ships):
        """
        Initialize the Fleet object.
        The function takes a fleet as a 2D numpy array and stores it as a 1D array.
        """
        self.nb_ships = nb_ships

    cdef void copy_2d_array(self, cnp.ndarray[DTYPE_t, ndim=2] ships):
        """
        Copy the fleet to the allocated memory.
        """
        self.ships = <DTYPE_t *> malloc(self.nb_ships * (SHIP_CHAR_SIZE - 1) * sizeof(DTYPE_t))
        cdef int ship_idx, ship_char
        for ship_idx in range(self.nb_ships):
            for ship_char in range((SHIP_CHAR_SIZE - 1)):
                self.ships[ship_idx * (SHIP_CHAR_SIZE - 1) + ship_char] = ships[ship_char, ship_idx]

    cdef bint is_empty(self):
        """
        Checks whether the fleet is empty
        """
        return self.nb_ships == 0
    
    @staticmethod
    def from_array(ships):
        """
        Convert a numpy array to an Armada.

        Parameters
        ----------
        ships : cnp.ndarray[DTYPE_t, ndim=2]
            The numpy array to convert.

        Returns
        -------
        Armada
            The created Armada object.
        """
        cdef int nb_ships = ships.shape[1]
        cdef Armada armada = Armada(nb_ships)
        armada.copy_2d_array(ships)
        return armada

    @property
    def nb_ships_(self):
        return self.nb_ships

    def __dealloc__(self):
        # Deallocate the fleet
        if self.ships is not NULL:
            free(self.ships)



cdef class CompactFleet:
    """
    Class to represent compactly a fleet.
    A compact fleet consists solely in a mapping of ships types to the number of ships
    in the armada.
    """
    cdef int nb_ship_types
    cdef int* ships

    def __cinit__(self, int nb_ship_types):
        self.nb_ship_types = nb_ship_types

    cdef int nb_ships(self):
        """
        Find the number of ships in the fleet.

        Returns
        -------
        int
            The number of ships in the fleet.
        """
        cdef int ship_type_idx, nb_ships = 0

        # Count the number of ships in the fleet
        for ship_type_idx in range(self.nb_ship_types):
            nb_ships += self.ships[ship_type_idx]

        return nb_ships
    
    cdef void initialize(self):
        """
        Initialize the compact fleet with zeros.
        """
        self.ships = <int *> malloc(self.nb_ship_types * sizeof(int))
        cdef int ship_type_idx
        for ship_type_idx in range(self.nb_ship_types):
            self.ships[ship_type_idx] = 0

    def to_array(self):
        """
        Convert the compact fleet to a numpy array.

        Returns
        -------
        cnp.ndarray[DTYPE_t, ndim=1]
            The compact fleet as a numpy array.
        """
        # Create a numpy array to store the compact fleet
        cdef ships = np.empty(self.nb_ship_types, dtype=int)
        cdef int ship_type_idx

        # Populate the numpy array with the ships from the compact fleet
        for ship_type_idx in range(self.nb_ship_types):
            ships[ship_type_idx] = self.ships[ship_type_idx]

        return ships

    @staticmethod
    def from_array(ships):
        """
        Convert a numpy array to a compact fleet.

        Parameters
        ----------
        ships : cnp.ndarray[DTYPE_t, ndim=1]
            The numpy array to convert.
        """
        # Create a new compact fleet with the number of ship types
        cdef int ship_type_idx, nb_ship_types
        nb_ship_types = ships.shape[0]

        cdef CompactFleet compact_fleet = CompactFleet(nb_ship_types)
        compact_fleet.initialize()

        # Populate the compact fleet with the ships from the numpy array
        for ship_type_idx in range(nb_ship_types):
            compact_fleet.ships[ship_type_idx] = ships[ship_type_idx]

        return compact_fleet

    def __dealloc__(self):
        """
        Deallocate the fleet
        """
        if self.ships is not NULL:
            free(self.ships)

cdef int nb_fleet_types(Fleet fleet):
    """
    Find the number of unique ship types in the fleet.

    Parameters
    ----------
    fleet : Fleet
        The fleet to analyze.

    Returns
    -------
    int
        The number of unique ship types in the fleet.
    """
    cdef int ship_idx
    cdef DTYPE_t ship_types = -1

    # Find the maximum ship type in the fleet
    for ship_idx in range(fleet.nb_ships):
        ship_types = max(ship_types, fleet.ships[ship_idx * SHIP_CHAR_SIZE + TYPE_IDX])

    return ship_types + 1

cpdef CompactFleet compact_fleet_from_fleet(Fleet fleet):
    """

    """
    cdef int nb_ship_types = nb_fleet_types(fleet)
    cdef CompactFleet compact_fleet = CompactFleet(nb_ship_types)
    cdef int ship_idx, ship_type

    cdef DTYPE_t ship_type_idx

    compact_fleet.initialize()
    
    # Initialize the compact fleet with zeros
    for ship_type_idx in range(nb_ship_types):
        compact_fleet.ships[ship_type_idx] = 0

    # Count the number of ships of each type in the fleet
    for ship_idx in range(fleet.nb_ships):
        ship_type = fleet.ships[ship_idx * SHIP_CHAR_SIZE + TYPE_IDX]
        compact_fleet.ships[ship_type] += 1

    return compact_fleet

cpdef Fleet fleet_from_compact_fleet(CompactFleet compact_fleet, Armada armada):
    """
    Convert a compact fleet to a fleet.

    Parameters
    ----------
    compact_fleet : CompactFleet
        The compact fleet to convert.
    armada : Armada
        The armada to use for the conversion.

    Returns
    -------
    Fleet
        The converted fleet.
    """

    # Create a new fleet with the number of ships in the compact fleet
    cdef Fleet fleet = Fleet(compact_fleet.nb_ships())
    fleet.initialize()

    cdef int ship_type_idx, ship_idx
    ship_idx = 0

    # Fill the fleet with ships from the armada
    for ship_type_idx in range(compact_fleet.nb_ship_types):
        for _ in range(compact_fleet.ships[ship_type_idx]):
            # Copy the ship type 
            fleet.ships[ship_idx * SHIP_CHAR_SIZE] = ship_type_idx
            
            # Copy the ship characteristics from the armada to the fleet
            for ship_char in range(1, SHIP_CHAR_SIZE):
                fleet.ships[ship_idx * SHIP_CHAR_SIZE + ship_char] = armada.ships[ship_type_idx * (SHIP_CHAR_SIZE - 1) + ship_char - 1]
            ship_idx += 1

    return fleet



cpdef void fight_fleets(FightField fight_field):
    """
    Simulate inplace a fight between two fleets. After the function call,
    the fleets are modified to reflect the fight.

    Parameters
    ----------
    fleet_1 : int[:, ::1]
        The first fleet.

    fleet_2 : int[:, ::1]
        The second fleet.

    
    """

    cdef int rounds = 0

    while rounds < fight_field.max_rounds and not (fight_field.fleet_1.is_empty() or fight_field.fleet_2.is_empty()):
        # Restore all shields
        restore_shields(fight_field.fleet_1)
        restore_shields(fight_field.fleet_2)

        # Simulate a single round of fight
        fight_single_round(fight_field)

        rounds += 1

cpdef void remove_destroyed_ships(Fleet fleet):
    """
    Remove inplace the destroyed ships from the fleet.

    Parameters
    ----------
    fleet : Fleet
        The fleet of ships with optionally destroyed ships in its midst to remove
    """
    cdef int intact_ships = 0
    cdef int current_ship, ship_char

    # Move the intact ships to the start of the fleet
    for current_ship in range(fleet.nb_ships):
        if fleet.ships[current_ship * SHIP_CHAR_SIZE + HULL_IDX] > 0:
            # Move the intact ship to the start of the fleet
            for ship_char in range(SHIP_CHAR_SIZE):
                fleet.ships[intact_ships * SHIP_CHAR_SIZE + ship_char] = fleet.ships[current_ship * SHIP_CHAR_SIZE + ship_char]
                
            intact_ships += 1

    fleet.nb_ships = intact_ships

    # Resize the fleet to remove the destroyed ships
    fleet.ships = <DTYPE_t *> realloc(fleet.ships, fleet.nb_ships * SHIP_CHAR_SIZE * sizeof(DTYPE_t))

cpdef void restore_shields(Fleet fleet):
    """
    Restore the shields of the fleet.

    Parameters
    ----------
    fleet : int[:, ::1]
        The fleet to restore the shields of.
    """
    cdef int ship_idx

    # Restore the shields of each ship in the fleet to its shield generator value
    for ship_idx in range(fleet.nb_ships):
        fleet.ships[SHIP_CHAR_SIZE * ship_idx + SHIELD_IDX] = fleet.ships[SHIP_CHAR_SIZE * ship_idx + SHIELD_GENERATOR_IDX]

cpdef void fight_single_round(FightField fight_field):
    """
    Simulate a single round of fight between two fleets.
    The function returns the remaining fleet after the fight.

    Parameters
    ----------
    fleet_1 : int[:, ::1]
        The first fleet.
    fleet_2 : int[:, ::1]
        The second fleet.
    """
    cdef Fleet fleet_1 = fight_field.fleet_1
    cdef Fleet fleet_2 = fight_field.fleet_2

    # Each fleet attacks the other fleet
    fight_one_way(fleet_1, fleet_2)
    fight_one_way(fleet_2, fleet_1)

    # Remove destroyed ships from both fleets
    remove_destroyed_ships(fleet_1)
    remove_destroyed_ships(fleet_2)

cpdef void fight_one_way(Fleet fleet_attacker, Fleet fleet_target):
    """
    Simulate inplace a one-way fight between two fleets.
    
    Parameters
    ----------
    fleet_attacker : Fleet
        The attacking fleet.

    fleet_target : Fleet
        The defending fleet.
    """
    cdef int attacker_idx, target_idx
    cdef int nb_ships_attacker, nb_ships_target
    cdef DTYPE_t damage, shielded_damage, unshielded_damage, shield_value

    nb_ships_attacker = fleet_attacker.nb_ships
    nb_ships_target = fleet_target.nb_ships

    target_idxes = np.random.randint(0, nb_ships_target,size=nb_ships_attacker)

    # Each ship fom the first fleet attacks a random ship of the second fleet
    for attacker_idx in range(nb_ships_attacker):
        # Select a random target from the second fleet
        target_idx = target_idxes[attacker_idx]

        # Calculate damage based on the attack value of the attacker and the shield value of the target
        damage = fleet_attacker.ships[SHIP_CHAR_SIZE * attacker_idx + ATTACK_IDX]
        shield_value = fleet_target.ships[SHIP_CHAR_SIZE * target_idx + SHIELD_IDX]
        shielded_damage = min(damage, shield_value)
        unshielded_damage = damage - shielded_damage

        # If the damage on the shield is not sufficient, the shots are deflected
        # and the shield is not depleted
        if shielded_damage * DEFLECTION_RATIO > shield_value:
            shielded_damage = 0

        # Apply damage to the target ship
        fleet_target.ships[SHIP_CHAR_SIZE * target_idx + SHIELD_IDX] -= shielded_damage
        fleet_target.ships[SHIP_CHAR_SIZE * target_idx + HULL_IDX] -= unshielded_damage
        fleet_target.ships[SHIP_CHAR_SIZE * target_idx + SHIELD_IDX] = max(0, fleet_target.ships[SHIP_CHAR_SIZE * target_idx + SHIELD_IDX])