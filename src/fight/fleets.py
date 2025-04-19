# -*-coding:utf-8 -*-
"""
@File      :   fleets.py
@Time      :   2025/04/19 15:02:23
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Module importing and containing the armadas of the game.
"""

import numpy as np
import pandas as pd

from utils.paths import data
from fight.fight import Armada


def read_armadas() -> pd.DataFrame:
    """
    Reads the armadas from the CSV file and returns a DataFrame.
    """
    # Reads the aramada ships from the csv
    armada_df = pd.read_csv(data("assets/ships_chars.csv"))
    armada_array = armada_df[
        ["attack_value", "shield_power", "shield_power", "structure_points"]
    ].values.T
    armada_array[
        3, :
    ] //= 10  # Convert structure points to the same scale as the other values

    # Reads the rapid fire data from the csv
    rapid_fire_df = pd.read_csv(data("assets/rapid_fires.csv"))
    rapid_fire_array = rapid_fire_df.values
    print(rapid_fire_array)

    # Wrap the armada and rapid fire data into an Armada object
    armada = Armada.from_array(armada_array, rapid_fire_array)
    return armada


def read_ship_names() -> pd.Series:
    """
    Reads the ship names from the CSV file and returns a Series.
    """
    ships_df = pd.read_csv(data("assets/ships_type_translation.csv"), index_col=1)
    return ships_df["name"]


ARMADA = read_armadas()
SHIP_NAMES = read_ship_names()
