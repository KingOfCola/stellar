# -*-coding:utf-8 -*-
"""
@File      :   rapid_fire_preparation.py
@Time      :   2025/04/19 17:30:25
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Module to prepare the rapid fire data for the game. The rapid fires are
an ability to fire more than once during one turn.
"""
import pandas as pd
import re
import numpy as np
from utils.paths import data

RAPID_FIRE_PATTERN = re.compile(
    r"\s*Feu rapide (?P<direction>de|contre) (?P<name>\w.+\w)\s*: (?P<percentage>[\d,]+%)\s*\((?P<value>\d+)\)"
)
SHIP_PATTERN = re.compile(r"^(?P<name>\w[\w ']*)$")

if __name__ == "__main__":
    # Load the ships names
    ships_df = pd.read_csv(data("assets/ships_type_translation.csv"), index_col=0)
    ships_names = ships_df["type_translation"]

    rapid_fires = np.zeros((len(ships_names), len(ships_names)), dtype=np.int32)

    attacker_ship_name = None
    attacker_ship_type = None

    # Load the rapid fire data
    with open(data("raw/rapid_fires.txt"), "r", encoding="utf-8") as file:
        for line in file.readlines():
            line = line.strip().replace("`", "'")
            if not line:
                continue
            # Check if the line contains a ship name
            ship_match = SHIP_PATTERN.match(line)
            rapid_fire_match = RAPID_FIRE_PATTERN.match(line)

            if rapid_fire_match:
                direction = rapid_fire_match.group("direction")
                target_ship_name = rapid_fire_match.group("name")
                rapid_fire_percentage = rapid_fire_match.group("percentage")
                rapid_fire_value = rapid_fire_match.group("value")

                if target_ship_name not in ships_names.index:
                    print(f"Unknown target ship name: `{target_ship_name}`")
                    continue
                target_ship_type = ships_names[target_ship_name]

                if direction == "contre":
                    rapid_fires[attacker_ship_type, target_ship_type] = int(
                        rapid_fire_value
                    )

            elif ship_match:
                attacker_ship_name = ship_match.group("name")
                # Check if the ship name is in the ships names
                if attacker_ship_name not in ships_names.index:
                    print(f"Unknown attacker ship name: {attacker_ship_name}")
                    continue
                # Get the ship type from the ships names
                attacker_ship_type = ships_names[attacker_ship_name]
            else:
                print(f"Unknown line format: {line}")
                continue

    # Save the rapid fire data
    rapid_fire_df = pd.DataFrame(
        rapid_fires, index=ships_names.values, columns=ships_names.values
    )
    rapid_fire_df.index.name = "attacker"
    rapid_fire_df.columns.name = "target"
    rapid_fire_df.to_csv(data("assets/rapid_fires.csv"), index=True)
