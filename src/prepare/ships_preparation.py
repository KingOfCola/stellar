# -*-coding:utf-8 -*-
"""
@File      :   ships_preparation.py
@Time      :   2025/04/16 18:40:10
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Prepares the ships data for the game.
"""

import pandas as pd

from utils.paths import data

if __name__ == "__main__":
    # Load the ships data
    ships_df = pd.read_csv(data("raw/ships.csv"))

    # Extract the relevant columns and rename them
    ships_chars = ships_df[
        ["Nom", "Points de structure", "Puissance du bouclier", "Valeur d'attaque"]
    ]
    ships_chars = ships_chars.rename(
        columns={
            "Nom": "name",
            "Points de structure": "structure_points",
            "Puissance du bouclier": "shield_power",
            "Valeur d'attaque": "attack_value",
        }
    )

    # Associate an integer ID to each ship type
    ships_type_translation = pd.Series(
        {name: i for i, name in enumerate(ships_chars["name"].values)},
        name="type_translation",
    )
    ships_chars["type"] = ships_chars["name"].map(ships_type_translation)
    ships_chars = ships_chars.drop(columns=["name"])
    ships_chars = ships_chars.astype(
        {"structure_points": int, "shield_power": int, "attack_value": int, "type": int}
    )

    # Save the prepared data
    ships_chars.to_csv(data("assets/ships_chars.csv"), index=False)
    ships_type_translation.to_csv(data("assets/ships_type_translation.csv"), index=True)
