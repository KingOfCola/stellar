# -*-coding:utf-8 -*-
"""
@File      :   colors.py
@Time      :   2025/04/24 15:50:45
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Color constants for the reporting module.
"""


from reportlab.lib.colors import Color


class Colors:
    """
    Class to hold color constants for the reporting module.
    """

    # Define color constants
    BLUE = Color(0.19, 0.49, 1, 1)
    BLACK = Color(0.2, 0.199, 0.199, 1)
    WHITE = Color(1, 1, 1, 1)
    LIGHT_GRAY = Color(0.8, 0.8, 0.8, 1)
    DARK_GRAY = Color(0.5, 0.5, 0.5, 1)
    CHARCOAL = Color(0.071, 0.090, 0.110, 1)
    ANTHRACITE = Color(0.051, 0.063, 0.078, 1)
    GOLD = Color(1, 0.843, 0, 1)
    SILVER = Color(0.75, 0.75, 0.75, 1)
