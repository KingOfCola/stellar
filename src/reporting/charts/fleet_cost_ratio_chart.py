# -*-coding:utf-8 -*-
"""
@File      :   fleet_cost_ratio_chart.py
@Time      :   2025/04/24 16:04:52
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Chart for the fleet cost ratio.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reporting.consts.colors import Colors
from reporting.charts.base_chart import BaseChart, SvgFlowable


class FleetCostRatioChart(BaseChart):
    """
    Class to create a fleet cost ratio chart.
    """

    def __init__(self, **kwargs):
        """
        Initialize the FleetCostRatioChart.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the fleet cost ratio data.
        **kwargs : keyword arguments
            Additional keyword arguments for the BaseChart class.
        """
        super().__init__(nrows=1, ncols=1, **kwargs)
        self.x_values = np.full(1, np.nan)
        self.y_values = np.full((2, 1), np.nan)
        self.labels = ["Attacker", "Defender"]
        self.colors = [Colors.GOLD.rgba(), Colors.SILVER.rgba()]

    def plot(self):
        """
        Plot the fleet cost ratio chart.
        """
        self.fig.patch.set_facecolor(Colors.ANTHRACITE.rgba())
        ax = self.axes
        ax.set_facecolor(Colors.ANTHRACITE.rgba())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(Colors.WHITE.rgba())
        ax.spines["bottom"].set_color(Colors.WHITE.rgba())
        ax.tick_params(axis="x", colors=Colors.WHITE.rgba())
        ax.tick_params(axis="y", colors=Colors.WHITE.rgba())

        for i, (label, color) in enumerate(zip(self.labels, self.colors)):
            ax.plot(
                self.x_values,
                self.y_values[i],
                color=color,
                linewidth=2,
                label=label,
            )
        ax.set_xlabel("Fleet Cost Ratio")
        ax.set_ylabel("Normalized Cost")
        ax.grid(
            color=Colors.LIGHT_GRAY.rgba(), linestyle="--", linewidth=0.5, which="both"
        )
        ax.legend(loc="upper right", fontsize=10, frameon=False)
        ax.set_xscale("log")
