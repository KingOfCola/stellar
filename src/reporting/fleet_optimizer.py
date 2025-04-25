import os
from reportlab.pdfgen import canvas
from rlextra.rml2pdf import rml2pdf
import preppy
import numpy as np

from utils.paths import output, data
from fight.fleets import RELATIVE_VALUES


class FleetOptimizationData:
    """
    Class to handle fleet optimization data.
    """

    def __init__(
        self,
        attacker_ship_counts,
        defender_ship_counts,
        attacker_costs_mean=None,
        defender_costs_mean=None,
        attacker_costs_std=None,
        defender_costs_std=None,
        fleet_cost_ratios=None,
        attacker_losses_by_ratio=None,
        defender_losses_by_ratio=None,
        resources_value=RELATIVE_VALUES,
    ):
        self.attacker_ship_counts = attacker_ship_counts
        self.defender_ship_counts = defender_ship_counts

        self.attacker_costs_mean = (
            attacker_costs_mean
            if attacker_costs_mean is not None
            else np.full(3, np.nan)
        )
        self.defender_costs_mean = (
            defender_costs_mean
            if defender_costs_mean is not None
            else np.full(3, np.nan)
        )
        self.attacker_costs_std = (
            attacker_costs_std if attacker_costs_std is not None else np.full(3, np.nan)
        )
        self.defender_costs_std = (
            defender_costs_std if defender_costs_std is not None else np.full(3, np.nan)
        )

        self.resources_value = resources_value

        # Stores the fleet losses by value ratio
        self.fleet_cost_ratios = (
            fleet_cost_ratios if fleet_cost_ratios is not None else np.full(3, np.nan)
        )
        self.attacker_losses_by_ratio = (
            attacker_losses_by_ratio
            if attacker_losses_by_ratio is not None
            else np.full((3, 3), np.nan)
        )
        self.defender_losses_by_ratio = (
            defender_losses_by_ratio
            if defender_losses_by_ratio is not None
            else np.full((3, 3), np.nan)
        )

        # Calculate the normalized losses by ratio
        self.attacker_normalized_losses_by_ratio = (
            self.resources_value @ self.attacker_losses_by_ratio
        )
        self.defender_normalized_losses_by_ratio = (
            self.resources_value @ self.defender_losses_by_ratio
        )


def make_optimization_report(
    fleet_optimization_data: FleetOptimizationData, output_filename: str
):
    """
    Generates a fleet optimization report using the provided data.

    Parameters
    ----------
    fleet_optimization_data : FleetOptimizationData
        The data to be used in the report.
    """
    # create a directory for the report if it doesn't exist
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Template path
    template_path = data("rml/fleet_report_template.prep").absolute().as_posix()

    template = preppy.getModule(template_path)

    rml = template.get(
        fleet_data=fleet_optimization_data,
    )

    rml2pdf.go(rml, output_filename)


if __name__ == "__main__":
    # create a directory for the report if it doesn't exist
    output_filename = output("fleet_report.pdf").absolute().as_posix()
    ratios = np.geomspace(0.2, 5, 11)

    fleet_data = FleetOptimizationData(
        attacker_ship_counts=np.arange(5, 30),
        defender_ship_counts=np.arange(5, 30) + 100,
        attacker_costs_mean=np.random.randint(0, 1000000, 3),
        defender_costs_mean=np.random.randint(0, 1000000, 3),
        attacker_costs_std=np.random.randint(0, 1000, 3),
        defender_costs_std=np.random.randint(0, 1000, 3),
        fleet_cost_ratios=ratios,
        attacker_losses_by_ratio=np.ones((3, 1)) / (1 + ratios[None, :]),
        defender_losses_by_ratio=np.ones((3, 1)) / (1 + 1 / ratios[None, :]),
    )

    make_optimization_report(fleet_data, output_filename)
