"""EXTENSION POINT for fire-spread / impact scenarios. NOTHING is implemented here on purpose.

A defensible spread model is a different scientific problem from occurrence-risk classification. It needs physical inputs this
project does not ingest, and validation against observed fire perimeters. A slider on top of the risk classifier is NOT a
spread simulator and must not be presented as one."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

REQUIRED_INPUTS = {
    "fuel_model_raster": "gridded fuel type/load (e.g. LANDFIRE-style fuel models) with documented CRS and resolution",
    "terrain": "DEM-derived slope and aspect at the simulation resolution",
    "wind_field": "time-varying wind speed/direction (observed or forecast) at the fire location",
    "fuel_moisture": "dead and live fuel moisture (observed, modelled, or documented assumptions)",
    "ignition": "ignition point or an observed fire perimeter at a known time (UTC)",
}
VALIDATION_REQUIREMENTS = (
    "Compare simulated perimeters/arrival times with independent observed perimeters (e.g. agency GIS perimeters or "
    "time-resolved active-fire detections) over several fires, report burned-area overlap (Jaccard/Sorensen), "
    "arrival-time error and bias, and do not tune on the evaluation fires.")


@dataclass
class SpreadRequest:
    inputs: dict = field(default_factory=dict)

    def missing(self):
        return [k for k in REQUIRED_INPUTS if k not in self.inputs]


class SpreadModel(ABC):
    name = "abstract"
    assumptions: tuple = ()

    @abstractmethod
    def simulate(self, request: SpreadRequest, hours: float):
        """Return a perimeter/arrival-time raster and a record of the assumptions used."""


def available_models():
    return []  # none implemented; register validated models here (e.g. an Rothermel-based cellular model) in the future
