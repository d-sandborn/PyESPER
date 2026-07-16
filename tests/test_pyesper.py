import pytest
import numpy as np
import pathlib
from PyESPER.lir import lir
from PyESPER.nn import nn
from PyESPER.mixed import mixed

OutputCoordinates = {
    "longitude": [0, -140, 67],
    "latitude": [0, -30, -40],
    "depth": [0, 1000, 100],
}

PredictorMeasurements = {
    "salinity": [35, 33, 32],
    "temperature": [5, 2, 0],
    "phosphate": [0.5, 0.6, 0.4],
    "nitrate": [9, 8, 10],
    "silicate": [21, 20, 19],
    "oxygen": [198, 215, 200],
}
EstDates = [1980, 2002, 2030]
Path = str(pathlib.Path(__file__).resolve().parent.parent / "Mat_fullgrid")


def test_dummy():
    """Are internal tests working?"""
    assert 1 == 1


def test_esper_lir():
    """Is PyESPER giving sufficiently-identical results to ESPER? (lir)"""
    EstimatesLIR1, CoefficientsLIR1, UncertaintiesLIR1 = lir(
        ["TA"],
        Path,
        OutputCoordinates,
        PredictorMeasurements,
        EstDates=EstDates,
        Equations=[2],
    )
    # assert np.isclose(EstimatesLIR1, 47.7868563, atol = 1e-3)


def test_esper_nn():
    """Is PyESPER giving sufficiently-identical results to ESPER? (nn)"""
    EstimatesNN1, UncertaintiesNN1 = nn(
        ["pH"],
        Path,
        OutputCoordinates,
        PredictorMeasurements,
        EstDates=EstDates,
        Equations=[5],
    )
    # assert np.isclose(EstimatesNN1, 47.7868563, atol = 1e-3)


def test_esper_mixed():
    """Is PyESPER giving sufficiently-identical results to ESPER? (mixed)"""
    EstimatesMixed1, UncertaintiesMixed1 = mixed(
        ["phosphate", "nitrate"],
        Path,
        OutputCoordinates,
        PredictorMeasurements,
        EstDates=EstDates,
        Equations=[1, 16],
    )
    # assert np.isclose(EstimatesMixed1, 47.7868563, atol = 1e-3)
