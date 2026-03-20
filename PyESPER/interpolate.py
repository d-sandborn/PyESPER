import numpy as np
from scipy.interpolate import RegularGridInterpolator


def interpolate(Path, Gdf={}, AAdata={}, Elsedata={}):
    """
    This LIR function performs the interpolation on user-defined data

    Inputs:
        Gdf: Dictionary of pre-trained data for ESPER v1 (processed)
        AAdata: Dictionary of user input for Atlantic or Arctic
        Elsedata: Dictionary of user input not for Atlantic/Arctic

    Outputs:
        aaLCs: List of points to be interpolated within the Atlantic or Arctic
            regions
        aaInterpolants_pre: Scipy interpolant for Atlantic/Arctic region
        elLCs: List of points to be inteprolated outside of Atlantic/Arctic
        elInterpolants_pre: Scipy interpolant for outside of Atlantic/Arctic
    """
    print("Performing local interpolation.")
    Gvalues = list(Gdf.values())
    grid = Gvalues[0]

    longitude_unique = np.sort(np.unique(grid["lon"]))
    latitude_unique = np.sort(np.unique(grid["lat"]))
    depth_unique = np.sort(np.unique(grid["d2d"]))

    cols = ["C_alpha", "C_S", "C_T", "C_A", "C_B", "C_C"]
    number_of_equations = len(Gvalues)
    number_of_coefficients = len(cols)

    grid_shape = (
        len(longitude_unique),
        len(latitude_unique),
        len(depth_unique),
    )

    values_array = np.empty(
        (*grid_shape, number_of_equations, number_of_coefficients),
        dtype=np.float64,
    )

    for i, grid_entry in enumerate(Gvalues):
        for j, name in enumerate(cols):
            data_series = grid_entry[name]
            raw_values = (
                data_series.values
                if hasattr(data_series, "values")
                else data_series
            )
            values_array[..., i, j] = np.ascontiguousarray(raw_values).reshape(
                grid_shape
            )

    interpolant = RegularGridInterpolator(
        (longitude_unique, latitude_unique, depth_unique),
        values_array,
        bounds_error=False,
        fill_value=np.nan,
    )

    def process_grid(data_values):
        if not data_values:
            return [], interpolant
        first_key = list(data_values.keys())[0]

        longitude_array = data_values[first_key]["Longitude"]
        latitude_array = data_values[first_key]["Latitude"]
        depth_array = data_values[first_key]["d2d"]

        points_to_interpolate = np.column_stack(
            [longitude_array, latitude_array, depth_array]
        )
        raw_results = interpolant(points_to_interpolate)
        results = [raw_results[:, i, :] for i in range(number_of_equations)]

        return results, interpolant

    aaLCs, aaInterpolants_pre = process_grid(AAdata)
    elLCs, elInterpolants_pre = process_grid(Elsedata)

    return aaLCs, aaInterpolants_pre, elLCs, elInterpolants_pre
