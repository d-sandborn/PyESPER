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

    import numpy as np
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay
    from os.path import join
    import pickle

    # Obtain data from the dictionaries
    Gvalues = list(Gdf.values())
    AAOvalues, ElseOvalues = list(AAdata.values()), list(Elsedata.values())
    
    try: #try to load pickled Delauney triangulation
        with open(join(Path, "PyESPER/tri.pkl"), "rb") as f:
            tri = pickle.load(f)
    except FileNotFoundError: #otherwise make it and save it now
        grid = list(Gdf.values())[0]
        points = np.array(
            [list(grid["lon"]), list(grid["lat"]), list(grid["d2d"])]
        ).T
        tri = Delaunay(points)
        with open(join(Path, "PyESPER/tri.pkl"), "wb") as f:
            pickle.dump(tri, f) 
            
    cols = ["C_alpha", "C_S", "C_T", "C_A", "C_B", "C_C"]    
    all_v = [np.column_stack([grid[c] for c in cols]) for grid in Gvalues]
    values = np.stack(all_v, axis=1)
        
    interpolant = LinearNDInterpolator(tri, values)

    def process_grid(grid_values, data_values):
        """
        A function to help process data from grid and user data for interpolations
            and interpolate based upon a Delaunay triangulation, using scipy's
            LinearNDInterpolator
        """
        
        data = data_values[0]
        points_to_interpolate = np.column_stack([
            data["Longitude"],
            data["Latitude"],
            data["d2d"],
        ])
        raw_results = interpolant(points_to_interpolate)
        results = [raw_results[:, i, :] for i in range(len(grid_values))]               

        return results, interpolant
    

    # Process AA (Atlantic/Arctic) and Else grids
    aaLCs, aaInterpolants_pre = process_grid(Gvalues, AAOvalues)
    elLCs, elInterpolants_pre = process_grid(Gvalues, ElseOvalues)
    
    return aaLCs, aaInterpolants_pre, elLCs, elInterpolants_pre
