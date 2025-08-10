import numpy as np
import pandas as pd
import csv
import sys
import time
import random
from mdance.tools import bts

def diversity_selection(matrix, percentage: int, metric, N_atoms=1, 
                        method='strat', start='medoid'):
    """*O(N)* method of selecting the most diverse subset of a data 
    matrix using the complementary similarity. 
    
    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        A feature array.
    percentage : int
        If ``method='strat'``, percentage indicates how many bins of stratified 
        data will be generated. If ``method='comp_sim'``, percentage indicates the
        percentage of data to be selected.
    metric : str, default='MSD'
        The metric to when calculating distance between *n* objects in an array. 
        It must be an options allowed by :func:`mdance.tools.bts.extended_comparison`.
    N_atoms : int, default=1
        Number of atoms in the system used for normalization.
        ``N_atoms=1`` for non-Molecular Dynamics datasets.
    method : {'strat', 'comp_sim'}, default='strat'
        The method to use for diversity selection. ``strat``: stratified
        sampling. ``comp_sim``: maximizing the MSD between the selected
        objects and the rest of the data.
    start : {'medoid', 'outlier', 'random', list}, default='medoid'
        The initial seed for initiating diversity selection. Either 
        from one of the options or a list of indices are valid inputs.

    Raises
    ------
    ValueError
        If ``start`` is not ``medoid``, ``outlier``, ``random``, or a list.
    ValueError
        If ``percentage`` is too high.
    
    Returns
    -------
    list
        List of indices of the diversity selected data.

    Examples
    --------
    >>> from mdance.tools import bts
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [2, 9], [1, 8], [2, 7]])
    >>> bts.diversity_selection(X, percentage=30, metric='MSD', N_atoms=1)
    [7 4]
    """
    n_total = len(matrix)
    n_max = int(np.floor(n_total * percentage / 100))
    if n_max > n_total:
        raise ValueError('Percentage is too high for the given matrix size')
    
    if method == 'strat':
        if n_max < 1:
            raise ValueError('Percentage is too low for the given matrix size')
        if n_max == 1:
            indices_to_select = [0]
        else:
            step = (n_total - 1) / (n_max - 1)
            indices_to_select = np.round(np.arange(n_max) * step).astype(int)
            indices_to_select[0] = 0
        comp_sims = bts.calculate_comp_sim(matrix, metric=metric, N_atoms=N_atoms)
        sorted_comps = np.argsort(-comp_sims, kind='stable')
        selected_n = sorted_comps[indices_to_select].tolist()
        
    elif method == 'comp_sim':
        total_indices = np.array(range(n_total))
        
        if start =='medoid':
            seed = bts.calculate_medoid(matrix, metric=metric, N_atoms=N_atoms)
            selected_n = [seed]
        elif start == 'outlier':
            seed = bts.calculate_outlier(matrix, metric=metric, N_atoms=N_atoms)
            selected_n = [seed]
        elif start == 'random':
            seed = random.randint(0, n_total - 1)
            selected_n = [seed]
        elif isinstance(start, list):
            selected_n = start
        else:
            raise ValueError('Select a correct starting point: medoid, outlier, \
                            random or outlier')

        n = len(selected_n)
        
        selection = [matrix[i] for i in selected_n] 
        selection = np.array(selection)
        selected_condensed = np.sum(selection, axis=0)
        if metric == 'MSD':
            sq_selection = selection ** 2
            sq_selected_condensed = np.sum(sq_selection, axis=0)
        
        while len(selected_n) < n_max:
            select_from_n = np.delete(total_indices, selected_n)
            if metric == 'MSD':
                new_index_n = get_new_index_n(matrix, metric=metric, 
                                            selected_condensed=selected_condensed,
                                            sq_selected_condensed=sq_selected_condensed, 
                                            n=n, select_from_n=select_from_n, 
                                            N_atoms=N_atoms)
                sq_selected_condensed += matrix[new_index_n] ** 2
            else:
                new_index_n = get_new_index_n(matrix, metric=metric, 
                                            selected_condensed=selected_condensed, 
                                            n=n, select_from_n=select_from_n)
            selected_condensed += matrix[new_index_n]
            selected_n.append(new_index_n)
            n = len(selected_n)
    else:
        raise ValueError('Select a correct sampling method: strat or comp_sim')
    return selected_n


def get_new_index_n(matrix, metric, selected_condensed, n, select_from_n, **kwargs):
    """Extract the new index to add to the list of selected indices.
    
    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        A feature array.
    metric : str, default='MSD'
        The metric to when calculating distance between *n* objects in an array. 
        It must be an options allowed by :func:`mdance.tools.bts.extended_comparison`.
    selected_condensed : array-like of shape (n_features,)
        Condensed sum of the selected fingerprints.
    n : int
        Number of selected objects.
    select_from_n : array-like of shape (n_samples,)
        Array of indices to select from. 
    sq_selected_condensed : array-like of shape (n_features,), optional
        Condensed sum of the squared selected fingerprints. (**kwargs)
    N_atoms : int, optional
        Number of atoms in the system used for normalization.
        ``N_atoms=1`` for non-Molecular Dynamics datasets. (**kwargs)
    
    Returns
    -------
    int
        index of the new fingerprint to add to the selected indices.
    """
    if 'sq_selected_condensed' in kwargs:
        sq_selected_condensed = kwargs['sq_selected_condensed']
    if 'N_atoms' in kwargs:
        N_atoms = kwargs['N_atoms']
    
    # Number of fingerprints already selected and the new one to add
    n_total = n + 1
    
    # Placeholders values
    min_value = -1
    index = len(matrix) + 1
    
    # Calculate MSD for each unselected object and select the index with the highest value.
    for i in select_from_n:
        if metric == 'MSD':
            sim_index = bts.extended_comparison([selected_condensed + matrix[i], sq_selected_condensed + (matrix[i] ** 2)],
                                            data_type='condensed', metric=metric, 
                                            N=n_total, N_atoms=N_atoms)
        else:
            sim_index = bts.extended_comparison([selected_condensed + matrix[i]], 
                                            data_type='condensed', 
                                            metric=metric, N=n_total)
        if sim_index > min_value:
            min_value = sim_index
            index = i
        else:
            pass
    return index


def generate_matrix(rows, cols, filename="testData.csv"):
    """Generate a matrix of random floats and save to CSV."""
    matrix = np.random.rand(rows, cols).astype(np.float32)  # Generate random float32 values
    numpy_to_csv(matrix, filename)
    return matrix


def numpy_to_csv(matrix, filename):
    with open("data/"+filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(matrix)
    print("\n" + filename + "\n----------------------")
    print("\n" + filename + "\n----------------------", file=sys.stderr)

def csv_to_numpy(filename="testData.csv"):
    """
    Reads a CSV file and converts it into a NumPy array.

    :param file_path: Path to the CSV file
    :return: NumPy array of the CSV data
    """
    try:
        df = pd.read_csv("data/"+filename, header=None)  # Read CSV using pandas
        np_array = df.to_numpy()  # Convert DataFrame to NumPy array
        print("\n" + filename + "\n----------------------")
        print("\n" + filename + "\n----------------------", file=sys.stderr)
        return np_array
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def str_parse(val):
    # if round(val) - val > -0.00005 and round(val) - val < 0.00005:
    #     return str(round(val))
    return str.format('{0:.10g}', val)

def print_matrix(mat):
    out = "[ "
    for i in range(len(mat)):
        for j in range(len(mat[i])-1):
            out+= str_parse(mat[i][j]) + ", "
        if i == len(mat)-1:
            out += str_parse(mat[i][-1]) + " ]"
        else:
            out += str_parse(mat[i][-1]) + "; "
    return out

def print_vector(vec):
    out = "[ "
    for i in range(len(vec)-1):
        out+= str_parse(vec[i]) + ", "
    out += str_parse(vec[-1]) + " ]"
    return out

def run_tests(matrix, N_atoms):
    # MSD test
    t_start = time.perf_counter()
    msd = bts.mean_sq_dev(matrix, N_atoms=N_atoms)
    t_end = time.perf_counter()
    t_dif = t_end-t_start
    print("msd: " + str.format('{0:.10g}', float(msd)))
    print("msd: " + str.format('{0:.10g}', t_dif), file=sys.stderr)

    # EC test
    t_start = time.perf_counter()
    ec = bts.extended_comparison(matrix, 'full', 'MSD', N_atoms=N_atoms)
    t_end = time.perf_counter()
    t_dif = t_end-t_start
    print("ec: " + str.format('{0:.10g}', float(ec)))
    print("ec: " + str.format('{0:.10g}', t_dif), file=sys.stderr)

    # condensed EC test
    N = len(matrix)
    c_sum = np.sum(matrix, axis=0)
    sq_sum = np.sum(matrix ** 2, axis=0)
    t_start = time.perf_counter()
    ec = bts.extended_comparison((c_sum, sq_sum), 'condensed', 'MSD', N, N_atoms)
    t_end = time.perf_counter()
    t_dif = t_end-t_start
    print("condensed ec: " + str.format('{0:.10g}', float(ec)))
    print("condensed ec: " + str.format('{0:.10g}', t_dif), file=sys.stderr)

    # Esim EC test
    t_start = time.perf_counter()
    ec = bts.extended_comparison([c_sum], 'condensed', metric='RR', N=N, c_threshold=None, w_factor='fraction')
    t_end = time.perf_counter()
    t_dif = t_end-t_start
    print("esim ec: " + str.format('{0:.10g}', float(ec)))
    print("esim ec: " + str.format('{0:.10g}', t_dif), file=sys.stderr)
    
    # calcMedoid test
    t_start = time.perf_counter()
    idx = bts.calculate_medoid(matrix, 'MSD', N_atoms=N_atoms)
    t_end = time.perf_counter()
    t_dif = t_end-t_start
    print("calcMedoid: " + str(idx))
    print("calcMedoid: " + str.format('{0:.10g}', t_dif), file=sys.stderr)

    # calcOutlier test
    t_start = time.perf_counter()
    idx = bts.calculate_outlier(matrix, 'MSD', N_atoms=N_atoms)
    t_end = time.perf_counter()
    t_dif = t_end-t_start
    print("calcOutlier: " + str(idx))
    print("calcOutlier: " + str.format('{0:.10g}', t_dif), file=sys.stderr)

    # trimOutlier test
    t_start = time.perf_counter()
    var = bts.trim_outliers(matrix, 0.1, 'MSD', N_atoms=N_atoms)
    t_end = time.perf_counter()
    t_dif = t_end-t_start
    print("trimOutlier: " + print_matrix(var))
    print("trimOutlier: " + str.format('{0:.10g}', t_dif), file=sys.stderr)

    # compSim tests
    mts = {0: 'MSD', 1: 'BUB', 2: 'Fai', 3: 'Gle', 4: 'Ja', 5: 'JT', 6: 'RT', 7: 'RR', 8: 'SM', 9: 'SS1', 10: 'SS2'}
    for mt in range(11):
        t_start = time.perf_counter()
        val = bts.calculate_comp_sim(matrix, mts[mt], N_atoms=N_atoms)
        t_end = time.perf_counter()
        t_dif = t_end-t_start
        print(mts[mt], "compSim: " + print_vector(val))
        print(mts[mt], "compSim: " + str.format('{0:.10g}', t_dif), file=sys.stderr)

    # diversitySelection tests
    t_start = time.perf_counter()
    var = diversity_selection(matrix, 40, metric='MSD', N_atoms=N_atoms)
    t_end = time.perf_counter()
    t_dif = t_end-t_start
    print("stratified", "diversitySelection: " + print_vector(var))
    print("stratified", "diversitySelection: " + str.format('{0:.10g}', t_dif), file=sys.stderr)

    starts = {0: 'medoid', 1: 'outlier', 2: 'random'}
    for mt in range(1):
        for start in range(2):
            t_start = time.perf_counter()
            var = diversity_selection(matrix, 40, mts[mt], N_atoms=N_atoms, method='comp_sim', start=starts[start])
            t_end = time.perf_counter()
            t_dif = t_end-t_start
            print(mts[mt], starts[start], "diversitySelection: " + print_vector(var))
            print(mts[mt], starts[start], "diversitySelection: " + str.format('{0:.10g}', t_dif), file=sys.stderr)

    t_start = time.perf_counter()
    var = diversity_selection(matrix, 30, 'MSD', N_atoms=N_atoms, method='comp_sim', start=[0, 2])
    t_end = time.perf_counter()
    t_dif = t_end-t_start
    print("list", "diversitySelection: " + print_vector(var))
    print("list", "diversitySelection: " + str.format('{0:.10g}', t_dif), file=sys.stderr)



# Run tests
matrix = csv_to_numpy("bit.csv")
run_tests(matrix, 1)
matrix = csv_to_numpy("continuous.csv")
run_tests(matrix, 2)
matrix = generate_matrix(10, 20, "small.csv")
# matrix = csv_to_numpy("small.csv")
run_tests(matrix, 3)
matrix = generate_matrix(200, 500, "mid.csv")
# matrix = csv_to_numpy("mid.csv")
run_tests(matrix, 3)
matrix = csv_to_numpy("sim.csv")
run_tests(matrix, 50)
# matrix = csv_to_numpy("1d.csv")
# run_tests(matrix, 1)
# matrix = generate_matrix(50000, 500, "large.csv")
# # matrix = csv_to_numpy("large.csv")
# run_tests(matrix, 3)

